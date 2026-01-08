import simpy
from collections import deque


class Node:
    """
    Event-driven Node:
      - inbox is a lightweight deque, NOT simpy.Store
      - a single process consumes inbox and triggers quorum checker
      - stop_event stops gossip/processing quickly after quorum reached
    """

    def __init__(self, env: simpy.Environment, node_id: int, net, qc, cfg: dict, rng, stop_event: simpy.Event):
        self.env = env
        self.id = int(node_id)
        self.net = net
        self.qc = qc
        self.cfg = cfg
        self.rng = rng
        self.stop_event = stop_event

        # v2.0 Logic Components
        self.hyper_bft = None
        self.bridge_observer = None
        
        # Initialize logic components if enabled in config
        if cfg.get("enable_hyperbft", False):
            from sim.consensus import HyperBFT
            N = int(cfg.get("_N", 1))
            self.hyper_bft = HyperBFT(env, self.id, N, net, cfg)
            self.hyper_bft.set_callbacks(self.broadcast, self.send_direct)
            self.hyper_bft.start()
            
        if cfg.get("enable_bridge_observer", False):
            from sim.bridge import BridgeObserver
            # We assume bridge_consensus is passed or accessible. 
            # For simplicity, let's assume it's attached to cfg or passed in constructor.
            # But the user hint was about async polling.
            # We need the global bridge_consensus object.
            # Let's assume it's in cfg["_bridge_consensus"] injected by runner.
            bc = cfg.get("_bridge_consensus")
            if bc:
                self.bridge_observer = BridgeObserver(env, self.id, cfg, bc)

        self._inbox = deque()
        self._inbox_ev = env.event()

        g = cfg.get("gossip", {})
        self.mode = str(g.get("mode", "gossip"))
        self.k = int(g.get("k", 4))
        self.max_rounds = int(g.get("max_rounds", 6))
        self.round_gap_s = float(g.get("round_gap_s", 0.01))

        b = cfg.get("byzantine", {})
        self.is_byzantine = False
        self.byz_delay_s = float(b.get("delay_s", 0.3))
        self.byz_refuse_prob = float(b.get("refuse_prob", 0.0))

        self.seen_round = None
        self.seen_time = None
        self.seen_hops = None

        env.process(self._consumer())

    def set_byzantine(self, is_b: bool):
        self.is_byzantine = bool(is_b)
        if self.bridge_observer:
            self.bridge_observer.is_byzantine = self.is_byzantine

    def _deliver_to(self, dst: int, payload: dict):
        # We assume cfg["_nodes"] is populated by the runner
        nodes = self.cfg.get("_nodes")
        if nodes and 0 <= dst < len(nodes):
            nodes[dst]._enqueue(payload)

    def _enqueue(self, payload: dict):
        if self.stop_event and self.stop_event.triggered:
            return
        self._inbox.append(payload)
        if not self._inbox_ev.triggered:
            self._inbox_ev.succeed()

    def _consumer(self):
        while True:
            if self.stop_event and self.stop_event.triggered:
                return

            if not self._inbox:
                self._inbox_ev = self.env.event()
                yield (self._inbox_ev | self.stop_event)
                continue

            payload = self._inbox.popleft()
            self._handle(payload)

    def _handle(self, payload: dict):
        if self.stop_event and self.stop_event.triggered:
            return

        kind = payload.get("kind", "")
        # Standardize schema usage
        msg_id = payload.get("msg_id", None)
        rnd = int(payload.get("round", 0))
        hops = int(payload.get("hops", 0))
        t0 = float(payload.get("t0", self.env.now))

        # v2.0 Message Dispatch
        if self.hyper_bft and kind.startswith("CONSENSUS_"):
            self.hyper_bft.receive_message(payload)
            return

        if kind == "PROP":
            if self.seen_time is None:
                self.seen_time = self.env.now
                self.seen_round = rnd
                self.seen_hops = hops

            if msg_id is not None and hasattr(self.qc, "note_proposal_seen"):
                try:
                    self.qc.note_proposal_seen(self.id, int(msg_id), float(self.env.now), int(hops), float(t0))
                except Exception:
                    pass

            if self.is_byzantine:
                if self.byz_refuse_prob > 0.0 and self.rng.random() < self.byz_refuse_prob:
                    return
                if self.byz_delay_s > 0.0:
                    self.env.process(self._delayed_vote(int(msg_id), t0))
                else:
                    self._vote(int(msg_id), t0)
            else:
                self._vote(int(msg_id), t0)

            if self.mode == "gossip" and rnd < self.max_rounds and not self.hyper_bft:
                # Only use legacy gossip if HyperBFT is NOT enabled
                self._gossip_fwd(payload)

        elif kind == "VOTE":
            if msg_id is None:
                return
            voter = payload.get("src", None)
            if voter is None:
                return
            
            # Use 'src' from payload, not self.id
            self.qc.on_vote(int(voter), int(msg_id), float(self.env.now), float(t0))

    def _delayed_vote(self, msg_id: int, t0: float):
        if self.stop_event and self.stop_event.triggered:
            return
        yield self.env.timeout(self.byz_delay_s)
        if self.stop_event and self.stop_event.triggered:
            return
        self._vote(int(msg_id), float(t0))

    def _vote(self, msg_id: int, t0: float):
        if self.stop_event and self.stop_event.triggered:
            return
        vote = {
            "kind": "VOTE",
            "msg_id": int(msg_id),
            "src": self.id,
            "round": 0,
            "hops": 0,
            "t0": float(t0),
        }
        # Votes are unicast to everyone? Or just collection?
        # The current design seems to treat collection as a magic side effect of on_vote 
        # but also sends it out.
        # But wait, send_unicast to WHOM? 
        # The original code did: self.net.send_unicast(self.id, 0, self._deliver_to, vote)
        # This sends it to node 0? That seems to imply node 0 is the collector?
        # ACTUALLY, checking the original code: 
        # -> self.net.send_unicast(self.id, 0, self._deliver_to, vote)
        # This implies Node 0 receives all votes?
        # BUT 'on_vote' is called on self.qc.
        # So the network message seems redundant unless it's modeling the network cost.
        # We should keep it to model network load/latency, but QuorumCollector grabs it via hook?
        # Wait, QuorumCollector is *passed* to the Node. The Node calls qc.on_vote() *when it processes a VOTE message*.
        # So a node must RECEIVE a vote to call qc.on_vote.
        # If I send to 0, only Node 0 calls qc.on_vote.
        # If I am node 1, I verify, send VOTE to 0. Node 0 receives, calls qc.on_vote(src=1).
        # If I am node 0, I verify, send VOTE to 0 (local). Node 0 receives, calls qc.on_vote(src=0).
        # This seems correct for a centralized collector model (or "leader collection").
        
        # Let's target the leader_id if possible, but default to 0.
        target = int(self.cfg.get("leader_id", 0))
        self.net.send_unicast(self.id, target, self._deliver_to, vote)

    def broadcast(self, payload):
        """Wrapper for v2.0 Logic Components to broadcast"""
        # We reuse _gossip_fwd logic or implement direct all-to-all broadcast
        # For small N (<100), all-to-all is fine in sim
        
        # Inject metadata if needed
        if "src" not in payload:
            payload["src"] = self.id
            
        N = int(self.cfg.get("_N", 0))
        for j in range(N):
            # if j == self.id: continue # v2.0 Fix: Allow Loopback for Leader Vote
            self.net.send_unicast(self.id, j, self._deliver_to, payload)

    def send_direct(self, dst, payload):
        """Wrapper for v2.0 Logic Components to send direct message"""
        if "src" not in payload:
            payload["src"] = self.id
        self.net.send_unicast(self.id, dst, self._deliver_to, payload)

    def start_broadcast(self, msg_id: int):
        if self.stop_event and self.stop_event.triggered:
            return
        payload = {
            "kind": "PROP",
            "msg_id": int(msg_id),
            "src": self.id,
            "round": 0,
            "hops": 0,
            "t0": float(self.env.now),
        }
        self._enqueue(payload)
        # Immediately gossip if we are starting it
        if self.mode == "gossip":
            self._gossip_fwd(payload)

    def _gossip_fwd(self, payload: dict):
        if self.stop_event and self.stop_event.triggered:
            return
        rnd = int(payload.get("round", 0))
        hops = int(payload.get("hops", 0))

        N = int(self.cfg.get("_N", 0))
        if N <= 1:
            return

        peers = []
        tries = 0
        while len(peers) < self.k and tries < self.k * 5:
            j = self.rng.randrange(0, N)
            tries += 1
            if j == self.id:
                continue
            if j in peers:
                continue
            peers.append(j)

        next_payload = dict(payload)
        next_payload["round"] = rnd + 1
        next_payload["hops"] = hops + 1

        for j in peers:
            self.net.send_unicast(self.id, j, self._deliver_to, next_payload)
