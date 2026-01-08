import simpy

class HyperBFT:
    """
    Hyperliquid L1 Consensus (HyperBFT).
    Simplified HotStuff-like consensus:
    - View/Round based
    - Primary (Leader) proposes block
    - Replicas (Validators) vote
    - Commit on Quorum
    """
    def __init__(self, env: simpy.Environment, node_id: int, N: int, network, config: dict):
        self.env = env
        self.node_id = node_id
        self.N = N
        self.network = network
        self.config = config
        
        self.view = 0
        self.quorum_size = int((2 * N / 3) + 1)
        
        # State
        self.current_proposal = None
        self.votes = {} # view -> set of voters
        self.committed_blocks = []
        
        # Timeout for view change (simplified)
        self.view_timeout_s = float(config.get("consensus_view_timeout_s", 1.0))

    def start(self):
        self.env.process(self._replica_loop())
        
    def _replica_loop(self):
        while True:
            is_leader = (self.view % self.N) == self.node_id
            
            if is_leader:
                self._propose_block()
            
            # Wait for view completion or timeout
            # (In a real impl, we'd wait for Quorum Certificate or Timeout)
            # For this sim, we just wait a fixed block time slightly randomized
            block_time = float(self.config.get("consensus_block_time_s", 0.2))
            yield self.env.timeout(block_time)
            
            self.view += 1

    def _propose_block(self):
        # Leader broadcasts proposal
        payload = {
            "kind": "CONSENSUS_PROPOSE",
            "view": self.view,
            "proposer": self.node_id
        }
        # Broadcast to all
        # We assume Node class handles the actual network.send call if we integrate closely,
        # but here we can just use network directly if we had the list of peers.
        # However, Node has the peers. 
        # So HyperBFT should probably emit an event or call a callback on Node.
        if hasattr(self, "on_broadcast_needed"):
            self.on_broadcast_needed(payload)

    def receive_message(self, msg):
        kind = msg.get("kind")
        if kind == "CONSENSUS_PROPOSE":
            self._handle_proposal(msg)
        elif kind == "CONSENSUS_VOTE":
            self._handle_vote(msg)

    def _handle_proposal(self, msg):
        view = msg.get("view")
        if view < self.view:
            return # Old view
        
        # In HotStuff, we check if proposal extends locked approach. 
        # Here simplified: always accept if view matches.
        
        # Reply with Vote
        vote = {
            "kind": "CONSENSUS_VOTE",
            "view": view,
            "voter": self.node_id
        }
        # Send vote back to leader (or broadcast in linear hotstuff)
        # We'll broadcast for simplicity or send to leader
        if hasattr(self, "on_send_direct_needed"):
            leader_id = view % self.N
            self.on_send_direct_needed(leader_id, vote)

    def _handle_vote(self, msg):
        view = msg.get("view")
        if view not in self.votes:
            self.votes[view] = set()
        
        self.votes[view].add(msg.get("voter"))
        
        # print(f"[DEBUG] Node {self.node_id} View {view} Votes: {len(self.votes[view])}/{self.quorum_size}")

        if len(self.votes[view]) >= self.quorum_size:
            if view not in self.committed_blocks:
                self._commit_view(view)

    def _commit_view(self, view):
        # Block finalized
        # print(f"[DEBUG] HyperBFT Committed View {view} at {self.env.now}")
        self.committed_blocks.append(view)
        if hasattr(self, "on_commit") and self.on_commit:
            self.on_commit(view, self.env.now)

    # Integration hooks
    def set_callbacks(self, broadcast_fn, send_direct_fn, commit_fn=None):
        self.on_broadcast_needed = broadcast_fn
        self.on_send_direct_needed = send_direct_fn
        self.on_commit = commit_fn
