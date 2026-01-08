import simpy
import random
from sim.oracle import SafetyOracle

class BridgeObserver:
    """
    Simulates a Hyperliquid Validator's 'Bridge Watcher' process.
    It simulates the polling lag when detecting deposit transactions on Arbitrum.
    """
    def __init__(self, env: simpy.Environment, node_id: int, config: dict, consensus_engine):
        self.env = env
        self.node_id = node_id
        self.consensus_engine = consensus_engine
        self.is_byzantine = False
        
        # Polling interval mean and variation
        self.poll_interval_mu = float(config.get("bridge_poll_interval_mu", 5.0))
        self.poll_interval_sigma = float(config.get("bridge_poll_interval_sigma", 1.0))
        
        # Each validator has a random initial offset to simulate async behavior
        # Use Lognormal to create HEAVY tail (User requirement for long tail)
        # Config: mu and sigma are passed directly to lognormvariate
        self.initial_offset = random.lognormvariate(self.poll_interval_mu, self.poll_interval_sigma)
        
        # Queue of events observed from Sequencer but not yet "polled"
        self._pending_observations = []
        
        # Start the polling loop
        self.env.process(self._polling_loop())

    def on_sequencer_event(self, payload):
        """
        Called when Sequencer emits an event. 
        In reality, this is the data becoming available on L2.
        The validator 'sees' it only at the next poll tick.
        """
        # print(f"[DEBUG] Observer {self.node_id} received seq event")
        self._pending_observations.append(payload)

    def _polling_loop(self):
        # Initial offset
        yield self.env.timeout(self.initial_offset)
        
        while True:
            # Process all pending observations
            while self._pending_observations:
                event = self._pending_observations.pop(0)
                # print(f"[DEBUG] Observer {self.node_id} processing event at {self.env.now}")
                self.env.process(self._process_observed_event(event))

            # Wait for next poll tick
            # Lognormal or Normal distribution for polling jitter?
            # User hint: "Asynchronous polling interval"
            delay = random.gauss(self.poll_interval_mu, self.poll_interval_sigma)
            if delay < 0.1: delay = 0.1
            yield self.env.timeout(delay)

    def _process_observed_event(self, event):
        # When validator sees the deposit, it signs it (Votes)
        # Verify -> Sign -> Broadcast Vote
        
        # Attack Logic: Lazy Validator
        # If is_byzantine, completely ignore the event (Lazy / No Vote)
        # This simulates a node that is online but unresponsive to bridge events
        if self.is_byzantine:
            return

        # In this sim, we send a vote to the consensus engine
        tx_id = event["tx_id"]
        
        # Simulate processing time for verification & signing
        verify_time = random.uniform(0.01, 0.05)
        yield self.env.timeout(verify_time)
        
        self.consensus_engine.process_vote(self.node_id, tx_id)


class BridgeConsensus:
    """
    Manages the aggregation of signatures (Votes) from validators.
    When 2/3 Quorum is reached, the Deposit is considered 'Bridged'.
    """
    def __init__(self, env: simpy.Environment, N: int, config: dict):
        self.env = env
        self.N = N
        self.quorum_size = int(N * 2 / 3) + 1 # Simple 2/3 majority
        self.enable_adaptive_safety = config.get("enable_adaptive_safety", False)
        
        if self.enable_adaptive_safety:
            self.oracle = SafetyOracle(env, config)
            # Inject oracle back to config so Sequencer can find it? 
            # Or better, pass it explicitly. For now, let's keep it here.
            config["_oracle_instance"] = self.oracle
        else:
            self.oracle = None

        self.pending_votes = {} # tx_id -> set(node_ids)
        # tx_id -> completion timestamp
        self.completed_deposits = {}
        
        # Hook for metrics
        self.on_deposit_finalized = None

    def process_vote(self, node_id, tx_id):
        # Simulate vote arrival
        # In this simplified model, we assume vote arrives instantly after processing
        
        # Adaptive Safety Check:
        # Before "counting" the vote towards finality, we might enforce a delay
        # But actually, the delay happens at the "Signing" phase (Node side) or "Finalizing" phase (Consensus side).
        # Let's verify implementation_plan.md: "BridgeConsensus... Delay leads to signing."
        # Actually, in this sim, 'process_vote' effectively means "Node signed and sent vote".
        # So the delay should be in the Node logic? 
        # But BridgeConsensus aggregates.
        
        # Let's implement the delay effect here: "We don't consider it finalized until Oracle says safe".
        # Alternatively, we can add the delay *before* yielding the final event.
        
        if tx_id not in self.pending_votes:
             self.pending_votes[tx_id] = set()
             
        self.pending_votes[tx_id].add(node_id)
        
        if len(self.pending_votes[tx_id]) >= self.quorum_size:
             self.env.process(self._finalize_with_safety(tx_id))

    def _finalize_with_safety(self, tx_id):
         # If already finalized, skip
         if tx_id in self.completed_deposits:
             return

         extra_delay = 0.0
         if self.enable_adaptive_safety and self.oracle:
             # Ask Oracle how much we should have waited
             # Note: real implementation would wait *before* signing. 
             # Here we simulate the *total* time impact.
             extra_delay = self.oracle.get_required_delay()
             
         if extra_delay > 0:
             yield self.env.timeout(extra_delay)
             
         self._finalize(tx_id)

    def _finalize(self, tx_id):
        if tx_id in self.completed_deposits:
            return
            
        self._finalize_deposit(tx_id)

    def _finalize_deposit(self, tx_id):
        # print(f"[DEBUG] Consensus Reached for {tx_id} at {self.env.now}")
        self.completed_deposits[tx_id] = self.env.now
        if self.on_deposit_finalized:
            self.on_deposit_finalized(tx_id, self.env.now)


class BridgeRelayer:
    """
    Simulates the L1 -> L2 Relay process for Withdrawals.
    Includes the 'Dispute Period' which is the dominant factor in withdraw latency.
    """
    def __init__(self, env: simpy.Environment, config: dict):
        self.env = env
        self.dispute_period_s = float(config.get("bridge_dispute_period_s", 200.0)) # ~3-5 mins
        self.relay_delay_mu = float(config.get("bridge_relay_delay_mu", 1.0)) # Messaging delay
        self.relay_delay_sigma = float(config.get("bridge_relay_delay_sigma", 0.1))
        
        self.completed_withdrawals = {}
        self.on_withdraw_finalized = None

    def on_l1_commit(self, tx_id, commit_time):
        """
        Called when HyperBFT commits a withdrawal bundle on L1.
        """
        self.env.process(self._relay_process(tx_id))

    def _relay_process(self, tx_id):
        # 1. Relay Message L1 -> L2 (Arbitrum)
        # Use Lognormal to match the high variance (Std ~12s) observed in real data
        relay_delay = random.lognormvariate(self.relay_delay_mu, self.relay_delay_sigma)
        if relay_delay < 0.1: relay_delay = 0.1
        
        yield self.env.timeout(relay_delay)
        
        # 2. Dispute Period on Arbitrum
        yield self.env.timeout(self.dispute_period_s)
        
        # Finalized
        self._finalize_withdraw(tx_id)

    def _finalize_withdraw(self, tx_id):
        # print(f"[DEBUG] Withdraw Finalized for {tx_id} at {self.env.now}")
        self.completed_withdrawals[tx_id] = self.env.now
        if self.on_withdraw_finalized:
            self.on_withdraw_finalized(tx_id, self.env.now)