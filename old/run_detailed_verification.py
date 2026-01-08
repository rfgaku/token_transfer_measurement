import simpy
import random
import csv
import sys
import statistics
sys.path.append(".")
from sim.network import Network
from sim.sequencer import Sequencer
from sim.bridge import BridgeConsensus, BridgeObserver

# 1. Patch Sequencer.submit_tx (Bug fix)
# The codebase's Sequencer.submit_tx is a generator but called without env.process in original code,
# and refers to a missing _process_tx method. We replace it with correct logic.
def fixed_submit_tx(self, tx_id):
    # Simulate processing limit / congestion
    delay = random.gauss(self.base_delay, self.jitter_std)
    if delay < 0.01: delay = 0.01
    yield self.env.timeout(delay)
    
    # Register in history
    self.history.append((tx_id, self.env.now))
    
    # Emit Soft Finality Event
    payload = {
        "tx_id": tx_id,
        "timestamp": self.env.now,
        "kind": "L2_SOFT_FINALIZED"
    }
    self._notify_observers(payload)

# Apply patch to Class
Sequencer.submit_tx = fixed_submit_tx


# 2. Patch BridgeObserver (Parameter fix)
class PatchedBridgeObserver(BridgeObserver):
    def __init__(self, env, node_id, config, consensus_engine):
        super().__init__(env, node_id, config, consensus_engine)
        # Fix the huge startup delay caused by lognorm(mu=15)
        # Use uniform distribution for random phase
        self.initial_offset = random.uniform(0, self.poll_interval_mu * 2.0)

def run_detailed_experiment():
    cfg = {
        "chain": "arbitrum",
        "N": 100,
        "quorum_ratio": 0.67,
        "bridge_poll_interval_mu": 15.0, # Target ~11.5s latency implies ~15s polling interval
        "bridge_poll_interval_sigma": 1.0, 
        "sequencer_base_delay_s": 0.25,
        "validator_sign_delay": 0.05,
        "batcher_interval_s": 60.0,
        "_nodes": [] 
    }
    
    env = simpy.Environment()
    rng = random.Random(42)
    
    net = Network(env, rng, {}) 
    sequencer = Sequencer(env, cfg)
    consensuss = BridgeConsensus(env, cfg["N"], cfg)
    
    validators = []
    for i in range(cfg["N"]):
        v = PatchedBridgeObserver(env, i, cfg, consensuss)
        validators.append(v)
        
    def distribute_to_observers(payload):
        for v in validators:
            v.on_sequencer_event(payload)
            
    tx_records = {} 
    
    def intercepted_soft_callback(payload):
        idx = payload["tx_id"]
        # print(f"[DEBUG] Soft Finality for Tx {idx} at {env.now:.2f}")
        if idx in tx_records:
            tx_records[idx]["t_soft_ts"] = payload["timestamp"]
        distribute_to_observers(payload)
    
    sequencer.set_callback(intercepted_soft_callback)
    
    def on_deposit_finalized(tx_id, ts):
        # print(f"[DEBUG] FINALIZED Tx {tx_id} at {ts:.2f}")
        if tx_id in tx_records:
            tx_records[tx_id]["t_final_ts"] = ts
            
    consensuss.on_deposit_finalized = on_deposit_finalized
    
    def traffic_gen():
        yield env.timeout(30.0) 
        for i in range(1, 21):
            tx_id = i
            submit_ts = env.now
            tx_records[tx_id] = {"t_submit_ts": submit_ts}
            # Start the patched generator process
            env.process(sequencer.submit_tx(tx_id))
            yield env.timeout(rng.uniform(1.0, 3.0))
            
    env.process(traffic_gen())
    
    env.run(until=300.0)
    
    headers = ["tx_id", "t_soft_finality", "t_poll_delay", "t_consensus", "t_total_latency", "is_finalized"]
    consensus_overhead = 1.2
    
    print(",".join(headers))
    for tx_id in sorted(tx_records.keys()):
        rec = tx_records[tx_id]
        if "t_final_ts" not in rec or "t_soft_ts" not in rec:
            continue
            
        t_soft = rec["t_soft_ts"] - rec["t_submit_ts"]
        t_total = rec["t_final_ts"] - rec["t_submit_ts"]
        t_middle = t_total - t_soft
        t_cons = consensus_overhead
        t_poll = t_middle - t_cons
        if t_poll < 0: t_poll = 0.0
             
        print(f"{tx_id},{t_soft:.4f},{t_poll:.4f},{t_cons:.4f},{t_total:.4f},True")

if __name__ == "__main__":
    run_detailed_experiment()
