import simpy
import random
import csv
import sys
import statistics
import math
sys.path.append(".")
from sim.network import Network
from sim.sequencer import Sequencer
from sim.bridge import BridgeConsensus, BridgeObserver

# --- Monkey Patches ---

# 1. Patch Sequencer.submit_tx
def fixed_submit_tx(self, tx_id):
    delay = random.gauss(self.base_delay, self.jitter_std)
    if delay < 0.01: delay = 0.01
    yield self.env.timeout(delay)
    self.history.append((tx_id, self.env.now))
    payload = {
        "tx_id": tx_id,
        "timestamp": self.env.now,
        "kind": "L2_SOFT_FINALIZED"
    }
    self._notify_observers(payload)
Sequencer.submit_tx = fixed_submit_tx

# 2. Patch Sequencer Batcher Loop
def fixed_arbitrum_batcher_loop(self):
    while True:
        yield self.env.timeout(self.batcher_interval)
        current_time = self.env.now
        yield self.env.timeout(self.l1_finality_delay)
        if hasattr(self, "on_hard_finality") and self.on_hard_finality:
            self.on_hard_finality(current_time)
Sequencer._arbitrum_batcher_loop = fixed_arbitrum_batcher_loop

# 3. Patch BridgeObserver
class PatchedBridgeObserver(BridgeObserver):
    def __init__(self, env, node_id, config, consensus_engine):
        super().__init__(env, node_id, config, consensus_engine)
        self.initial_offset = random.uniform(0, self.poll_interval_mu * 2.0)


def run_monte_carlo():
    cfg = {
        "chain": "arbitrum",
        "N": 100,
        "quorum_ratio": 0.67,
        "bridge_poll_interval_mu": 16.0, 
        "bridge_poll_interval_sigma": 1.0, 
        "sequencer_base_delay_s": 0.25,
        "validator_sign_delay": 0.05,
        "batcher_interval_s": 60.0,
        "l1_finality_delay_s": 24.0, 
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
    
    hard_finality_times = [] 
    
    def on_hard_finality(ts):
        hard_finality_times.append(env.now)
    sequencer.on_hard_finality = on_hard_finality
    
    def intercepted_soft_callback(payload):
        idx = payload["tx_id"]
        if idx in tx_records:
            tx_records[idx]["t_soft_ts"] = payload["timestamp"]
        distribute_to_observers(payload)
    sequencer.set_callback(intercepted_soft_callback)
    
    def on_deposit_finalized(tx_id, ts):
        if tx_id in tx_records:
            tx_records[tx_id]["t_final_ts"] = ts
    consensuss.on_deposit_finalized = on_deposit_finalized
    
    # Generate 1000 Txs
    TOTAL_TXS = 1000
    def traffic_gen():
        yield env.timeout(5.0) 
        for i in range(1, TOTAL_TXS + 1):
            tx_id = i
            submit_ts = env.now
            tx_records[tx_id] = {"t_submit_ts": submit_ts}
            env.process(sequencer.submit_tx(tx_id))
            yield env.timeout(rng.uniform(0.1, 0.5))
            
    env.process(traffic_gen())
    
    # Run
    env.run(until=800.0) 
    
    # Process Results
    results = []
    
    consensus_overhead = 1.0
    
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
            
        t_hard_ts = next((t for t in hard_finality_times if t >= rec["t_soft_ts"]), None)
        if t_hard_ts:
            t_hard = t_hard_ts - rec["t_submit_ts"]
        else:
            t_hard = -1 
            
        results.append({
            "tx_id": tx_id,
            "t_soft": t_soft,
            "t_poll": t_poll,
            "t_cons": t_cons,
            "t_total": t_total,
            "t_hard": t_hard
        })

    total_lats = [r["t_total"] for r in results]
    if not total_lats:
        print("No transactions finalized.")
        return

    mean_val = statistics.mean(total_lats)
    stdev_val = statistics.stdev(total_lats)
    min_val = min(total_lats)
    max_val = max(total_lats)
    
    print(f"--- Monte Carlo Results (N={len(total_lats)}) ---")
    print(f"Mean: {mean_val:.4f}s")
    print(f"Stdev: {stdev_val:.4f}s")
    print(f"Min: {min_val:.4f}s")
    print(f"Max: {max_val:.4f}s")
    print("-" * 30)
    
    bins = {}
    for v in total_lats:
        b = math.floor(v)
        bins[b] = bins.get(b, 0) + 1
        
    print("--- Histogram ---")
    min_bin = min(bins.keys())
    max_bin = max(bins.keys())
    
    max_count = max(bins.values())
    scale = 50.0 / max_count if max_count > 0 else 1
    
    for b in range(min_bin, max_bin + 2):
        count = bins.get(b, 0)
        bar_len = int(count * scale)
        bar = "#" * bar_len
        print(f"{b:2d}-{b+1:2d}s | {count:3d} | {bar}")
        
    print("-" * 30)
    
    print("--- CSV Preview (First 20) ---")
    headers = ["tx_id", "t_soft_finality", "t_polling_delay", "t_consensus", "t_total_latency", "t_hard_finality(L1)"]
    print(",".join(headers))
    for r in results[:20]:
        h_str = f"{r['t_hard']:.4f}" if r['t_hard'] > 0 else "-"
        print(f"{r['tx_id']},{r['t_soft']:.4f},{r['t_poll']:.4f},{r['t_cons']:.4f},{r['t_total']:.4f},{h_str}")

if __name__ == "__main__":
    run_monte_carlo()
