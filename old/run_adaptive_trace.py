import simpy
import random
import csv
import time
from sim.sequencer import Sequencer
from sim.node import Node
from sim.bridge import BridgeConsensus, BridgeRelayer

# Configuration for Trace
CONFIG = {
    "N": 21,
    "reorg_mode": "probabilistic",
    "reorg_mtbf_s": 20.0,
    "reorg_depth_avg_s": 5.0,
    "enable_adaptive_safety": True,
    "oracle_sensitivity": 2.0,
    "oracle_decay_rate": 0.05,
    "oracle_high_risk_delay_s": 15.0,
    "oracle_safe_delay_s": 0.0,
    "sequencer_base_delay_s": 0.25,
    "duration_s": 200.0,
    "tx_interval_s": 2.0
}

def run_trace():
    env = simpy.Environment()
    config = CONFIG.copy()
    
    # Components
    sequencer = Sequencer(env, config)
    consensus = BridgeConsensus(env, config["N"], config)
    
    # We don't need full node/network simulation for this trace, 
    # we just want to see "Oracle Risk Score" and "Effective Delay".
    # But to show "Latency", we need the flow.
    # Let's verify: Latency = SoftFinality + OracleDelay.
    # Sequencer emits SoftFinality.
    
    trace_data = [] # (ts, latency, risk_score)
    reorg_events = [] # (ts, depth)
    
    # Hook Reorg
    def on_reorg(ts, depth, invalidated):
        reorg_events.append((ts, depth))
    sequencer.on_reorg = on_reorg
    
    # Traffic Generator
    def traffic_gen():
        tx_id = 0
        while True:
            yield env.timeout(config["tx_interval_s"])
            tx_id += 1
            env.process(process_tx(tx_id))
            
    # Transaction Processor (Mocking Node/Consensus flow)
    def process_tx(tx_id):
        start_time = env.now
        
        # 1. Submit to Sequencer
        sequencer.submit_tx(tx_id)
        
        # 2. Wait for Soft Finality (Sequencer delay)
        # We hook into sequencer history or just wait
        # Sequencer._process_tx does: delay -> history -> event
        # We can't easily await the event without the node logic.
        # Let's simulate valid soft finality time:
        sf_delay = sequencer.base_delay # simplified
        yield env.timeout(sf_delay)
        
        # 3. Oracle Check (Adaptive Delay)
        oracle = consensus.oracle
        risk = oracle.risk_score
        extra_delay = oracle.get_required_delay()
        
        yield env.timeout(extra_delay)
        
        # 4. Finalize
        # Total Latency = now - start
        latency = env.now - start_time
        
        trace_data.append({
            "timestamp": env.now,
            "latency": latency,
            "risk_score": risk
        })

    # Start
    env.process(traffic_gen())
    env.run(until=config["duration_s"])
    
    # Save CSVs
    with open("result/trace_adaptive_tx.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "latency", "risk_score"])
        writer.writeheader()
        writer.writerows(trace_data)
        
    with open("result/trace_adaptive_reorg.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "depth"])
        writer.writerows(reorg_events)
        
    print("Trace generated: result/trace_adaptive_tx.csv, result/trace_adaptive_reorg.csv")

if __name__ == "__main__":
    run_trace()
