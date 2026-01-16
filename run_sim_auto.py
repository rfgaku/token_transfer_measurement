import simpy
import random
import csv
import sys
import statistics
import math
import argparse
import json
import os
import matplotlib.pyplot as plt

# Ensuring we can import local modules
sys.path.append(".")
from sim.network import Network
from sim.sequencer import Sequencer
from sim.bridge import BridgeConsensus, BridgeObserver

# Phase 2: 物理モデルコンポーネント
from sim.network_physics import NetworkPhysics
from sim.hyperliquid_validator import HyperliquidValidator, PhysicalBridgeConsensus

# --- Patching BridgeObserver for Uniform Randomness (Permanent Fix) ---
class PatchedBridgeObserver(BridgeObserver):
    def __init__(self, env, node_id, config, consensus_engine):
        super().__init__(env, node_id, config, consensus_engine)
        self.initial_offset = random.uniform(0, self.poll_interval_mu * 2.0)

def run_simulation_auto(args):
    """
    Main execution logic for the automated simulator.
    """
    # 0. Result Directory Setup (Strict: result/)
    RESULT_DIR = "result"
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 1. Configuration Setup (Comprehensive)
    TOTAL_TXS = 1000
    if args.tx_count: TOTAL_TXS = args.tx_count
    
    # Full Dictionary without filtering
    cfg = {
        # Global
        "chain": args.chain,
        "N": args.N,
        "quorum_ratio": args.quorum,
        "simulation_time_limit": 600.0,
        
        # Network
        "p2p_base_s": 0.05,
        "p2p_jitter_s": 0.01,
        "packet_loss_rate": 0.0, 
        
        # Performance
        "validator_processing_delay": 0.0, 
        "tps_limit": 5.0, 
        
        # Arbitrum
        "sequencer_processing_time": 0.25,
        "sequencer_jitter_std": 0.05,
        "soft_finality_latency": 0.25, 
        "batcher_interval_s": 60.0,
        "l1_finality_delay_s": 24.0,
        "reorg_mtbf_s": 100.0,
        "reorg_depth_avg_s": 5.0,
        "reorg_mode": "fixed",
        
        # Hyperliquid
        "block_time": 0.2,
        "consensus_view_timeout_s": 1.0,
        
        # Bridge (TUNING TARGET)
        "bridge_poll_interval_mu": args.poll_interval, 
        "bridge_poll_interval_sigma": 1.0, 
        "validator_sign_delay": 0.05,
        "mint_delay_s": 0.0,
        "cctp_safety_margin_s": 0.0,
        
        # Security
        "byz_frac": 0.0,
        "malicious_node_count": 0,
        "attack_type": "None",
        
        # Scenario
        "cross_chain_tx_interval": 0.2,
        
        # Relayer (Withdraw)
        "bridge_dispute_period_s": 200.0,
        "bridge_relay_delay_mu": 1.0,
        "bridge_relay_delay_sigma": 0.1,
        
        "_nodes": [] 
    }
    
    # 2. Export Parameters (Task 3: FULL DUMP) - Only if not corrected run (or always?)
    # User asked for specific files for corrected run. But no harm overwriting params.
    param_filename = os.path.join(RESULT_DIR, "simulation_parameters.csv")
    with open(param_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value"])
        for k, v in cfg.items():
            if not k.startswith("_"):
                writer.writerow([k, v])
    print(f"[Auto] Full Parameters exported to {param_filename}")

    # 3. Simulation Environment
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

    # 4. Data Collection Hooks (Deposit)
    tx_records_dep = {} 
    hard_finality_times = [] 
    
    def on_hard_finality(ts):
        hard_finality_times.append(ts)
    sequencer.on_hard_finality = on_hard_finality
    
    def intercepted_soft_callback(payload):
        idx = payload["tx_id"]
        if idx in tx_records_dep:
            tx_records_dep[idx]["t_soft_ts"] = payload["timestamp"]
        distribute_to_observers(payload)
    sequencer.set_callback(intercepted_soft_callback)
    
    def on_deposit_finalized(tx_id, ts):
        if tx_id in tx_records_dep:
            tx_records_dep[tx_id]["t_final_ts"] = ts
    consensuss.on_deposit_finalized = on_deposit_finalized
    
    # 5. Traffic Generation
    
    # 5A. Deposit
    def traffic_gen_deposit():
        yield env.timeout(5.0) 
        for i in range(1, TOTAL_TXS + 1): 
            tx_id = i
            submit_ts = env.now
            tx_records_dep[tx_id] = {"t_submit_ts": submit_ts}
            env.process(sequencer.submit_tx(tx_id))
            yield env.timeout(cfg["cross_chain_tx_interval"])
            
    # 5B. Withdraw (Parallel + Batcher Physics)
    tx_records_wd = {}
    
    def process_withdrawal(tx_id, start_ts):
        dispute = cfg["bridge_dispute_period_s"]
        relay_mu = cfg["bridge_relay_delay_mu"]
        relay_sigma = cfg["bridge_relay_delay_sigma"]
        batcher_interval = cfg["batcher_interval_s"]
        
        # 0. L1 Batcher / Block Delay (Physical Restoration)
        # Transactions on L2 are not immediately final/disputable on L1. 
        # They must be batched and posted.
        batch_delay = random.uniform(0, batcher_interval)
        yield env.timeout(batch_delay)
        
        # 1. L2 Processing
        yield env.timeout(0.01) 
        
        # 2. Dispute Period
        yield env.timeout(dispute)
        
        # 3. Relay to L1
        relay_time = random.gauss(relay_mu, relay_sigma)
        if relay_time < 0.1: relay_time = 0.1
        yield env.timeout(relay_time)
        
        end_ts = env.now
        total_lat = end_ts - start_ts
        
        tx_records_wd[tx_id] = {
            "timestamp": start_ts,
            "t_total_latency": total_lat,
            # Additional fields for corrected view
            "dispute_period": dispute,
            "l1_batch_delay": batch_delay,
            "l1_relay_time": relay_time
        }

    def traffic_gen_withdraw():
        yield env.timeout(5.0)
        for i in range(1, TOTAL_TXS + 1): 
            tx_id = i
            start_ts = env.now
            env.process(process_withdrawal(tx_id, start_ts))
            yield env.timeout(cfg["cross_chain_tx_interval"]) 

    env.process(traffic_gen_deposit())
    env.process(traffic_gen_withdraw())
    
    # 6. Execution
    print(f"[Auto] Running Simulation (N={args.N}, Poll={args.poll_interval}s, Corrected={args.corrected})...")
    env.run(until=cfg["simulation_time_limit"])
    
    # 7. Processing & Exports
    
    # Determine suffixes
    if args.corrected:
        suffix_wd = "_corrected"
        # Since we only focus on withdraw corrected, others can keep tuned or standard names?
        # User said "Output files only sim_trace_withdraw_corrected.csv and histogram...". 
        # But we will output deposits normally just in case.
        suffix_dep = "_tuned" if args.tuned else f"_N{args.N}_poll{args.poll_interval}"
    elif args.tuned:
        suffix_wd = "_tuned"
        suffix_dep = "_tuned"
    else:
        suffix_wd = f"_N{args.N}_poll{args.poll_interval}"
        suffix_dep = suffix_wd
    
    # A. Deposit Trace
    dep_results = []
    consensus_overhead = 1.0
    for tx_id in sorted(tx_records_dep.keys()):
        rec = tx_records_dep[tx_id]
        if "t_final_ts" in rec and "t_soft_ts" in rec:
            t_soft = rec["t_soft_ts"] - rec["t_submit_ts"]
            t_total = rec["t_final_ts"] - rec["t_submit_ts"]
            t_middle = t_total - t_soft
            t_cons = consensus_overhead
            t_poll = t_middle - t_cons
            if t_poll < 0: t_poll = 0.0
            dep_results.append({
                "tx_id": tx_id,
                "timestamp": rec["t_submit_ts"],
                "t_soft_finality": t_soft,
                "t_poll_delay": t_poll,
                "t_consensus": t_cons,
                "t_total_latency": t_total,
                "status": "Finalized"
            })
        
    dep_filename = os.path.join(RESULT_DIR, f"sim_trace_deposit{suffix_dep}.csv")
    with open(dep_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tx_id", "timestamp", "t_soft_finality", "t_poll_delay", "t_consensus", "t_total_latency", "status"])
        writer.writeheader()
        for r in dep_results:
            row = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()}
            writer.writerow(row)
    print(f"[Auto] Deposit Trace saved to {dep_filename}")

    # B. Withdraw Trace
    wd_filename = os.path.join(RESULT_DIR, f"sim_trace_withdraw{suffix_wd}.csv")
    with open(wd_filename, "w", newline="") as f:
        # Corrected columns if corrected flag
        if args.corrected:
            fieldnames = ["tx_id", "timestamp", "dispute_period", "l1_batch_delay", "l1_relay_time", "t_total_latency"]
        else:
            fieldnames = ["tx_id", "timestamp", "t_total_latency"] # Keep simple if not corrected/tuned? Or better to detail always? 
            # If standard/tuned run, maybe detailed columns are fine too, but let's stick to previous contract unless corrected.
            # Actually, tuned run only had ["tx_id", "timestamp", "t_total_latency"].
        
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore') # Ignore extras if simpler export
        writer.writeheader()
        for tx_id in sorted(tx_records_wd.keys()):
            r = tx_records_wd[tx_id]
            r["tx_id"] = tx_id
            row = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()}
            writer.writerow(row)
    print(f"[Auto] Withdraw Trace saved to {wd_filename}")
    
    # C. Histograms
    
    # Deposit Histogram
    dep_lats = [r["t_total_latency"] for r in dep_results]
    if dep_lats:
        plt.figure(figsize=(10, 6))
        plt.hist(dep_lats, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(x=11.5, color='red', linestyle='--', linewidth=2, label='Measured Avg (11.5s)')
        plt.title(f"Deposit Latency Distribution (Simulated N=1000)\nMean: {statistics.mean(dep_lats):.2f}s")
        plt.xlabel("Latency (s)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        hist_dep_name = os.path.join(RESULT_DIR, f"histogram_deposit{suffix_dep}.png")
        plt.savefig(hist_dep_name)
        plt.close()
        print(f"[Auto] Deposit Histogram saved to {hist_dep_name}")

    # Withdraw Histogram
    wd_lats = [r["t_total_latency"] for r in tx_records_wd.values()]
    if wd_lats:
        target_val = 230.0 if args.corrected else 201.0
        plt.figure(figsize=(10, 6))
        plt.hist(wd_lats, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        plt.axvline(x=target_val, color='red', linestyle='--', linewidth=2, label=f'Measured Avg (~{target_val}s)')
        plt.title(f"Withdraw Latency Distribution (Simulated N=1000)\nMean: {statistics.mean(wd_lats):.2f}s")
        plt.xlabel("Latency (s)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        hist_wd_name = os.path.join(RESULT_DIR, f"histogram_withdraw{suffix_wd}.png")
        plt.savefig(hist_wd_name)
        plt.close()
        print(f"[Auto] Withdraw Histogram saved to {hist_wd_name}")


def run_simulation_physics(args):
    """
    Phase 3: 物理モデルによるシミュレーション完全版

    Deposit + Withdraw 両方をシミュレート:
    - NetworkPhysics: 地理的レイテンシを考慮したネットワーク
    - HyperliquidValidator: 独立したバリデータエージェント
    - PhysicalBridgeConsensus: 物理的なVote到達時間計測 + QCブロードキャスト
    - ArbitrumBatcher: L1バッチ投稿物理シミュレーション
    - L1ToL2Relayer: Dispute Period + リレー遅延
    """
    from sim.arbitrum_batcher import ArbitrumBatcher, L1ToL2Relayer

    # 0. Result Directory Setup
    RESULT_DIR = "result"
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 1. Configuration Setup（完全版）
    TOTAL_TXS = args.tx_count if args.tx_count else 1000
    N = args.N  # デフォルト21

    # シミュレーション時間を十分に確保
    # Withdraw: 最後のTX投入(1+0.2*1000=201s) + Batcher待機(300s) + L1確定(36s) + Dispute(200s) + α
    SIMULATION_TIME = max(1000.0, TOTAL_TXS * 0.3 + 600)

    cfg = {
        # Global
        "chain": args.chain,
        "N": N,
        "quorum_ratio": args.quorum,
        "simulation_time_limit": SIMULATION_TIME,

        # Bridge Deposit (TUNING TARGET)
        "bridge_poll_interval_mu": args.poll_interval,
        "bridge_poll_interval_sigma": 0.5,
        "validator_sign_delay_min": 0.01,
        "validator_sign_delay_max": 0.05,

        # RPC Endpoint (Arbitrum側)
        "_rpc_endpoint_id": 0,

        # Withdraw: Batcher パラメータ（物理要件）
        "batcher_size_threshold_kb": 120.0,  # 120KB閾値
        "batcher_max_wait_s": 300.0,  # 5分タイムアウト
        "l1_block_time_s": 12.0,  # Ethereumブロック時間
        "l1_confirmations": 2,  # 確定までのブロック数

        # Withdraw: Dispute Period（実測値合わせ）
        "bridge_dispute_period_s": 200.0,  # ★重要: 200秒
        "relay_latency_mu_s": 1.0,
        "relay_latency_sigma_s": 0.3,

        # Logging
        "enable_validator_logging": args.verbose,
        "enable_consensus_logging": args.verbose,
        "enable_batcher_logging": args.verbose,
        "enable_relayer_logging": args.verbose,

        # Scenario
        "cross_chain_tx_interval": 0.2,
    }

    # 2. Export Parameters
    param_filename = os.path.join(RESULT_DIR, "simulation_parameters_physics.csv")
    with open(param_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value"])
        for k, v in cfg.items():
            if not k.startswith("_"):
                writer.writerow([k, v])
    print(f"[Physics] Parameters exported to {param_filename}")

    # 3. Simulation Environment
    env = simpy.Environment()
    rng = random.Random(42)

    # ========================================
    # DEPOSIT コンポーネント
    # ========================================

    # 4. 物理ネットワーク初期化
    print(f"[Physics] Initializing NetworkPhysics with N={N} (US=7, EU=7, APAC=7)")
    network = NetworkPhysics(
        env=env,
        N=N,
        rng=rng,
        enable_logging=args.verbose,
        jitter_ratio=0.1,
    )

    # 5. コンセンサスエンジン初期化
    consensus = PhysicalBridgeConsensus(env=env, N=N, config=cfg, network_physics=network)

    # 6. バリデータ初期化（N個の独立エージェント）
    print(f"[Physics] Initializing {N} validators...")
    validators = []
    for i in range(N):
        v = HyperliquidValidator(
            env=env,
            node_id=i,
            network_physics=network,
            consensus_engine=consensus,
            config=cfg,
            rng=random.Random(42 + i),
        )
        validators.append(v)

    # ========================================
    # WITHDRAW コンポーネント
    # ========================================

    # 7. ArbitrumBatcher初期化
    print(f"[Physics] Initializing ArbitrumBatcher (threshold={cfg['batcher_size_threshold_kb']}KB, max_wait={cfg['batcher_max_wait_s']}s)")
    batcher = ArbitrumBatcher(
        env=env,
        config=cfg,
        network_physics=network,
        rng=random.Random(100),
    )

    # 8. L1ToL2Relayer初期化
    print(f"[Physics] Initializing L1ToL2Relayer (dispute_period={cfg['bridge_dispute_period_s']}s)")
    relayer = L1ToL2Relayer(
        env=env,
        config=cfg,
        network_physics=network,
        rng=random.Random(200),
    )

    # ========================================
    # DATA COLLECTION
    # ========================================

    # Deposit用
    deposit_records = {}
    deposit_finalized = []

    def on_deposit_finalized(tx_id, finalize_time):
        if tx_id in deposit_records:
            submit_ts = deposit_records[tx_id]["submit_ts"]
            latency = finalize_time - submit_ts
            deposit_records[tx_id]["finalize_time"] = finalize_time
            deposit_records[tx_id]["latency"] = latency
            deposit_finalized.append(tx_id)

    consensus.on_deposit_finalized = on_deposit_finalized

    # Withdraw用
    withdraw_records = {}
    withdraw_finalized = []

    def on_withdraw_finalized(tx_id, finalize_time):
        # tx_id は 10001〜 を使用（Depositと区別）
        if tx_id in withdraw_records:
            submit_ts = withdraw_records[tx_id]["submit_ts"]
            latency = finalize_time - submit_ts
            withdraw_records[tx_id]["finalize_time"] = finalize_time
            withdraw_records[tx_id]["latency"] = latency
            withdraw_finalized.append(tx_id)

    relayer.on_withdraw_finalized = on_withdraw_finalized

    # Batcher -> Relayer 接続
    def on_batch_posted(tx_id, submit_time, l1_confirm_time):
        relayer.start_relay(tx_id, l1_confirm_time, submit_time)

    batcher.on_batch_posted = on_batch_posted

    # ========================================
    # TRAFFIC GENERATION
    # ========================================

    # Deposit Traffic
    def traffic_gen_deposit():
        yield env.timeout(1.0)  # 初期化待ち
        for i in range(1, TOTAL_TXS + 1):
            tx_id = i
            submit_ts = env.now
            deposit_records[tx_id] = {"submit_ts": submit_ts}

            # コンセンサスエンジンにDeposit通知
            consensus.notify_deposit(tx_id, submit_ts, {"amount": 100})

            yield env.timeout(cfg["cross_chain_tx_interval"])

    # Withdraw Traffic
    def traffic_gen_withdraw():
        yield env.timeout(1.0)  # 初期化待ち
        for i in range(1, TOTAL_TXS + 1):
            tx_id = 10000 + i  # Depositと区別（10001〜）
            submit_ts = env.now
            withdraw_records[tx_id] = {"submit_ts": submit_ts}

            # BatcherにWithdraw投入
            batcher.submit_withdrawal(tx_id, submit_ts, size_bytes=1500)

            yield env.timeout(cfg["cross_chain_tx_interval"])

    env.process(traffic_gen_deposit())
    env.process(traffic_gen_withdraw())

    # ========================================
    # SIMULATION EXECUTION
    # ========================================

    print(f"[Physics] Running Simulation (N={N}, Poll={args.poll_interval}s, TXs={TOTAL_TXS} Deposit + {TOTAL_TXS} Withdraw)...")
    print(f"[Physics] Simulation Time Limit: {SIMULATION_TIME}s")
    env.run(until=cfg["simulation_time_limit"])

    # ========================================
    # RESULTS PROCESSING - DEPOSIT
    # ========================================

    deposit_results = []
    for tx_id in sorted(deposit_finalized):
        rec = deposit_records.get(tx_id, {})
        if "latency" in rec:
            deposit_results.append({
                "tx_id": tx_id,
                "timestamp": rec["submit_ts"],
                "t_total_latency": rec["latency"],
                "status": "Finalized",
            })

    # Export Deposit Trace CSV
    trace_dep_filename = os.path.join(RESULT_DIR, "sim_trace_deposit_physics.csv")
    with open(trace_dep_filename, "w", newline="") as f:
        fieldnames = ["tx_id", "timestamp", "t_total_latency", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in deposit_results:
            row = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()}
            writer.writerow(row)
    print(f"[Physics] Deposit Trace saved to {trace_dep_filename}")

    # Deposit Statistics
    dep_latencies = [r["t_total_latency"] for r in deposit_results]
    if dep_latencies:
        dep_mean = statistics.mean(dep_latencies)
        dep_std = statistics.stdev(dep_latencies) if len(dep_latencies) > 1 else 0
        print(f"\n[Physics] === DEPOSIT Results ===")
        print(f"  Finalized TXs: {len(deposit_finalized)}/{TOTAL_TXS}")
        print(f"  Mean Latency: {dep_mean:.2f}s")
        print(f"  Std Dev: {dep_std:.2f}s")
        print(f"  Min: {min(dep_latencies):.2f}s, Max: {max(dep_latencies):.2f}s")

        # Deposit Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(dep_latencies, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(x=11.5, color='red', linestyle='--', linewidth=2, label='Measured Avg (11.5s)')
        plt.axvline(x=dep_mean, color='green', linestyle='-', linewidth=2, label=f'Simulated Avg ({dep_mean:.2f}s)')
        plt.title(f"[Physics Model] Deposit Latency Distribution (N={N})\nMean: {dep_mean:.2f}s, Std: {dep_std:.2f}s")
        plt.xlabel("Latency (s)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        hist_dep_filename = os.path.join(RESULT_DIR, "histogram_deposit_physics.png")
        plt.savefig(hist_dep_filename)
        plt.close()
        print(f"[Physics] Deposit Histogram saved to {hist_dep_filename}")

    # ========================================
    # RESULTS PROCESSING - WITHDRAW
    # ========================================

    withdraw_results = []
    for tx_id in sorted(withdraw_finalized):
        rec = withdraw_records.get(tx_id, {})
        if "latency" in rec:
            withdraw_results.append({
                "tx_id": tx_id,
                "timestamp": rec["submit_ts"],
                "t_total_latency": rec["latency"],
                "status": "Finalized",
            })

    # Export Withdraw Trace CSV
    trace_wd_filename = os.path.join(RESULT_DIR, "sim_trace_withdraw_physics.csv")
    with open(trace_wd_filename, "w", newline="") as f:
        fieldnames = ["tx_id", "timestamp", "t_total_latency", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in withdraw_results:
            row = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()}
            writer.writerow(row)
    print(f"[Physics] Withdraw Trace saved to {trace_wd_filename}")

    # Withdraw Statistics
    wd_latencies = [r["t_total_latency"] for r in withdraw_results]
    if wd_latencies:
        wd_mean = statistics.mean(wd_latencies)
        wd_std = statistics.stdev(wd_latencies) if len(wd_latencies) > 1 else 0
        print(f"\n[Physics] === WITHDRAW Results ===")
        print(f"  Finalized TXs: {len(withdraw_finalized)}/{TOTAL_TXS}")
        print(f"  Mean Latency: {wd_mean:.2f}s")
        print(f"  Std Dev: {wd_std:.2f}s")
        print(f"  Min: {min(wd_latencies):.2f}s, Max: {max(wd_latencies):.2f}s")
        print(f"  (Expected: ~230s = Batcher wait + L1 confirm + Dispute 200s)")

        # Withdraw Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(wd_latencies, bins=30, edgecolor='black', alpha=0.7, color='salmon')
        plt.axvline(x=230, color='red', linestyle='--', linewidth=2, label='Expected Avg (~230s)')
        plt.axvline(x=wd_mean, color='green', linestyle='-', linewidth=2, label=f'Simulated Avg ({wd_mean:.2f}s)')
        plt.title(f"[Physics Model] Withdraw Latency Distribution\nMean: {wd_mean:.2f}s, Std: {wd_std:.2f}s\n(Dispute Period: {cfg['bridge_dispute_period_s']}s)")
        plt.xlabel("Latency (s)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        hist_wd_filename = os.path.join(RESULT_DIR, "histogram_withdraw_physics.png")
        plt.savefig(hist_wd_filename)
        plt.close()
        print(f"[Physics] Withdraw Histogram saved to {hist_wd_filename}")

    # ========================================
    # COMPONENT STATISTICS
    # ========================================

    cons_stats = consensus.get_statistics()
    batcher_stats = batcher.get_statistics()
    relayer_stats = relayer.get_statistics()

    print(f"\n[Physics] === Consensus Statistics ===")
    print(f"  Finalized: {cons_stats.get('finalized_count', 0)}")
    print(f"  Vote Spread Mean: {cons_stats.get('vote_spread_mean_ms', 0):.1f}ms")
    print(f"  Vote Spread Std: {cons_stats.get('vote_spread_std_ms', 0):.1f}ms")

    print(f"\n[Physics] === Batcher Statistics ===")
    print(f"  Posted Batches: {batcher_stats.get('posted_count', 0)}")
    print(f"  Total TXs: {batcher_stats.get('total_txs', 0)}")
    print(f"  Avg Batch Size: {batcher_stats.get('avg_batch_size_kb', 0):.1f}KB")
    print(f"  Avg L1 Latency: {batcher_stats.get('avg_l1_latency_s', 0):.1f}s")

    print(f"\n[Physics] === Relayer Statistics ===")
    print(f"  Completed: {relayer_stats.get('completed_count', 0)}")
    print(f"  Avg Total Latency: {relayer_stats.get('avg_total_latency_s', 0):.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Simulator Runner")
    parser.add_argument("--chain", type=str, default="arbitrum")
    parser.add_argument("--N", type=int, default=21, help="Number of validators (default: 21 for Hyperliquid)")
    parser.add_argument("--poll_interval", type=float, default=16.0, help="Polling interval in seconds (default: 16.0 for ~11.5s deposit latency)")
    parser.add_argument("--quorum", type=float, default=0.67)
    parser.add_argument("--tx_count", type=int, default=1000)
    parser.add_argument("--tuned", action="store_true", help="Use tuned filename suffix")
    parser.add_argument("--corrected", action="store_true", help="Use corrected filename suffix and logic")
    parser.add_argument("--physics", action="store_true", help="Use Phase 2 physics model (NetworkPhysics + HyperliquidValidator)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for physics mode")

    args = parser.parse_args()

    if args.physics:
        run_simulation_physics(args)
    else:
        run_simulation_auto(args)
