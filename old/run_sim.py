import argparse
import csv
import json
import math
import os
import random
import statistics
import time

import simpy

from sim.network import Network
from sim.node import Node
from sim.quorum import QuorumCollector
# v2.0 imports
from sim.sequencer import Sequencer
from sim.bridge import BridgeConsensus, BridgeRelayer
from sim.attestation import AttestationService


def _pctl(xs, p: float):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = int(math.ceil(p * (len(ys) - 1)))
    return ys[k]


def one_run(rng: random.Random, cfg: dict, N: int, q: int, byz_frac: float):
    timeout_s = float(cfg.get("timeout_s", 30.0))
    dispute_ref_s = float(cfg.get("dispute_ref_s", 200.0))

    env = simpy.Environment()

    # Pass strict arguments
    qc = QuorumCollector(env=env, q=q, N=N, timeout_s=timeout_s)
    # Stop event comes from qc
    stop_event = qc.quorum_event

    # Fix: latency config logic. 
    # Config files might have "network": { "latency": ... } or "network_latency": ...
    # We prefer cfg["network"]["latency"].
    net_cfg = cfg.get("network", {})
    lat_cfg = net_cfg.get("latency", cfg.get("network_latency", {}))
    
    # v2.0 Fix: Pass full network config to Network to support partitions
    # We merge latency config into it or just use net_cfg and handle structure in Network
    full_net_cfg = dict(net_cfg) # copy
    # Flatten or keep structure? Network.sample_latency_s uses keys directly like 'p2p_base_s'
    # So we should merge latency keys into the top level for backward compatibility
    # OR update Network to read from sub-key.
    # Quick fix: merge latency props into full_net_cfg
    if hasattr(lat_cfg, "items"):
        full_net_cfg.update(lat_cfg)
    
    net = Network(env=env, rng=rng, latency_config=full_net_cfg, stop_event=stop_event)

    leader_id = int(cfg.get("leader_id", 0))
    if leader_id < 0 or leader_id >= N:
        leader_id = 0

    exclude_leader = bool(cfg.get("byzantine", {}).get("exclude_leader", True))
    all_ids = list(range(N))
    if exclude_leader and leader_id in all_ids:
        all_ids.remove(leader_id)

    byz_n = int(round(byz_frac * N))
    byz_n = min(byz_n, len(all_ids))
    # rng.sample from all_ids
    byz_set = set(rng.sample(all_ids, byz_n)) if byz_n > 0 else set()

    honest_set = set(range(N)) - byz_set
    qc.set_honest_nodes(honest_set)

    nodes = [None] * N
    # IMPORTANT: We inject nodes into cfg so they can find each other via _deliver_to
    cfg["_nodes"] = nodes
    cfg["_N"] = int(N)

    # v2.0 Logic Setup
    sequencer = None
    bridge_consensus = None
    bridge_relayer = None
    
    # Check execution mode
    use_v2_logic = cfg.get("enable_v2", False)
    
    if use_v2_logic:
        # Initialize Shared Components
        sequencer = Sequencer(env, cfg)
        # Define finish event for v2
        finish_event = env.event()

        if cfg.get("enable_cctp", False):
            # CCTP Mode: Sequencer -> AttestationService -> Finish
            attestation_service = AttestationService(env, cfg)
            
            def on_attestation_done(tx_id, ts):
                 if not finish_event.triggered:
                     finish_event.succeed(value=ts)
            
            attestation_service.set_callback(on_attestation_done)
            cfg["_attestation_service"] = attestation_service
            
        else:
            # Default Bridge Mode
            bridge_consensus = BridgeConsensus(env, N, cfg)
            
            # Inject into cfg for Node access
            cfg["_bridge_consensus"] = bridge_consensus
            
            def on_deposit_done(tx_id, ts):
                 if not finish_event.triggered:
                     finish_event.succeed(value=ts)
            
            bridge_consensus.on_deposit_finalized = on_deposit_done
            
            # Withdraw Logic Components
            bridge_relayer = BridgeRelayer(env, cfg)
            
            def on_withdraw_done(tx_id, ts):
                 if not finish_event.triggered:
                     finish_event.succeed(value=ts)
            bridge_relayer.on_withdraw_finalized = on_withdraw_done
            
            # Inject relayer to cfg if needed (or just wire up)
            cfg["_bridge_relayer"] = bridge_relayer
    else:
        finish_event = qc.quorum_event # fallback to legacy

    # Probabilistic Reorg Tracking (Phase 7)
    reorg_count = 0
    reorg_total_depth = 0.0
    all_invalidated_txs = set()
    reorg_history = [] # (ts, invalidated_set)
    
    def on_reorg_callback(ts, depth, invalidated_txs):
        nonlocal reorg_count, reorg_total_depth
        reorg_count += 1
        reorg_total_depth += depth
        all_invalidated_txs.update(invalidated_txs)
        reorg_history.append((ts, invalidated_txs))
        
    if use_v2_logic and sequencer:
        sequencer.on_reorg = on_reorg_callback

    for i in range(N):
        nodes[i] = Node(
            env=env,
            node_id=i,
            net=net,
            qc=qc,
            cfg=cfg,
            rng=rng,
            stop_event=stop_event,
        )
        nodes[i].set_byzantine(i in byz_set)

    # ---- start broadcast at t=0 ----
    msg_id = int(rng.getrandbits(31))
    
    # Leader starts
    qc.note_proposal_seen(leader_id, msg_id, 0.0, 0, 0.0)
    
    # Withdraw Scenario Wire-up
    scenario = cfg.get("scenario", "deposit")
    
    # Warmup for Phase 8 (Oracle Risk Buildup)
    warmup_s = float(cfg.get("warmup_s", 0.0))
    if warmup_s > 0:
        env.run(until=warmup_s)
        # print(f"[DEBUG] Warmed up for {warmup_s}s. Reorg Count: {reorg_count}")
    
    if use_v2_logic:
        if (scenario == "deposit" or scenario == "reorg") and sequencer:
             def distribute_to_observers(payload):
                 # print(f"[DEBUG] Distributing payload to {len(nodes)} nodes")
                 
                 # 1. CCTP Path
                 if cfg.get("enable_cctp", False):
                     attestation_service.on_sequencer_event(payload)
                     
                 # 2. Bridge Observer Path (Standard Bridge)
                 elif cfg.get("enable_bridge_observer", False):
                     for n in nodes:
                         if n.bridge_observer:
                             n.bridge_observer.on_sequencer_event(payload)
                             
             sequencer.set_callback(distribute_to_observers)
             
             # Start Deposit
             sequencer.submit_tx(msg_id)
             
        elif scenario == "withdraw":
             # HyperBFT -> Relayer Wire-up
             def on_commit_to_relay(view, ts):
                 # print(f"[DEBUG] L1 Commit View {view}, triggering Relayer")
                 # Only track the first (View 0) transaction for latency measurement
                 if view == 0:
                     bridge_relayer.on_l1_commit(view, ts)
                 
             # Attach to Leader's HyperBFT (or ALL replicas? Relayer watches L1 chain)
             # In reality, Relayer watches the chain (consensus output).
             # So we can hook into ANY honest node's commit, or just the leader's for simplicity.
             # Better: Relayer watches the canonical chain. We can hook into Node 0 (Leader).
             nodes[leader_id].hyper_bft.set_callbacks(
                 nodes[leader_id].broadcast,
                 nodes[leader_id].send_direct,
                 commit_fn=on_commit_to_relay
             )
             
             # Start Withdraw: User submits to Leader
             # We assume Leader proposes it immediately
             # Just trigger the HyperBFT via normal proposal flow?
             # HyperBFT starts loop, Leader proposes every block_time.
             # So we just wait?
             # But we need to track a specific msg_id. 
             # HyperBFT proposals use 'view'. 
             # We can map msg_id to view 0?
             pass
             
    elif not use_v2_logic:
        # Legacy Broadcast
        nodes[leader_id].start_broadcast(msg_id)

    t0_wall = time.time()
    
    # Run until quorum or timeout
    # v2 logic might use finish_event, v1 uses qc.quorum_event
    env.run(until=(finish_event | env.timeout(timeout_s)))
    
    wall_s = time.time() - t0_wall

    quorum_time_rel = None
    if qc.t_quorum is not None and qc.t0 is not None:
        quorum_time_rel = float(qc.t_quorum - qc.t0)

    if quorum_time_rel is None:
        t_quorum = timeout_s
        fail = 1.0
    else:
        t_quorum = float(quorum_time_rel)
        fail = 0.0

    # Overwrite metrics for v2 logic if successful
    if use_v2_logic and finish_event.triggered:
        final_ts = finish_event.value
        t_quorum = final_ts # Total time since 0
        fail = 0.0

    # finalize reach metrics
    qc.finalize_reach(msg_id=msg_id, t_end_rel=float(env.now), t_quorum_rel=(None if fail else t_quorum))

    t_total = dispute_ref_s + t_quorum
    
    # === REORG DOUBLE SPEND CHECK ===
    double_spend = 0
    
    # Phase 7: Probabilistic Check
    if cfg.get("reorg_mode") == "probabilistic":
        if use_v2_logic and finish_event.triggered:
            # Calculate Effective Mint Time
            # Note: AttestationService already included safety_margin in 'final_ts' via delay.
            # But for Legacy (or extra manual delay), we add 'mint_delay_s'.
            mint_delay_s = float(cfg.get("mint_delay_s", 0.0))
            raw_mint_time = finish_event.value
            effective_mint_time = raw_mint_time + mint_delay_s
            
            # If we delayed minting, we should reflect that in the latency metric
            t_quorum = effective_mint_time
            
            # Wait simulation time if needed (simulating the delay)
            # We already waited 'finish_event.value' (raw_mint_time).
            # Now wait the extra mint_delay_s
            if mint_delay_s > 0:
                env.run(until=env.timeout(mint_delay_s))

            # Wait for Observation Period (to see if Reorg happens AFTER mint)
            obs_period = float(cfg.get("reorg_observation_s", 30.0))
            env.run(until=env.timeout(obs_period))
            
            # Check Double Spend:
            # D.S. occurs if our tx IS invalidated AND the invalidation happened AFTER effective_mint_time.
            # If invalidation happened BEFORE effective_mint_time, we assume Bridge logic caught it (Safe).
            
            # Optimization: Check if it was ever invalidated first
            if msg_id in all_invalidated_txs:
                # Find the FIRST reorg that killed it
                first_reorg_ts = None
                for (ts, inv_set) in reorg_history:
                    if msg_id in inv_set:
                         first_reorg_ts = ts
                         break
                
                if first_reorg_ts is not None:
                     if first_reorg_ts > effective_mint_time:
                         double_spend = 1
                         # print(f"[DEBUG] DOUBLE SPEND! Minted at {effective_mint_time}, Reorged at {first_reorg_ts}")
                     else:
                         # Reorg happened before Mint. We assume we aborted.
                         # This is NOT a Double Spend. It's a "Prevention".
                         double_spend = 0
                         # TODO: Should we mark 'fail=1'? 
                         # For trade-off analysis, we care about DS rate. Safe Fail is fine.
                
    # Phase 6: Fixed Scenario Check
    elif scenario == "reorg" and finish_event.triggered:
        # Mint (Bridge Finalization) happened at env.now
        mint_time = env.now
        
        # 1. Wait 'mint_delay_s' (Bridge Safety Margin)
        mint_delay_s = float(cfg.get("mint_delay_s", 0.0))
        if mint_delay_s > 0:
            env.run(until=env.timeout(mint_delay_s))
            
        # 2. Trigger Reorg (simulating L1 Reorg happening NOW)
        reorg_depth_s = float(cfg.get("reorg_depth_s", 5.0))
        invalidated_txs = sequencer.trigger_reorg(reorg_depth_s)
        
        # 3. Check Condition:
        if msg_id in invalidated_txs:
            double_spend = 1
        else:
            double_spend = 0
    
    out = {
        "fail": fail,
        "double_spend": double_spend,
        "t_quorum": float(t_quorum),
        "t_total": float(t_total),
        "reach_at_quorum": float(qc.reach_at_quorum),
        "reach_at_end": float(qc.reach_at_end),
        "reach_t_p99": float(qc.reach_t_p99),
        "reach_hops_mean": float(qc.reach_hops_mean),
        "wall_s": float(wall_s),
        "reorg_count": reorg_count,
        "reorg_total_depth": reorg_total_depth,
    }
    return out



def run_batch(cfg, N, q, bf, md, cm, mtbf, qr, runs_per, base_seed):
    # Update config for this run
    cfg["mint_delay_s"] = float(md)
    cfg["cctp_safety_margin_s"] = float(cm)
    cfg["reorg_mtbf_s"] = float(mtbf)
    
    fails, double_spends, tqs, ttot = [], [], [], []
    raq, rae, rtp, rhm, wall = [], [], [], [], []
    reorg_cnts, reorg_depths = [], []

    # Seed folding (Distinct seed for each param combo)
    rng = random.Random(base_seed ^ (int(N) << 16) ^ int(float(qr) * 1e6) ^ int(float(bf) * 1e6) ^ int(float(mtbf)*100) ^ int(float(md)*10))
    
    t_start_setting = time.time()
    
    for _ in range(runs_per):
        out = one_run(rng, cfg, int(N), int(q), float(bf))
        fails.append(out["fail"])
        double_spends.append(out.get("double_spend", 0))
        tqs.append(out["t_quorum"])
        ttot.append(out["t_total"])
        raq.append(out["reach_at_quorum"])
        rae.append(out["reach_at_end"])
        rtp.append(out["reach_t_p99"])
        rhm.append(out["reach_hops_mean"])
        wall.append(out["wall_s"])
        reorg_cnts.append(out.get("reorg_count", 0))
        reorg_depths.append(out.get("reorg_total_depth", 0.0))

    fail_rate = float(sum(fails) / len(fails))
    ds_rate = float(sum(double_spends) / len(double_spends)) if double_spends else 0.0
    
    # Safer stats if runs_per is small
    if runs_per > 0:
         avg_tq = statistics.mean(tqs)
         std_tq = statistics.stdev(tqs) if len(tqs) > 1 else 0.0
         p99_tq = _pctl(tqs, 0.99)
         p99_tt = _pctl(ttot, 0.99)
         avg_raq = statistics.mean(raq)
         avg_rae = statistics.mean(rae)
         avg_rtp = statistics.mean(rtp)
         avg_rhm = statistics.mean(rhm)
         avg_wall = statistics.mean(wall)
         avg_reorg_cnt = statistics.mean(reorg_cnts)
         avg_reorg_depth = statistics.mean(reorg_depths)
    else:
         avg_tq = 0
         p99_tq = 0
         p99_tt = 0
         avg_raq = 0
         avg_rae = 0
         avg_rtp = 0
         avg_rhm = 0
         avg_wall = 0
         avg_reorg_cnt = 0
         avg_reorg_depth = 0

    row = {
        "N": int(N),
        "quorum_ratio": float(qr),
        "q": int(q),
        "byz_frac": float(bf),
        "mint_delay_s": float(md),
        "cctp_safety_margin_s": float(cm),
        "reorg_mtbf_s": float(mtbf),
        "fail_rate": fail_rate,
        "double_spend_rate": ds_rate,
        "t_quorum_mean": float(avg_tq),
        "t_quorum_std": float(std_tq),
        "t_quorum_p99": float(p99_tq),
        "t_total_p99": float(p99_tt),
        "reach_at_quorum_mean": float(avg_raq),
        "reach_at_end_mean": float(avg_rae),
        "reach_t_p99_mean": float(avg_rtp),
        "reach_hops_mean_mean": float(avg_rhm),
        "wall_s_mean": float(avg_wall),
        "reorg_count_mean": float(avg_reorg_cnt),
        "reorg_total_depth_mean": float(avg_reorg_depth),
    }
    return row

def sweep(cfg: dict):
    sweep_cfg = cfg.get("sweep", {})
    Ns = sweep_cfg.get("N", [21])
    quorum_ratios = sweep_cfg.get("quorum_ratio", [2 / 3])
    byz_fracs = sweep_cfg.get("byz_frac", [0.0, 0.1, 0.2, 0.3])
    mint_delays = sweep_cfg.get("mint_delay_s", [0.0]) # v2.0 Sweep Support
    cctp_margins = sweep_cfg.get("cctp_safety_margin_s", [0.0]) # Phase 7 Sweep Support
    reorg_mtbfs = sweep_cfg.get("reorg_mtbf_s", [cfg.get("reorg_mtbf_s", 100.0)]) # Phase 8 Sweep
    runs_per = int(sweep_cfg.get("runs_per_setting", 200))

    base_seed = int(cfg.get("seed", 12345))

    rows = []
    summary = []

    for N in Ns:
        for qr in quorum_ratios:
            q = int(math.ceil(float(qr) * int(N)))
            for bf in byz_fracs:
                for md in mint_delays:
                    for cm in cctp_margins:
                        for mtbf in reorg_mtbfs:
                             row = run_batch(cfg, N, q, bf, md, cm, mtbf, qr, runs_per, base_seed)
                             rows.append(row)
                             summary.append(row)

    return rows, summary


def write_csv(path: str, rows: list):
    out_dir = os.path.dirname(path) or "."
    if out_dir != "." and not os.path.isdir(out_dir):
        # We don't auto-create the dir if it's missing (strict) or we could? 
        # Original code raised an error. We can just create it.
        os.makedirs(out_dir, exist_ok=True)

    cols = [
        "N",
        "quorum_ratio",
        "q",
        "byz_frac",
        "mint_delay_s",
        "cctp_safety_margin_s",
        "reorg_mtbf_s",
        "fail_rate",
        "double_spend_rate",
        "t_quorum_mean",
        "t_quorum_std",
        "t_quorum_p99",
        "t_total_p99",
        "reach_at_quorum_mean",
        "reach_at_end_mean",
        "reach_t_p99_mean",
        "reach_hops_mean_mean",
        "wall_s_mean",
        "reorg_count_mean",
        "reorg_total_depth_mean",
    ]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", help="path to config json or alias (optional if --scenario is used)")
    ap.add_argument("--scenario", default="", help="Scenario alias (phase4, phase5, phase6, phase7, phase8)")
    ap.add_argument("--chain", default="", help="Chain Override (arbitrum, polygon, base, solana, hyperliquid)")
    ap.add_argument("--out", default="", help="output csv path (default: cfg.out_csv or result/sim_sweep_withdraw.csv)")
    args = ap.parse_args()

    # Scenario Mapping
    SCENARIO_MAP = {
        "phase4": "configs/scenario_phase4.json",
        "phase5": "configs/scenario_phase5.json",
        "phase6": "configs/scenario_phase6.json",
        "phase7": "configs/v3_phase7_tradeoff_legacy.json",
        "phase8": "configs/v4_phase8_adaptive.json",
    }

    config_path = args.config
    if args.scenario:
        if args.scenario in SCENARIO_MAP:
            config_path = SCENARIO_MAP[args.scenario]
            print(f"Loaded scenario '{args.scenario}' from {config_path}")
        else:
            print(f"Error: Unknown scenario '{args.scenario}'. Available: {list(SCENARIO_MAP.keys())}")
            return
    elif not config_path:
        print("Error: Must provide config path or --scenario")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Allow --out to override
    out_csv = args.out or cfg.get("out_csv", "result/sim_sweep_withdraw.csv")
    
    # Allow --chain to override
    if args.chain:
        print(f"[INFO] Overriding chain to: {args.chain}")
        cfg["chain"] = args.chain

    rows, summary = sweep(cfg)
    write_csv(out_csv, rows)

    print("==== SWEEP SUMMARY (broadcast robust runner) ====")
    print("N,quorum_ratio,q,byz_frac,mint_delay_s,fail_rate,double_spend_rate,t_quorum_mean,t_quorum_std,t_quorum_p99,t_total_p99,reach_at_quorum_mean,reach_at_end_mean,reach_t_p99_mean,reach_hops_mean_mean,wall_s_mean")
    for r in summary:
        print(
            f"{r['N']},{r['quorum_ratio']:.7f},{r['q']},{r['byz_frac']:.1f},{r.get('mint_delay_s', 0.0):.1f},{r.get('cctp_safety_margin_s', 0.0):.1f},{r.get('reorg_mtbf_s', 0.0):.1f},{r['fail_rate']:.1f},{r['double_spend_rate']:.1f},"
            f"{r['t_quorum_mean']:.3f},{r['t_quorum_std']:.3f},{r['t_quorum_p99']:.3f},{r['t_total_p99']:.3f},"
            f"{r['reach_at_quorum_mean']:.6f},{r['reach_at_end_mean']:.6f},{r['reach_t_p99_mean']:.6f},"
            f"{r['reach_hops_mean_mean']:.6f},{r['wall_s_mean']:.6f},"
            f"{r.get('reorg_count_mean', 0):.2f}"
        )


if __name__ == "__main__":
    main()
