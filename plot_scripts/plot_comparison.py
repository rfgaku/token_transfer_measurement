
import json
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import argparse

# Import simulation logic
sys.path.append(os.getcwd())
try:
    from run_sim import one_run
except ImportError:
    print("Error: run_sim.py not found using current path")
    sys.exit(1)

def load_real_data(path: str, col: str = "latency(ms)") -> np.ndarray:
    if not os.path.exists(path):
        print(f"Bypassing real data load (file not found): {path}")
        return np.array([])
    
    df = pd.read_csv(path)
    if col not in df.columns:
        x_ms = df.iloc[:, 0].astype(float).to_numpy()
    else:
        x_ms = df[col].astype(float).to_numpy()
    return x_ms / 1000.0  # ms to sec

def generate_sim_data(cfg_path: str, runs: int = 1000, target="t_quorum"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    sweep = cfg.get("sweep", {})
    N = int(sweep.get("N", [14])[0])
    qr = float(sweep.get("quorum_ratio", [0.6666])[0])
    q = int(math.ceil(qr * N))
    byz_frac = float(sweep.get("byz_frac", [0.0])[0])
    
    # Override seed to ensure variability if not handled well
    base_seed = int(cfg.get("seed", 12345))
    rng = random.Random(base_seed)
    
    print(f"Simulating {cfg_path} (N={N}, q={q}, runs={runs})...")
    vals = []
    for i in range(runs):
        out = one_run(rng, cfg, N, q, byz_frac)
        if out["fail"] == 0.0:
            if target == "t_total":
                vals.append(out["t_total"])
            else:
                vals.append(out["t_quorum"])
    return np.array(vals)

def plot_hist(real, sim, title, out_path, range_max=None):
    plt.figure(figsize=(10, 6))
    bins = 50
    alpha = 0.6
    
    # Auto range
    if range_max is None:
        if len(real) > 0:
            rmax = max(real.max(), sim.max())
        else:
            rmax = sim.max()
        range_max = rmax * 1.1

    if len(real) > 0:
        plt.hist(real, bins=bins, density=True, alpha=alpha, label='Real', color='blue', range=(0, range_max))
    
    plt.hist(sim, bins=bins, density=True, alpha=alpha, label='Sim', color='orange', range=(0, range_max))
    
    plt.title(title)
    plt.xlabel("Latency (s)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    
    # Stats
    print(f"--- {title} Stats ---")
    if len(real) > 0:
        print(f"Real: Mean={real.mean():.4f}, Std={real.std():.4f}")
    print(f"Sim : Mean={sim.mean():.4f}, Std={sim.std():.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["deposit", "withdraw"], required=True)
    args = parser.parse_args()

    if args.mode == "deposit":
        real = load_real_data("result/deposit_latency.csv")
        sim = generate_sim_data("configs/calibrated_deposit.json", runs=1000, target="t_quorum")
        plot_hist(real, sim, "Deposit Latency Comparison", "comparison_deposit.png", range_max=30)
        
    elif args.mode == "withdraw":
        real = load_real_data("result/withdraw_latency.csv")
        sim = generate_sim_data("configs/calibrated_withdraw.json", runs=1000, target="t_total")
        plot_hist(real, sim, "Withdraw Latency Comparison", "comparison_withdraw.png", range_max=300)

if __name__ == "__main__":
    main()
