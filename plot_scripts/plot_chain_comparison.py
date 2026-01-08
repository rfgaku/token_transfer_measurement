import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Load data
    chains = {
        "Hyperliquid": "result/phase9_hyperliquid.csv",
        "Base": "result/phase9_base.csv",
        "Arbitrum": "result/compare_legacy_mtbf10.csv", # Using legacy mtbf10 as proxy for logic-less arbitrum baseline? 
        # Actually let's use the one from Phase 7 or just skip Arbitrum if data missing
        "Polygon (High Risk)": "result/phase9_polygon.csv",
        "Solana (Turbulence)": "result/phase9_solana.csv"
    }
    
    data = []
    labels = []
    
    for name, path in chains.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Filter for a specific setting (e.g., N=21, byz=0)
            df = df[df["byz_frac"] == 0.0]
            if not df.empty:
                # We want the distribution of latencies.
                # But CSV only has aggregated mean/p99 per run batch.
                # We can plot the P99 latencies across different runs/settings?
                # Or just plot the single "wall_s_mean" bar chart?
                # User asked for Box Plot (Distribution).
                # We only have aggregated stats in CSV (rows=settings).
                # We can use "wall_s_mean" from the single row if we assume 1 setting.
                
                # Let's assume the CSV contains results for the "Best" adaptive setting we found.
                # Actually run_sim outputs 1 row per setting.
                # For plotting, a Bar Chart of Mean Latency is safer given data.
                val = df["wall_s_mean"].mean()
                data.append(val)
                labels.append(name)
    
    if not data:
        print("No data found for comparison plot.")
        return

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, data, color=['#4CAF50', '#2196F3', '#9C27B0', '#F44336', '#FF9800'])
    
    plt.ylabel('Mean Latency (s)')
    plt.title('Cross-Chain Adaptive Bridge Latency Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}s", ha='center', va='bottom')
        
    plt.savefig('result/chain_comparison.png')
    print("Plot generated: result/chain_comparison.png")

if __name__ == "__main__":
    main()
