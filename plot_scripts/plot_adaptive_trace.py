import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    if not os.path.exists("result/trace_adaptive_tx.csv"):
        print("Trace CSV not found.")
        return

    df_tx = pd.read_csv("result/trace_adaptive_tx.csv")
    df_reorg = pd.read_csv("result/trace_adaptive_reorg.csv")
    
    # Filter out initial warmup noise if any
    start_time = 0
    df_tx = df_tx[df_tx["timestamp"] > start_time]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Latency (Left Axis)
    ax1.set_xlabel('Simulation Time (s)')
    ax1.set_ylabel('Transaction Latency (s)', color='tab:blue')
    ax1.plot(df_tx['timestamp'], df_tx['latency'], 'o-', color='tab:blue', markersize=4, label='Latency', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(-1, 30) # Latency usually 0-20s
    
    # Plot Risk Score (Right Axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Oracle Risk Score', color='tab:orange')
    ax2.plot(df_tx['timestamp'], df_tx['risk_score'], '-', color='tab:orange', label='Risk Score', alpha=0.6, linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylim(0, 1.2)
    
    # Plot Reorg Events (Vertical Lines)
    for _, row in df_reorg.iterrows():
        ts = row['timestamp']
        depth = row['depth']
        ax1.axvline(x=ts, color='red', linestyle='--', alpha=0.5)
        # ax1.text(ts, 25, f"Reorg\n(d={depth:.1f})", color='red', fontsize=8, rotation=90)
    
    plt.title('Validation: Adaptive Responsiveness to Reorgs')
    fig.tight_layout()
    plt.savefig('result/adaptive_behavior.png')
    print("Plot generated: result/adaptive_behavior.png")

if __name__ == "__main__":
    main()
