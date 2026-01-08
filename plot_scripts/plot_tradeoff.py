import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Load Data
    try:
        df_legacy = pd.read_csv("result/tradeoff_legacy.csv")
        df_cctp = pd.read_csv("result/tradeoff_cctp.csv")
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    # Filter relevant columns
    # Legacy: sweep mint_delay_s. Label="Legacy"
    # CCTP: sweep cctp_safety_margin_s. Label="CCTP"
    
    # We want to plot X=t_quorum_mean (Latency), Y=double_spend_rate
    
    # Process Legacy
    df_legacy["Model"] = "Legacy Bridge"
    # Latency is t_quorum_mean.
    # Note: run_sim.py adds mint_delay_s to t_quorum for us.
    
    # Process CCTP
    df_cctp["Model"] = "CCTP"
    
    # Combine
    df = pd.concat([df_legacy, df_cctp], ignore_index=True)
    
    # Setup Plot
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    
    # Filter data
    df_leg = df[df["Model"] == "Legacy Bridge"]
    df_cctp = df[df["Model"] == "CCTP"]
    
    # Plot Legacy
    plt.plot(
        df_leg["t_quorum_mean"], 
        df_leg["double_spend_rate"], 
        label="Legacy Bridge", 
        marker='o', 
        linestyle='-', 
        linewidth=2
    )
    
    # Plot CCTP
    plt.plot(
        df_cctp["t_quorum_mean"], 
        df_cctp["double_spend_rate"], 
        label="CCTP", 
        marker='s', 
        linestyle='--', 
        linewidth=2
    )
    
    # Annotate Risk Zone
    plt.axhline(y=0.0, color='green', linestyle='--', alpha=0.3)
    xmin = df["t_quorum_mean"].min()
    plt.text(xmin, -0.05, "Safe Zone", color='green', fontsize=10)
    
    # Labels
    plt.title("Risk-Reward Trade-off: Probabilistic Reorg (MTBF=20s, Depth=10s)", fontsize=14)
    plt.xlabel("Finality Latency (seconds)", fontsize=12)
    plt.ylabel("Double Spend Risk (Rate)", fontsize=12)
    plt.ylim(-0.1, 1.1)
    
    plt.legend()
    plt.grid(True)
    
    # Save
    out_path = "tradeoff_cctp.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
