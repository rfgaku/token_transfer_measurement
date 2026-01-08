
import simpy
import matplotlib.pyplot as plt
import pandas as pd
from sim.chain_solana import SolanaProtocol

def run_solana_verification():
    env = simpy.Environment()
    protocol = SolanaProtocol(env)
    
    # Data Collection
    history_blocks = [] # (ts, slot, height, hash, type="block"|"fork")
    history_skips = []  # (ts, slot, type="skip")
    
    def on_slot(slot, skipped=False):
        if skipped:
            history_skips.append({
                "ts": env.now, 
                "slot": slot,
                "type": "skip"
            })

    def on_block(block, forked=False):
        history_blocks.append({
            "ts": env.now,
            "slot": block.slot,
            "height": block.height,
            "hash": block.hash,
            "type": "fork" if forked else "block"
        })
        
    protocol.on_slot = on_slot
    protocol.on_block = on_block
    
    print("Running Solana Simulation for 30s (~75 slots)...")
    protocol.start()
    env.run(until=30)
    
    # Analysis
    df_blocks = pd.DataFrame(history_blocks)
    df_skips = pd.DataFrame(history_skips)
    
    total_slots = getattr(protocol, "current_slot", 0)
    print(f"Total Slots: {total_slots}")
    print(f"Blocks Produced: {len(df_blocks)}")
    print(f"Skipped Slots: {len(df_skips)}")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot Blocks
    if not df_blocks.empty:
        # Regular blocks
        regular = df_blocks[df_blocks["type"] == "block"]
        plt.scatter(regular["ts"], regular["slot"], c="blue", label="Confirmed Block", s=20)
        
        # Forked blocks
        forks = df_blocks[df_blocks["type"] == "fork"]
        if not forks.empty:
            plt.scatter(forks["ts"], forks["slot"], c="red", marker="x", label="Micro-Fork (Latency)", s=50, zorder=5)
            
    # Plot Skips
    if not df_skips.empty:
        # Visualize skips as vertical lines or gaps
        for _, row in df_skips.iterrows():
            plt.axvline(x=row["ts"], color="orange", alpha=0.3, linestyle="--")
        # Add a dummy artist for legend
        plt.plot([], [], color="orange", linestyle="--", label="Skipped Slot (Jitter)")

    plt.title("Solana (0.4s Slot) Turbulence Analysis: Skips & Micro-forks")
    plt.xlabel("Time (s)")
    plt.ylabel("Slot Number")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("solana_turbulence.png")
    print("Generated solana_turbulence.png")

if __name__ == "__main__":
    run_solana_verification()
