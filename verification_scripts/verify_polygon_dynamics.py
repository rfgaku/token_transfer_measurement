
import simpy
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sim.chain_polygon import PolygonProtocol

# --- Configuration ---
DURATION_SECONDS = 3600 # 1 Hour
NUM_BOR_NODES = 4

# --- Data Collection ---
history_blocks = []      # (timestamp, height, producer, is_main_chain)
history_forks = []       # (timestamp, height, producer, depth)
history_checkpoints = [] # (timestamp, height)

def run_verification():
    env = simpy.Environment()
    
    # Setup Protocol
    protocol = PolygonProtocol(env, num_bor=NUM_BOR_NODES)
    
    # Hook into events for data collection
    
    # 1. Block Production Hook
    # We need to monkey-patch or use existing hooks.
    # Existing hooks: on_soft_finality (Block), on_reorg (Depth), on_hard_finality (Height)
    
    original_soft_finality = protocol.on_soft_finality
    
    def on_block_produced(block):
        # Log every block produced as a candidate
        # We don't know yet if it stays in Main Chain or ends up as Fork.
        # But for the graph, we plot the "Head" evolution.
        history_blocks.append({
            "timestamp": env.now,
            "height": block.height,
            "producer": block.producer_id,
            "type": "block"
        })
        if original_soft_finality:
            original_soft_finality(block)
            
    protocol.on_soft_finality = on_block_produced
    
    # 2. Reorg Hook
    def on_reorg_event(depth):
        # We record that a reorg happened at this time
        # We can estimate the "discarded" height based on current time/block time
        # The hook provides 'depth' in seconds? or blocks?
        # In chain_polygon.py, it passes 2.0 (seconds).
        # Let's log it.
        history_forks.append({
            "timestamp": env.now,
            "depth_s": depth,
            "type": "reorg"
        })
        
    protocol.on_reorg = on_reorg_event
    
    # 3. Checkpoint Hook
    def on_checkpoint(height):
        history_checkpoints.append({
            "timestamp": env.now,
            "height": height,
            "type": "checkpoint"
        })
        
    protocol.on_hard_finality = on_checkpoint
    
    # Run
    print(f"Running Polygon Simulation for {DURATION_SECONDS} seconds (virtual)...")
    protocol.start()
    env.run(until=DURATION_SECONDS)
    
    # --- Analysis ---
    df_blocks = pd.DataFrame(history_blocks)
    df_forks = pd.DataFrame(history_forks)
    df_checkpoints = pd.DataFrame(history_checkpoints)
    
    total_blocks = len(df_blocks)
    total_forks = len(df_forks)
    total_checkpoints = len(df_checkpoints)
    
    # Calculate Latency (End-to-End)
    # Latency = Checkpoint Time - Block Time (approx)
    # We can average the delay for each checkpoint.
    latencies = []
    if not df_checkpoints.empty and not df_blocks.empty:
        for _, cp in df_checkpoints.iterrows():
            # Find the block with this height
            block_row = df_blocks[df_blocks["height"] == cp["height"]]
            if not block_row.empty:
                # Use the first occurrence (production time)
                prod_time = block_row.iloc[0]["timestamp"]
                commit_time = cp["timestamp"]
                latencies.append(commit_time - prod_time)
                
    avg_latency = np.mean(latencies) if latencies else 0.0
    
    print("\n=== Quantitative Report ===")
    print(f"Total Blocks: {total_blocks}")
    print(f"Total Forks (Reorgs): {total_forks}")
    print(f"Total Checkpoints: {total_checkpoints}")
    print(f"Avg Finality Latency: {avg_latency:.2f} s")
    print("===========================")
    
    # --- Visualization ---
    plt.figure(figsize=(14, 8))
    
    # 1. Main Block Growth
    # Filter to visualize the "Winning" chain roughly
    # Since reorgs are small (2s), the macro line is straight.
    plt.plot(df_blocks["timestamp"], df_blocks["height"], label="Bor Chain Height", color="blue", alpha=0.6, linewidth=1)
    
    # 2. Checkpoints
    if not df_checkpoints.empty:
        plt.scatter(df_checkpoints["timestamp"], df_checkpoints["height"], color="green", s=100, marker="D", label="Heimdall Local Checkpoint", zorder=5)
        
        # Add L1 Commit lines (Simulating the 10s+ delay)
        # We know Checkpoint happens some time after Block.
        # Let's verify the delay visually.
    
    # 3. Fork Events
    # Mark where forks happened on the Timeline
    if not df_forks.empty:
        # We don't have the exact height in the Reorg event payload (simplified sim),
        # but we can map it to the block height at that timestamp.
        # Find nearest block height
        fork_heights = []
        for _, fork in df_forks.iterrows():
            # Find block closest to this timestamp
            idx = (df_blocks["timestamp"] - fork["timestamp"]).abs().idxmin()
            fork_heights.append(df_blocks.loc[idx, "height"])
            
        plt.scatter(df_forks["timestamp"], fork_heights, color="red", s=50, marker="x", label="Fork / Reorg Event", zorder=6)

    plt.title("Polygon PoS Dynamics: Bor Production vs Heimdall Finality")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Block Height")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoom in to a specific Sprint transition if possible?
    # No, full view is better for evidence of "Long Latency".
    
    plt.tight_layout()
    plt.savefig("polygon_dynamics.png")
    print("Generated 'polygon_dynamics.png'")

if __name__ == "__main__":
    run_verification()
