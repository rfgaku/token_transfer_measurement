
import simpy
import matplotlib.pyplot as plt
import pandas as pd
from sim.chain_base import BaseProtocol

def run_base_verification():
    env = simpy.Environment()
    protocol = BaseProtocol(env)
    
    # Data Collectors
    history_unsafe = []     # (ts, height)
    history_finalized = []  # (ts, height)
    
    # Hooks
    def on_unsafe(block):
        history_unsafe.append((env.now, block.height))
        # Log finalized height too at this tick, or use latch
        history_finalized.append((env.now, protocol.finalized_head))

    def on_finalized(height):
        # When finality bumps, record it
        history_finalized.append((env.now, height))
        # Record unsafe height too for continuity
        history_unsafe.append((env.now, protocol.unsafe_head))
        
    protocol.on_soft_finality = on_unsafe
    protocol.on_hard_finality = on_finalized
    
    print("Running Base Simulation for 600s...")
    protocol.start()
    env.run(until=600)
    
    # Plotting
    df_unsafe = pd.DataFrame(history_unsafe, columns=["ts", "height"])
    df_finalized = pd.DataFrame(history_finalized, columns=["ts", "height"])
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_unsafe["ts"], df_unsafe["height"], label="Unsafe Head (L2)", color="blue")
    plt.step(df_finalized["ts"], df_finalized["height"], label="Finalized Head (L1)", color="green", where="post")
    
    plt.title("Base (OP Stack) Finality Lag Analysis")
    plt.xlabel("Time (s)")
    plt.ylabel("Block Height")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("base_finality_lag.png")
    print("Generated base_finality_lag.png")

if __name__ == "__main__":
    run_base_verification()
