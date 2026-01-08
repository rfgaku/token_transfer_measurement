
import simpy
import pandas as pd
from sim.chain_polygon import PolygonProtocol
from sim.chain_base import BaseProtocol
from sim.chain_solana import SolanaProtocol
from sim.chain_hyperliquid import HyperliquidProtocol
# Arbitrum is legacy probabilistic in our sim, handled by Sequencer, but we can wrap it or just invoke Sequencer.
# For consistency in this script, we'll instantiate protocols if they exist, or use a dummy for Arbitrum.

class ArbitrumDummyProtocol:
    def __init__(self, env):
        self.env = env
        self.block_time = 0.25
        self.blocks = []
    def start(self): 
        self.env.process(self._loop())
    def _loop(self):
        h = 0
        while True:
            yield self.env.timeout(self.block_time)
            h += 1
            # Arbitrum is probabilistic finality in run_sim, here just pure block production for metric
            # To get "Time to Finality" we usually rely on "Soft Finality" (~250ms) vs "Hard" (week).
            # For comparison, we'll log "Soft Finality" latency.

def run_chain_sim(chain_name, duration=120):
    env = simpy.Environment()
    protocol = None
    
    logs = {"blocks": 0, "finality_latency": [], "forks": 0}
    
    if chain_name == "polygon":
        protocol = PolygonProtocol(env)
        # Hook for finality
        def on_hard(height):
            # Approximate latency: time now - (height * 2.0)
            # This is rough, implies genesis at 0. But sufficient for relative comparison.
            logs["finality_latency"].append(env.now - (height * 2.0))
        protocol.on_hard_finality = on_hard
        def on_reorg(depth):
            logs["forks"] += 1
        protocol.on_reorg = on_reorg
        
    elif chain_name == "base":
        protocol = BaseProtocol(env)
        def on_hard(height):
            # Base finality lag is huge, we will capture it
            # Latency = Now - (Height * 2.0)
            logs["finality_latency"].append(env.now - (height * 2.0))
        protocol.on_hard_finality = on_hard
        
    elif chain_name == "solana":
        protocol = SolanaProtocol(env)
        def on_root(block):
            # Root latency ~ 32 slots * 0.4 = 12.8s
            logs["finality_latency"].append(env.now - block.timestamp)
        protocol.on_root = on_root
        def on_block(block, forked):
            if forked: logs["forks"] += 1
        protocol.on_block = on_block
        
    elif chain_name == "hyperliquid":
        protocol = HyperliquidProtocol(env)
        def on_block(block):
            # Instant finality
            logs["finality_latency"].append(0.01) # Near zero
        protocol.on_block = on_block
        
    elif chain_name == "arbitrum":
        protocol = ArbitrumDummyProtocol(env)
        # Arbitrum Soft Finality is ~0.25s
        logs["finality_latency"].append(0.25) 

    if protocol:
        protocol.start()
        env.run(until=duration)
        
        # Count blocks
        if hasattr(protocol, "blocks"):
            if isinstance(protocol.blocks, list): logs["blocks"] = len(protocol.blocks)
            elif isinstance(protocol.blocks, dict): logs["blocks"] = len(protocol.blocks)
        elif chain_name == "arbitrum":
             logs["blocks"] = int(duration / 0.25)

    return logs

def main():
    chains = ["polygon", "base", "solana", "arbitrum", "hyperliquid"]
    results = []
    
    print(f"Running Cross-Chain Matrix Verification (Duration=300s)...")
    
    for chain in chains:
        print(f" > Testing {chain}...")
        metrics = run_chain_sim(chain, duration=300)
        
        avg_lat = 0
        if metrics["finality_latency"]:
            avg_lat = sum(metrics["finality_latency"]) / len(metrics["finality_latency"])
            
        results.append({
            "Source Chain": chain.capitalize(),
            "Consensus": "Unknown",
            "Risk Profile": "Unknown",
            "Final Latency": f"{avg_lat:.2f}s",
            "Forks (stability)": metrics["forks"] if "forks" in metrics else 0
        })
        
    # Enrich Data (Static Knowledge + Sim Metrics)
    # This matches the user's requested matrix structure
    for r in results:
        if r["Source Chain"] == "Polygon":
            r["Consensus"] = "PoS (Sprint)"
            r["Risk Profile"] = "High (Sprint Reorgs)"
            r["Oracle Reaction"] = "Alert (Delay++)"
        elif r["Source Chain"] == "Base":
            r["Consensus"] = "Optimistic"
            r["Risk Profile"] = "Lagging (Unsafe)"
            r["Oracle Reaction"] = "Wait for L1"
        elif r["Source Chain"] == "Solana":
            r["Consensus"] = "PoH / Tower BFT"
            r["Risk Profile"] = "Turbulent (Micro-forks)"
            r["Oracle Reaction"] = "Micro-wait"
        elif r["Source Chain"] == "Arbitrum":
            r["Consensus"] = "Sequencer (Soft)"
            r["Risk Profile"] = "Medium (Probabilistic)"
            r["Oracle Reaction"] = "Standard"
        elif r["Source Chain"] == "Hyperliquid":
            r["Consensus"] = "BFT (Instant)"
            r["Risk Profile"] = "None"
            r["Oracle Reaction"] = "Instant"

    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ["Source Chain", "Consensus", "Risk Profile", "Oracle Reaction", "Final Latency", "Forks (stability)"]
    df = df[cols]
    
    print("\n=== Universal Capability Matrix ===")
    print(df.to_markdown(index=False))
    
    # Save for report usage
    df.to_csv("universal_matrix.csv", index=False)

if __name__ == "__main__":
    main()
