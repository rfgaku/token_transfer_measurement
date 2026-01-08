
import simpy

class HyperliquidBlock:
    def __init__(self, height, timestamp):
        self.height = height
        self.timestamp = timestamp
        self.status = "FINALIZED" # Instant Finality

    def __repr__(self):
        return f"Block(H={self.height}, Status={self.status})"

class HyperliquidProtocol:
    """
    Simulates Hyperliquid (Tendermint-like L1).
    - Block Time: 0.8s (Fast)
    - Consensus: Instant Finality (No Forks, No Reorgs)
    """
    def __init__(self, env):
        self.env = env
        self.block_time = 0.8
        self.current_height = 0
        self.blocks = []
        
        # Hooks
        self.on_block = None

    def start(self):
        self.env.process(self._consensus_loop())

    def _consensus_loop(self):
        while True:
            # 1. Propose & Vote Delay (Simplified BFT rounds)
            yield self.env.timeout(self.block_time)
            
            # 2. Commit
            self.current_height += 1
            new_block = HyperliquidBlock(self.current_height, self.env.now)
            self.blocks.append(new_block)
            
            # print(f"[Hyperliquid] Block #{new_block.height} Finalized Instantly at {self.env.now:.2f}s")
            
            if self.on_block:
                self.on_block(new_block)
