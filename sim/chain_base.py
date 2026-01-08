
import simpy
import random

class BaseBlock:
    def __init__(self, height, timestamp):
        self.height = height
        self.timestamp = timestamp
        self.status = "UNSAFE" # UNSAFE, SAFE, FINALIZED

    def __repr__(self):
        return f"Block(H={self.height}, Status={self.status})"

class BaseProtocol:
    """
    Simulates Base (OP Stack) Behavior.
    Characteristics:
    - Fixed 2.0s Block Time (No variance).
    - Batcher submits to L1 periodically.
    - L1 Finality lag.
    """
    def __init__(self, env):
        self.env = env
        self.block_time = 2.0
        self.current_height = 0
        
        # Batcher config
        self.batch_submission_interval = 120.0 # 2 minutes
        self.l1_confirmation_time = 12.0 # Ethereum Block Time
        self.l1_finality_depth = 2 # Epochs or Blocks for Finality (Simplified)
        
        # State
        self.blocks = []
        self.unsafe_head = 0
        self.safe_head = 0
        self.finalized_head = 0
        
        # Hooks
        self.on_soft_finality = None # Unsafe Head Update (Sequencer)
        self.on_hard_finality = None # Finalized Head Update (Bridge)

    def start(self):
        self.env.process(self._block_production_loop())
        self.env.process(self._batcher_loop())

    def _block_production_loop(self):
        """
        Produces blocks exactly every 2.0 seconds.
        """
        while True:
            # Wait exactly 2.0s
            # In OP Stack, this is driven by the Sequencer's clock.
            # It's very stable.
            yield self.env.timeout(self.block_time)
            
            self.current_height += 1
            new_block = BaseBlock(self.current_height, self.env.now)
            self.blocks.append(new_block)
            self.unsafe_head = new_block.height
            
            # Log
            print(f"[Base] Produced Block #{new_block.height} (Unsafe) at {self.env.now:.2f}s")
            
            # Notify Listener (Sequencer -> Unsafe Head update)
            if self.on_soft_finality:
                self.on_soft_finality(new_block)

    def _batcher_loop(self):
        """
        Periodically collects UNSAFE blocks and submits to L1.
        """
        while True:
            # Wait for next batch
            yield self.env.timeout(self.batch_submission_interval)
            
            # Identify blocks to submit
            # Everything > safe_head and <= unsafe_head
            to_submit = [b for b in self.blocks if b.height > self.safe_head and b.height <= self.unsafe_head]
            
            if not to_submit:
                continue
                
            last_block = to_submit[-1]
            print(f"[Base:Batcher] Submitting Batch (Blocks {to_submit[0].height}-{last_block.height}) to L1...")
            
            # Simulate L1 inclusion delay
            yield self.env.timeout(self.l1_confirmation_time)
            
            # Update Safe Head
            self.safe_head = last_block.height
            for b in to_submit:
                b.status = "SAFE"
            print(f"[Base:L1] Batch Confirmed. Safe Head Updated to #{self.safe_head}")
            
            # Process Finality (Simulate simplified L1 Finality lag after inclusion)
            # In sim, we can assume Finality follows Safe quickly for visualization
            # or add extra deep delay. Let's add 'Finality Delay'.
            self.env.process(self._finalize_batch(last_block.height))

    def _finalize_batch(self, height):
        # Ethereum Finality ~15 mins (2 epochs). 
        # For simulation, we scale this down or use a parameter.
        # Let's say 60 seconds for verified visualization.
        yield self.env.timeout(60.0)
        
        if height > self.finalized_head:
            self.finalized_head = height
            # Update status
            for b in self.blocks:
                if b.height <= height and b.status != "FINALIZED":
                    b.status = "FINALIZED"
            
            print(f"[Base:Finality] Block #{height} Finalized on L1.")
            
            if self.on_hard_finality:
                self.on_hard_finality(height)
