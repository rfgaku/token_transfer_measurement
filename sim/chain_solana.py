
import simpy
import random

class SolanaBlock:
    def __init__(self, slot, timestamp, parent_hash, leader_id):
        self.slot = slot
        self.timestamp = timestamp
        self.parent_hash = parent_hash
        self.leader_id = leader_id
        self.hash = f"{slot}_{leader_id}"
        self.status = "PROCESSED" # PROCESSED -> CONFIRMED -> ROOT
        self.height = 0 # Logical height (not slot)

    def __repr__(self):
        return f"Block(Slot={self.slot}, Leader={self.leader_id}, Status={self.status})"

class SolanaProtocol:
    """
    Simulates Solana's High Throughput & Turbulence.
    - Slot: 0.4s
    - Leader Schedule: Deterministic
    - Turbulence: Micro-forks & Skips
    """
    def __init__(self, env, num_validators=4):
        self.env = env
        self.block_time = 0.4 # 400ms
        self.validators = [f"Val-{i}" for i in range(num_validators)]
        
        # State
        self.current_slot = 0
        self.blocks = {} # hash -> Block
        self.genesis_block = SolanaBlock(0, 0, "GENESIS", "System")
        self.genesis_block.status = "ROOT"
        self.blocks[self.genesis_block.hash] = self.genesis_block
        self.head_hash = self.genesis_block.hash
        
        # Turbulence Config
        self.p_skip_slot = 0.05    # 5% chance leader misses slot
        self.p_micro_fork = 0.03   # 3% chance leader builds on old block (latency)
        
        # Hooks
        self.on_slot = None
        self.on_block = None
        self.on_root = None

    def start(self):
        self.env.process(self._slot_loop())

    def get_leader(self, slot):
        # Round-robin schedule
        idx = slot % len(self.validators)
        return self.validators[idx]

    def _slot_loop(self):
        while True:
            # 1. Wait for Slot boundary
            yield self.env.timeout(self.block_time)
            self.current_slot += 1
            
            leader = self.get_leader(self.current_slot)
            
            # 2. Turbulence Check (Skip Slot)
            if random.random() < self.p_skip_slot:
                print(f"[Solana] Slot {self.current_slot} Skipped! (Leader: {leader} - Network Jitter)")
                if self.on_slot: self.on_slot(self.current_slot, skipped=True)
                continue
            
            # 3. Determine Parent (Micro-fork logic)
            parent_hash = self.head_hash
            forked = False
            
            if random.random() < self.p_micro_fork:
                # Simulate latency: Leader didn't see the very latest block
                # Build on parent's parent (if exists)
                head_block = self.blocks.get(self.head_hash)
                if head_block and head_block.parent_hash != "GENESIS":
                    parent_hash = head_block.parent_hash
                    forked = True
                    print(f"[Solana] Slot {self.current_slot} Forked! (Leader: {leader} built on {parent_hash})")
            
            # 4. Produce Block
            new_block = SolanaBlock(self.current_slot, self.env.now, parent_hash, leader)
            parent_block = self.blocks.get(parent_hash)
            new_block.height = parent_block.height + 1 if parent_block else 0
            
            self.blocks[new_block.hash] = new_block
            
            # 5. Greedy Choice (LFC) - Simplification: Deepest chain wins instantly
            # In (real) Solana, it's weighted by stake. Here we assume uniform stake.
            # If forked, we created a branch. If it's longer (impossible since just +1) or equivalent?
            # Solana logic: Switch to heaviest fork. Here we simplify:
            # If we built on Head, we update Head.
            # If we built on Old, we created a fork but we are NOT the head yet unless we extend.
            
            # Logic: Always update Head to this new block?
            # No, if forked, we have two tips.
            # Let's simple Greedy GHOST: Heavier observed subtree.
            # Since linear, usually Height is metric.
            
            if new_block.height > self.blocks[self.head_hash].height:
                self.head_hash = new_block.hash
                msg = "Confirmed"
            elif new_block.height == self.blocks[self.head_hash].height:
                # Tie-breaking (e.g., lower hash or slot). Prefer newer slot.
                self.head_hash = new_block.hash 
                msg = "Fork Switched"
            else:
                msg = "Minor Fork (Orphaned)"

            print(f"[Solana] Slot {self.current_slot} (Leader: {leader}) -> {msg}")
            
            if self.on_block:
                self.on_block(new_block, forked)
            
            # 6. Finality (Root) Process
            # Periodically finalize blocks older than X slots (Max Lockout)
            # 32 slots ~ 12.8s
            self.env.process(self._update_root())

    def _update_root(self):
        # Simplified Root update:
        # Trace back 32 slots from Head. If consistent, finalize.
        # But for simulation logs, simply finalizing "Head - 32" is enough visual cue.
        
        yield self.env.timeout(0) # async
        head = self.blocks.get(self.head_hash)
        if not head: return
        
        # Traverse back
        curr = head
        depth = 0
        while curr.parent_hash != "GENESIS" and depth < 32:
            curr = self.blocks.get(curr.parent_hash)
            depth += 1
            
        if depth == 32 and curr.status != "ROOT":
            curr.status = "ROOT"
            # print(f"[Solana] Root Updated to Slot {curr.slot}")
            if self.on_root: self.on_root(curr)
