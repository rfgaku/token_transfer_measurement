import simpy
import random
from sim.chain import create_chain_model
from sim.chain_polygon import PolygonProtocol
from sim.chain_base import BaseProtocol
from sim.chain_solana import SolanaProtocol
from sim.chain_hyperliquid import HyperliquidProtocol

class Sequencer:
    """
    Arbitrum Sequencer Simulator.
    Simulates 'Soft Finality' by ordering transactions with a small delay.
    """
    def __init__(self, env: simpy.Environment, config: dict):
        self.env = env
        self.config = config
        
        # Phase 9: Chain Abstraction
        chain_name = config.get("chain", "arbitrum")
        self.chain = create_chain_model(chain_name, config)
        
        # Basic processing time for Soft Finality (default ~250ms)
        model_delay = self.chain.get_soft_finality_delay()
        self.base_delay = float(config.get("sequencer_base_delay_s", model_delay))
        self.jitter_std = float(config.get("sequencer_jitter_std", 0.05))
        
        self.soft_finality_event = env.event()
        self.history = []
        
        self.reorg_mode = config.get("reorg_mode", "fixed")
        mtbf, depth = self.chain.get_reorg_params()
        
        self.reorg_mtbf = float(config.get("reorg_mtbf_s", mtbf if mtbf else 100.0))
        self.reorg_depth_avg = float(config.get("reorg_depth_avg_s", depth if depth else 5.0))
        
        if mtbf == 0.0:
             self.reorg_mtbf = 0.0
        
        self.on_reorg = None
        self.protocol = None
        
        if chain_name == "polygon":
            print(f"[Sequencer] Initializing Deep Polygon Protocol (Bor/Heimdall)...")
            self.protocol = PolygonProtocol(env)
            self.protocol.on_soft_finality = self._on_protocol_block
            self.protocol.on_reorg = self._on_protocol_reorg
            self.protocol.start()
            self.reorg_mtbf = 0.0

        if chain_name == "base":
            print(f"[Sequencer] Initializing Base Protocol (OP Stack)...")
            self.protocol = BaseProtocol(env)
            self.protocol.on_soft_finality = self._on_protocol_block
            self.protocol.start()
            self.reorg_mtbf = 0.0

        if chain_name == "solana":
            print(f"[Sequencer] Initializing Solana Protocol (Turbulence Mode)...")
            self.protocol = SolanaProtocol(env)
            self.protocol.on_block = self._on_protocol_block
            self.protocol.start()
            self.reorg_mtbf = 0.0

        if chain_name == "hyperliquid":
            print(f"[Sequencer] Initializing Hyperliquid Protocol (Instant BFT)...")
            self.protocol = HyperliquidProtocol(env)
            self.protocol.on_block = self._on_protocol_block
            self.protocol.start()
            self.reorg_mtbf = 0.0

        if self.reorg_mode == "probabilistic" and not self.protocol:
             self.env.process(self._reorg_loop())

        # Task 0: L1 Batcher for Arbitrum (Decoupled Hard Finality)
        if chain_name == "arbitrum":
            self.batcher_interval = float(config.get("batcher_interval_s", 60.0)) # ~1 min varies
            self.l1_finality_delay = float(config.get("l1_finality_delay_s", 12.0 * 2)) # ~2 blocks
            self.env.process(self._arbitrum_batcher_loop())
            self.on_hard_finality = None

    def _reorg_loop(self):
        while True:
            delay = self.chain.get_next_reorg_delay()
            if delay is None: break
            yield self.env.timeout(delay)
            depth = self.chain.get_reorg_depth()
            if depth > 0:
                invalidated = self.trigger_reorg(depth)
                if self.on_reorg:
                    self.on_reorg(self.env.now, depth, invalidated)
                oracle = self.config.get("_oracle_instance")
                if oracle:
                    oracle.on_reorg_event(depth)

    def _arbitrum_batcher_loop(self):
        """
        Simulates the Arbitrum Batcher posting data to Ethereum L1.
        """
        while True:
            yield self.env.timeout(self.batcher_interval)
            current_time = self.env.now
            yield self.env.timeout(self.l1_finality_delay)
            
            if hasattr(self, "on_hard_finality") and self.on_hard_finality:
                self.on_hard_finality(current_time)

    def submit_tx(self, tx_id):
        """
        User submits a transaction to the Sequencer.
        """
        # Phase 9.5: Protocol Coupling
        if self.protocol:
            if not hasattr(self, "pending_txs"): self.pending_txs = []
            self.pending_txs.append((tx_id, self.env.now))
            # Do NOT yield or finalize here. Wait for _on_protocol_block.
            # But making this function a generator in protocol mode needs careful handling.
            # The original architecture calls this with env.process().
            # So we must yield even if idling, or just return?
            # If we return, the process finishes immediately. That's fine for "Fire and Forget".
            return

        # Simulate processing limit / congestion
        delay = random.gauss(self.base_delay, self.jitter_std)
        if delay < 0.01: delay = 0.01
        
        yield self.env.timeout(delay)
        
        # Register in history
        self.history.append((tx_id, self.env.now))
        
        # Emit Soft Finality Event
        payload = {
            "tx_id": tx_id,
            "timestamp": self.env.now,
            "kind": "L2_SOFT_FINALIZED"
        }
        self._notify_observers(payload)

    def _notify_observers(self, payload):
        if hasattr(self, "on_soft_finalized") and callable(self.on_soft_finalized):
            self.on_soft_finalized(payload)

    def set_callback(self, callback_fn):
        self.on_soft_finalized = callback_fn
        
    def _on_protocol_block(self, block):
        if self.protocol:
            if not hasattr(self, "pending_txs"): self.pending_txs = []
            current_batch = list(self.pending_txs) 
            self.pending_txs = [] 
            
            for item in current_batch:
                if not isinstance(item, tuple): continue
                tx_id, _ = item
                self.history.append((tx_id, self.env.now))
                payload = {
                    "tx_id": tx_id,
                    "timestamp": self.env.now,
                    "kind": "L2_SOFT_FINALIZED",
                    "block_height": block.height
                }
                self._notify_observers(payload)
        else:
            pass 

    def _on_protocol_reorg(self, depth):
        print(f"[Sequencer] Protocol triggered Reorg (Depth={depth})")
        invalidated = self.trigger_reorg(depth)
        if self.on_reorg:
            self.on_reorg(self.env.now, depth, invalidated)
        oracle = self.config.get("_oracle_instance")
        if oracle:
            oracle.on_reorg_event(depth)
            
    def trigger_reorg(self, depth_s: float):
        now = self.env.now
        threshold = now - depth_s
        invalidated_txs = set()
        new_history = []
        for tx_id, ts in self.history:
            if ts > threshold:
                invalidated_txs.add(tx_id)
            else:
                new_history.append((tx_id, ts))
        self.history = new_history
        oracle = self.config.get("_oracle_instance")
        if oracle and self.reorg_mode == "fixed":
             oracle.on_reorg_event(depth_s)
        return invalidated_txs
