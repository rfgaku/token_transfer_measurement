import random

class ChainModel:
    """
    Abstract base class for L1 Chain characteristics.
    Defines Block Time, Finality, and Reorg behavior.
    """
    def __init__(self, config):
        self.name = "Generic"
        self.block_time = 1.0
        self.base_finality_delay = 0.0
        
    def get_soft_finality_delay(self):
        """Returns the time until 'Soft Finality' (or 'Confirmed')."""
        return self.block_time

    def get_next_reorg_delay(self):
        """
        Returns the delay until the next reorg event.
        Default: Poisson process (1/MTBF).
        """
        mtbf, _ = self.get_reorg_params()
        if mtbf is None or mtbf <= 0:
            return None
        return random.expovariate(1.0 / mtbf)

    def get_reorg_depth(self):
        """
        Returns the depth of the next reorg.
        Default: Exponential distribution (avg_depth).
        """
        _, avg = self.get_reorg_params()
        if avg is None: return 0.0
        return random.expovariate(1.0 / avg)

    def get_reorg_params(self):
        """
        Legacy method for generic params.
        """
        return None, None

class ArbitrumChain(ChainModel):
    def __init__(self, config):
        self.name = "Arbitrum"
        self.block_time = 0.25 # Soft Finality
        
    def get_reorg_params(self):
        # Default probabilistic reorgs (Phase 7 baseline)
        return 100.0, 5.0 

class PolygonChain(ChainModel):
    def __init__(self, config):
        self.name = "Polygon"
        self.block_time = 2.0
        self.sprint_length = 32 # blocks
        # Probability of sprint failure causing reorg
        self.sprint_fail_prob = 0.05 
        
    def get_reorg_params(self):
        return 50.0, 64.0 

    def get_next_reorg_delay(self):
        # LOGIC: Polygon reorgs happen when Sprints end (or fail).
        # We simulate "Burst Risk" instead of uniform Poisson.
        # Intervals are multiples of sprint time (32 * 2s = 64s).
        # But for simulation, let's say it's Probabilistic but biased towards typical Sprint failure.
        
        # Simulating "Sprint Failure":
        # Every 64s, there is a chance X of a deep reorg.
        # We can simulate this by returning a time that aligns with the next sprint?
        # A simpler robust approximation: Frequent small reorgs (latency), Occasional HUGE reorgs.
        
        # Let's mix two processes:
        # 1. Micro reorgs (propagation delay): MTBF = 20s, Depth = 2s
        # 2. Deep reorgs (Sprint failure): MTBF = 300s, Depth = 64s
        
        if random.random() < 0.2:
             # Deep One
             return random.expovariate(1.0 / 300.0)
        else:
             # Shallow One
             return random.expovariate(1.0 / 20.0)
             
    def get_reorg_depth(self):
        # If it was a deep interval, depth is deep.
        # We can't know which one triggered it here easily without state.
        # Let's make it bimodal.
        if random.random() < 0.2:
            return 32.0 * 2.0 # Full Sprint Depth (64s)
        return 2.0 # Shallow

class BaseChain(ChainModel):
    def __init__(self, config):
        self.name = "Base"
        self.block_time = 2.0
        
    def get_reorg_params(self):
        # Low Risk: Dependent on L1.
        # Very rare reorgs
        return 1000.0, 2.0

class SolanaChain(ChainModel):
    def __init__(self, config):
        self.name = "Solana"
        self.block_time = 0.4
        
    def get_reorg_params(self):
        return 20.0, 2.0
        
    def get_next_reorg_delay(self):
        # LOGIC: Frequent Micro-Forks (Turbulence)
        # Occurs very often (e.g. nearly every few seconds) but very shallow.
        return random.expovariate(1.0 / 5.0) # Every 5 seconds

    def get_reorg_depth(self):
        # Depth is always small (just current slot or two)
        return random.uniform(0.4, 1.2) # 1-3 slots

class HyperliquidChain(ChainModel):
    def __init__(self, config):
        self.name = "Hyperliquid"
        self.block_time = 0.2
        
    def get_reorg_params(self):
        # BFT: No Reorgs (unless >1/3 byzantine, which is a different fail mode)
        return 0.0, 0.0 # Disabled

def create_chain_model(chain_name, config):
    name = chain_name.lower()
    if name == "arbitrum": return ArbitrumChain(config)
    if name == "polygon": return PolygonChain(config)
    if name == "base": return BaseChain(config)
    if name == "solana": return SolanaChain(config)
    if name == "hyperliquid": return HyperliquidChain(config)
    return ArbitrumChain(config) # Default
