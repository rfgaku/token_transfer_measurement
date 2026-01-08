
import math

class RiskModel:
    """
    Plan A: Stochastic Probability Layer.
    Calculates P(Reorg | t) based on time elapsed and network risk.
    """
    def __init__(self):
        self.alpha = 1.0 # Probability at t=0 (Immediate Reorg is almost certain if we don't wait)
        self.lambda_base = 0.5 # Base decay rate (Normal network settles quickly)

    def get_reorg_probability(self, time_elapsed, risk_coefficient):
        """
        Returns P(Reorg) after waiting 'time_elapsed' seconds using exponential decay.
        High risk reduces the decay rate (lambda), keeping probability high for longer.
        
        lambda_eff = lambda_base / risk_coefficient
        P = alpha * exp(-lambda_eff * t)
        """
        if time_elapsed < 0: return 1.0
        
        lambda_eff = self.lambda_base / risk_coefficient
        p = self.alpha * math.exp(-lambda_eff * time_elapsed)
        return p
