
import random
import numpy as np

class NetworkMonitor:
    """
    Plan D: Network Physics Layer.
    Monitors propagation delay variance to estimate current 'Risk Coefficient'.
    """
    def __init__(self, env=None):
        self.env = env
        self.history = []
        self.window_size = 10
        self.base_variance = 0.1 # Normal network jitter

    def record_observation(self, delay):
        self.history.append(delay)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def get_risk_coefficient(self, current_variance_override=None):
        """
        Calculates Risk Coefficient based on variance of propagation delays.
        Variance increases during 'Stormy' conditions (congestion, attacks).
        
        Formula: Risk = 1.0 + (Variance / Base_Variance)
        Mean: 1.0 (Normal) -> 10.0+ (Stormy)
        """
        if current_variance_override is not None:
             # Simulation hook for "Stormy Network Scenario" where we inject variance directly
             variance = current_variance_override
        elif len(self.history) > 1:
            variance = np.var(self.history)
        else:
            variance = self.base_variance

        # Normalize
        # If variance <= base, risk ~ 1.0
        # If variance is high, risk scales up
        risk_coefficient = 1.0 + (max(0, variance - self.base_variance) * 10.0)
        return risk_coefficient
