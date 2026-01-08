import random

class SafetyOracle:
    """
    Simulates a 'Safety Oracle' running on Validator nodes.
    It monitors the L1 (Arbitrum) state for risk indicators (Reorgs, Gas Spikes, Uncle Rates).
    
    Mechanism:
    - Maintains a local 'risk_score' (0.0 to 1.0).
    - When Sequencer reports a Reorg, risk_score jumps up.
    - over time, if no Reorgs, risk_score decays (Safe).
    - Returns 'required_confirmations' or 'delay_seconds' based on risk.
    """
    def __init__(self, env, config):
        self.env = env
        self.sensitivity = float(config.get("oracle_sensitivity", 1.0))
        self.decay_rate = float(config.get("oracle_decay_rate", 0.1)) # Score decay per second
        self.risk_score = 0.0
        
        # Thresholds
        self.high_risk_threshold = 0.5
        self.safe_delay_s = float(config.get("oracle_safe_delay_s", 0.0))  # Latency in Low Risk (e.g. 0.0)
        self.high_risk_delay_s = float(config.get("oracle_high_risk_delay_s", 12.0)) # Latency in High Risk
        
        self.last_update_ts = env.now

    def on_reorg_event(self, depth):
        """
        Called when a reorg is observed (simulated external feed or p2p gossip).
        """
        self._update_decay()
        # Jump risk score based on depth
        impact = depth * 0.2 * self.sensitivity
        self.risk_score = min(1.0, self.risk_score + impact)
        # print(f"[Oracle] Reorg detected (depth={depth}). Risk Score -> {self.risk_score:.2f}")

    def get_required_delay(self):
        """
        Returns the recommended delay (seconds) before signing.
        """
        self._update_decay()
        
        if self.risk_score > self.high_risk_threshold:
             return self.high_risk_delay_s
        else:
             return self.safe_delay_s

    def _update_decay(self):
        now = self.env.now
        dt = now - self.last_update_ts
        if dt > 0:
            decay = self.decay_rate * dt
            self.risk_score = max(0.0, self.risk_score - decay)
        self.last_update_ts = now
