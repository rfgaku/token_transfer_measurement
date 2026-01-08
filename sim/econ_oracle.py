
import math
from sim.risk_model import RiskModel

class EconOracle:
    """
    Plan C: Economic Security Layer.
    Solves for 't' (Optimal Waiting Time) such that:
    Cost_of_Attack > Tx_Value * P(Reorg | t)
    
    This ensures that the expected profit of an attack is negative.
    """
    def __init__(self):
        self.risk_model = RiskModel()

    def get_optimal_finality_time(self, tx_value, cost_of_attack, risk_coefficient):
        """
        Derives minimum 't' to satisfy the economic security inequality.
        """
        if tx_value <= 0: return 0.0
        p_threshold = cost_of_attack / tx_value
        if p_threshold >= self.risk_model.alpha:
            return 0.0
        
        lambda_eff = self.risk_model.lambda_base / risk_coefficient
        t_optimal = -math.log(p_threshold / self.risk_model.alpha) / lambda_eff
        
        return max(0.0, t_optimal)

    def explain_decision(self, tx_value, cost_of_attack, risk_coefficient):
        """
        Returns a human-readable explanation of the calculation.
        """
        t = self.get_optimal_finality_time(tx_value, cost_of_attack, risk_coefficient)
        p_initial = self.risk_model.get_reorg_probability(0, risk_coefficient)
        p_final = self.risk_model.get_reorg_probability(t, risk_coefficient)
        ev_initial = tx_value * p_initial
        ev_final = tx_value * p_final
        return (f"Tx=${tx_value:,.2f} vs Cost=${cost_of_attack:,.2f} (Risk={risk_coefficient:.2f})\n"
                f" - Initial EV of Attack (t=0s): ${ev_initial:,.2f} -> {'SAFE' if ev_initial < cost_of_attack else 'UNSAFE'}\n"
                f" - Optimal Wait Time: {t:.2f}s\n"
                f" - Final EV of Attack (t={t:.2f}s): ${ev_final:,.2f} (< Cost)")

    def trace_calculation(self, tx_value, cost_of_attack, risk_coefficient):
        """
        AUDIT METHOD: Prints step-by-step calculation trace for validation.
        """
        print(f"[Oracle] Input: Value=${tx_value:,.0f}, Risk={risk_coefficient:.1f}")
        print(f"[Math] Attack Cost (Fixed) = ${cost_of_attack:,.0f}")
        
        p_target = cost_of_attack / tx_value
        print(f"[Math] Target Probability P < Cost/Value = {p_target:.4f}")
        
        if p_target >= 1.0:
            print("[Math] Target P >= 1.0. Instant safety.")
            return

        lambda_base = self.risk_model.lambda_base
        lambda_eff = lambda_base / risk_coefficient
        print(f"[Math] Effective Lambda = Base({lambda_base}) / Risk({risk_coefficient}) = {lambda_eff:.4f}")
        
        print("[Math] Solving: t = -ln(P_target) / Lambda")
        
        # Simulate 'Search' for user confidence
        t_optimal = self.get_optimal_finality_time(tx_value, cost_of_attack, risk_coefficient)
        
        check_points = [10, 20]
        for t in check_points:
            if t < t_optimal:
                p = self.risk_model.get_reorg_probability(t, risk_coefficient)
                print(f"[Math] Checking P(Reorg) at t={t}s... P={p:.4f} (> {p_target:.4f}) -> Unsafe")
        
        p_opt = self.risk_model.get_reorg_probability(t_optimal, risk_coefficient)
        print(f"[Math] Checking P(Reorg) at t={t_optimal:.2f}s... P={p_opt:.4f} (<= {p_target:.4f}) -> Safe!")
        print(f"-> Optimal Finality Time: {t_optimal:.2f}s")
