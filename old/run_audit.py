
from sim.econ_oracle import EconOracle

def run_audit_trace():
    oracle = EconOracle()
    # Scenario 4: Stormy Whale
    # Value $10M, Risk 5.0, Cost $1M
    oracle.trace_calculation(10_000_000.0, 1_000_000.0, 5.0)

if __name__ == "__main__":
    run_audit_trace()
