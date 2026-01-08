
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sim.econ_oracle import EconOracle
from sim.network_monitor import NetworkMonitor

def run_scenarios():
    oracle = EconOracle()
    # Assume fixed Cost of Attack for simulation (e.g., $1,000,000 to revert L1)
    cost_of_attack = 1_000_000.0
    
    scenarios = [
        {"name": "Micro-Payment", "value": 10.0, "risk": 1.0},
        {"name": "Whale Transfer", "value": 10_000_000.0, "risk": 1.0},
        {"name": "Stormy Micro ($10)", "value": 10.0, "risk": 5.0}, 
        {"name": "Stormy Whale ($10M)", "value": 10_000_000.0, "risk": 5.0},
    ]
    
    print("=== Econophysical Dynamic Finality Verification ===")
    print(f"Fixed Cost of Attack: ${cost_of_attack:,.2f}\n")
    
    for s in scenarios:
        print(f"--- Scenario: {s['name']} ---")
        explanation = oracle.explain_decision(s['value'], cost_of_attack, s['risk'])
        print(explanation)
        print("")

def plot_surface():
    oracle = EconOracle()
    cost_of_attack = 1_000_000.0
    
    # Grid
    values = np.logspace(0, 8, 50) # $1 to $100M (Log scale)
    risks = np.linspace(1.0, 10.0, 50) # Risk 1.0 to 10.0
    
    X, Y = np.meshgrid(values, risks)
    Z = np.zeros_like(X)
    
    for i in range(len(risks)):
        for j in range(len(values)):
            Z[i, j] = oracle.get_optimal_finality_time(values[j], cost_of_attack, risks[i])
            
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    # X axis log scale manually for plot
    surf = ax.plot_surface(np.log10(X), Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    
    ax.set_xlabel('Log10(Tx Value) [$]')
    ax.set_ylabel('Network Risk Coeff')
    ax.set_zlabel('Optimal Wait Time [s]')
    ax.set_title(f'Econophysical Dynamic Finality Surface\n(Attack Cost = ${cost_of_attack:,.0f})')
    
    fig.colorbar(surf, shrink=0.5, aspect=5, label="Wait Time (s)")
    
    plt.savefig("econophysical_surface.png")
    print("Generated econophysical_surface.png")

if __name__ == "__main__":
    run_scenarios()
    plot_surface()
