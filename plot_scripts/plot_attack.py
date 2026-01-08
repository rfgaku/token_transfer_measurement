
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Load CSV
csv_path = "result/attack_scenario_lazy.csv"
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: {csv_path} not found.")
    sys.exit(1)

# Plot Mean Latency vs Byzantine Fraction
plt.figure(figsize=(10, 6))
plt.plot(df["byz_frac"], df["t_quorum_mean"], marker='o', linestyle='-', color='b', label='Mean Latency')

# Highlight Failure Rate on secondary axis
ax2 = plt.gca().twinx()
ax2.plot(df["byz_frac"], df["fail_rate"], marker='x', linestyle='--', color='r', label='Failure Rate')
ax2.set_ylabel('Failure Rate', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Impact of "Lazy Validator" Attack on Deposit Latency (v2.0)')
plt.xlabel('Byzantine Fraction (byz_frac)')
plt.ylabel('Mean Latency (s)')
plt.grid(True)

# Add theoretical limit line
plt.axvline(x=0.333, color='k', linestyle=':', label='PBFT Limit (1/3)')

# Save plot
plt.savefig("result/attack_impact.png")
print("Plot saved to result/attack_impact.png")
