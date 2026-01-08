
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Load CSV
csv_path = "result/scenario_reorg.csv"
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: {csv_path} not found.")
    sys.exit(1)

# Plot Double Spend Rate vs Mint Delay
plt.figure(figsize=(10, 6))
plt.plot(df["mint_delay_s"], df["double_spend_rate"], marker='o', linestyle='-', color='r', label='Double Spend Rate')

# Add Reorg Depth Line (Theoretical Safety Threshold)
reorg_depth = 5.0 # hardcoded from config
plt.axvline(x=reorg_depth, color='k', linestyle=':', label=f'Reorg Depth ({reorg_depth}s)')

plt.title('Double Spend Risk vs Bridge Wait Time (Reorg Depth=5s)')
plt.xlabel('Mint Delay (s)')
plt.ylabel('Double Spend Probability')
plt.grid(True)
plt.legend()

# Save plot
plt.savefig("result/reorg_safety.png")
print("Plot saved to result/reorg_safety.png")
