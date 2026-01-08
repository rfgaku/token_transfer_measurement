import matplotlib.pyplot as plt

def draw_clean_architecture():
    # Canvas size slightly increased, coordinates adjusted to avoid overlaps.
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Style definitions
    box_props = dict(boxstyle="round,pad=0.6", ec="black", lw=2)
    process_props = dict(boxstyle="round,pad=0.4", ec="blue", lw=2, ls="--")
    container_props = dict(boxstyle="round,pad=0.8", ec="gray", lw=3, ls="-", fc="none")
    
    # --- 1. Main Container: SimPy Environment ---
    # Use a large text box as container frame
    ax.text(7.5, 5.0, "", ha="center", va="center", size=40, bbox=dict(**container_props))
    ax.text(7.5, 9.5, "SimPy Simulation Environment (Global Time & Event Loop)", fontsize=16, fontweight="bold", color="#555555", ha="center")

    # --- 2. Input / Output (External) ---
    # Input (Top Left)
    ax.text(1.5, 8.5, "Input:\nParameters.csv\n(Config)", ha="center", va="center", size=11, bbox=dict(fc="#e1d5e7", **box_props))
    # Output (Top Right)
    ax.text(13.5, 8.5, "Output:\nTrace Logs (CSV)\nHistograms (PNG)", ha="center", va="center", size=11, bbox=dict(fc="#e1d5e7", **box_props))

    # --- 3. Internal Components (Agents & Modules) ---
    
    # Traffic Generator Process (Left)
    ax.text(2.5, 6.5, "Traffic Generator\n[SimPy Process]", ha="center", va="center", size=12, bbox=dict(fc="#dae8fc", **process_props))
    
    # Sequencer Class (Center)
    ax.text(6.5, 6.5, "Class: Sequencer\n(Mempool & Batching Logic)", ha="center", va="center", size=12, bbox=dict(fc="#d5e8d4", **box_props))
    
    # Network Module (Center Bottom)
    ax.text(7.5, 3.5, "Module: Network Physics\n(Latency Calculation, Jitter, Packet Loss)", ha="center", va="center", size=12, bbox=dict(fc="#ffe6cc", **box_props))

    # Validators Container (Right Side)
    # Frame for validators
    ax.text(11.5, 4.5, "", ha="center", va="center", size=25, bbox=dict(fc="none", ec="orange", lw=2, ls="--"))
    ax.text(11.5, 7.0, "Hyperliquid Validators\n(N Processes)", fontsize=12, color="orange", fontweight="bold", ha="center")
    
    # Validator Instances
    ax.text(11.5, 5.8, "Class: Validator [i]\n(Polling & State)", ha="center", va="center", size=11, bbox=dict(fc="#fff2cc", **box_props))
    ax.text(11.5, 4.5, ". . .", ha="center", va="center", size=16, fontweight="bold")
    ax.text(11.5, 3.2, "Class: Validator [j]\n(Consensus Vote)", ha="center", va="center", size=11, bbox=dict(fc="#fff2cc", **box_props))

    # L1 Interaction Component (Bottom Center)
    ax.text(6.5, 1.5, "L1 Interface\n(Hard Finality Delay Sim)", ha="center", va="center", size=12, bbox=dict(fc="#f8cecc", **box_props))


    # --- 4. Data Flow Arrows (Adjusted paths) ---
    arrow_args = dict(arrowstyle="->", lw=2, color="#333333")
    
    # Param -> Traffic
    ax.annotate("", xy=(2.5, 7.3), xytext=(1.5, 8.0), arrowprops=arrow_args)
    
    # Traffic -> Sequencer (Tx Injection)
    ax.annotate("Tx Inject", xy=(4.8, 6.5), xytext=(3.8, 6.5), arrowprops=arrow_args, ha="center", va="bottom", fontsize=10)

    # Sequencer -> Network (Broadcast)
    ax.annotate("Broadcast", xy=(7.0, 5.5), xytext=(6.5, 5.8), arrowprops=arrow_args, ha="center", va="top", fontsize=10)
    ax.annotate("", xy=(7.5, 4.3), xytext=(7.0, 5.5), arrowprops=arrow_args) # Path adjustment

    # Network -> Validators (Propagation)
    ax.annotate("P2P Prop.", xy=(10.2, 5.8), xytext=(9.2, 3.5), arrowprops=arrow_args, ha="right", va="top", fontsize=10)
    ax.annotate("", xy=(10.2, 3.2), xytext=(9.2, 3.5), arrowprops=arrow_args)

    # Validators -> Sequencer (Polling Check) - Red Arrow
    ax.annotate("Polling\nCheck", xy=(7.8, 6.2), xytext=(10.2, 5.8), arrowprops=dict(arrowstyle="->", lw=2, color="red", ls="--"), ha="center", va="bottom", color="red", fontsize=10)

    # Sequencer -> L1 (Batch Commit)
    ax.annotate("Batch Commit", xy=(6.5, 2.3), xytext=(6.5, 5.8), arrowprops=arrow_args, ha="center", va="bottom", fontsize=10)

    # Validators -> Output (Logging)
    ax.annotate("Log Data", xy=(13.5, 8.0), xytext=(12.8, 5.8), arrowprops=arrow_args, ha="center", va="bottom", fontsize=10)


    # Title
    ax.text(1, 9.5, "Figure: Simulator Component Architecture", ha="left", va="center", size=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig('result/simulator_architecture_v2.png', dpi=150)
    print("Clean architecture diagram generated.")

if __name__ == "__main__":
    draw_clean_architecture()
