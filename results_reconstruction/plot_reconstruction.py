import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def plot_comparison(
    search_pattern="results_*.npz", 
    output_filename="benchmark_comparison.png",
    title="Memory Decay Dynamics: Palimpsa vs Baselines",
    xlim=None
):
    # 1. Find Data
    files = sorted(glob.glob(search_pattern))
    if not files:
        print(f"❌ No files found matching '{search_pattern}'")
        return

    print(f"found {len(files)} runs: {[os.path.basename(f) for f in files]}")

    # 2. Setup Plot
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Use a high-contrast colormap
    colors = plt.cm.get_cmap('tab10', len(files))

    # 3. Plot Loop
    for i, fpath in enumerate(files):
        try:
            data = np.load(fpath)
            mean = data['mean']
            std = data['std']
            x = data['x']
            name = str(data['name'])
            
            # Clean up the name for the legend (optional)
            label = name.replace("results_", "").replace(".npz", "")
            
            color = colors(i)
            
            # Draw Shadow (Std Dev)
            plt.fill_between(
                x, 
                mean - std, 
                mean + std, 
                color=color, 
                alpha=0.1, # Light shadow
                edgecolor=None
            )
            
            # Draw Mean Line
            plt.plot(x, mean, label=label, linewidth=2.5, color=color)
            
        except Exception as e:
            print(f"⚠️ Failed to load {fpath}: {e}")

    # 4. Styling
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Token Index (History Distance)", fontsize=12)
    plt.ylabel("Reconstruction Error (L2 Norm)", fontsize=12)
    
    if xlim:
        plt.xlim(0, xlim)
        
    plt.legend(fontsize=10, loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # 5. Save
    plt.savefig(output_filename)
    print(f"✅ Comparison plot saved to: {output_filename}")
    # plt.show() # Uncomment if running locally

if __name__ == "__main__":
    # You can just run `python plot_results.py`
    plot_comparison()