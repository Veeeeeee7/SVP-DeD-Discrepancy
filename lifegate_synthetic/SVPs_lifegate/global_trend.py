import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_paramwise_inconsistency(results, save_dir="results/heatmaps"):
    os.makedirs(save_dir, exist_ok=True)

    incons_map = results["incons_map"]  # shape: [len(theta), len(zeta)]
    zetas = results["zeta_vals"]
    thetas = results["dt_vals"]

    # Print top-5 most consistent parameter pairs
    flat_consistent = np.argwhere(incons_map == 0)
    print("\nTop 5 parameter pairs with full consistency:")
    for i in range(min(5, len(flat_consistent))):
        ti, zi = flat_consistent[i]
        print(f"- zeta={zetas[zi]:.2f}, theta_D={thetas[ti]:.2f} → Consistent")

    # Generate and save heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(incons_map, cmap="RdYlGn_r", origin="lower", aspect="auto",
               extent=[zetas[0], zetas[-1], thetas[0], thetas[-1]])
    plt.colorbar()
    plt.xlabel("zeta (ζ)")
    plt.ylabel("death_threshold (θ_D)")
    plt.tight_layout()

    heatmap_path = os.path.join(save_dir, "param_inconsistency_heatmap.pdf")
    plt.savefig(heatmap_path)
    plt.show()
    plt.close()

    return heatmap_path

def main():
    results_path = "results/trained_pairs_full.pkl"
    if not os.path.exists(results_path):
        print(f"Error: File '{results_path}' not found.")
        return

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    analyze_paramwise_inconsistency(results)

if __name__ == "__main__":
    main()
