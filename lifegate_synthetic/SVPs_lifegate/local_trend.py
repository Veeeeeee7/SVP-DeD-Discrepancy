import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# Replace this with the actual counts: e.g., np.sum(incons_tensor, axis=(1, 2))
with open("results/trained_pairs_full.pkl", "rb") as f:
    results = pickle.load(f)

incons_tensor = np.array(results["incons_tensor"])
state_conflict_counts = np.sum(incons_tensor, axis=(1, 2))

# Normalize the conflict counts
normalized_conflicts = state_conflict_counts / np.max(state_conflict_counts)

# Reshape to 10x10 grid
grid_conflicts = normalized_conflicts.reshape((10, 10))

# Plot the heatmap
plt.figure(figsize=(6, 6))
plt.imshow(grid_conflicts, cmap='hot_r', origin='upper')
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.tight_layout()

# Save the heatmap
os.makedirs("results/heatmaps", exist_ok=True)
plt.savefig("results/heatmaps/empirical_state_conflict_frequency.pdf")
plt.show()