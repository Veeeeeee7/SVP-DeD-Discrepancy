# visualize_search_pair.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_inconsistency_heatmap(zeta_vals, dt_vals, incons_map):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(incons_map, origin='lower', cmap='Reds', vmin=0, vmax=1, aspect='auto')

    # pick every 5th index for ticks
    xtick_idxs = np.arange(0, len(zeta_vals), 5)
    ytick_idxs = np.arange(0, len(dt_vals), 5)

    ax.set_xticks(xtick_idxs)
    ax.set_xticklabels([f"{zeta_vals[i]:.2f}" for i in xtick_idxs], rotation=45)
    ax.set_yticks(ytick_idxs)
    ax.set_yticklabels([f"{dt_vals[i]:.2f}"   for i in ytick_idxs])

    ax.set_xlabel("ζ (zeta)")
    ax.set_ylabel("death_threshold")
    ax.set_title("Policy Inconsistency\n(green = fully consistent)")

    # overlay green square on zero‐conflict cells
    zero_cells = np.argwhere(incons_map == 0)
    for i, j in zero_cells:
        rect = patches.Rectangle(
            (j - 0.5, i - 0.5),  # lower-left corner
            1, 1,                # width, height
            facecolor="green",
            edgecolor="none",
            alpha=0.8
        )
        ax.add_patch(rect)
    plt.tight_layout()

def visualize_svp_size(zeta_vals, dt_vals, svp_map):
    # pick every 5th index for ticks
    xtick_idxs = np.arange(0, len(zeta_vals), 5)
    ytick_idxs = np.arange(0, len(dt_vals), 5)

    xtick_labels = [f"{zeta_vals[i]:.2f}" for i in xtick_idxs]
    ytick_labels = [f"{dt_vals[i]:.2f}"   for i in ytick_idxs]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(svp_map, origin='lower', cmap='viridis', aspect='auto')
    ax.set_xticks(xtick_idxs)
    ax.set_xticklabels(xtick_labels, rotation=45)
    ax.set_yticks(ytick_idxs)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel("ζ (zeta)")
    ax.set_ylabel("death_threshold")
    ax.set_title("Avg SVP policy size")
    cbar = plt.colorbar(im, ax=ax, label="actions per state")
    plt.tight_layout()
    plt.savefig('figures/avg svp.pdf', bbox_inches='tight')
    plt.show()


def visualize_bad_size(zeta_vals, dt_vals, bad_map):
    # pick every 5th index for ticks
    xtick_idxs = np.arange(0, len(zeta_vals), 5)
    ytick_idxs = np.arange(0, len(dt_vals), 5)

    xtick_labels = [f"{zeta_vals[i]:.2f}" for i in xtick_idxs]
    ytick_labels = [f"{dt_vals[i]:.2f}"   for i in ytick_idxs]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(bad_map, origin='lower', cmap='viridis', aspect='auto')
    ax.set_xticks(xtick_idxs)
    ax.set_xticklabels(xtick_labels, rotation=45)
    ax.set_yticks(ytick_idxs)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel("ζ (zeta)")
    ax.set_ylabel("death_threshold")
    ax.set_title("Avg eliminated policy size")
    cbar = plt.colorbar(im, ax=ax, label="eliminated actions per state")
    plt.tight_layout()
    plt.savefig('figures/avg eliminated.pdf', bbox_inches='tight')
    plt.show()


def avg_svp(zeta_vals, svp_map):
    svp_size = svp_map[0]

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(zeta_vals, svp_size, linestyle='-', color='green')
    ax.set_xlabel("ζ (zeta)")
    ax.set_ylabel("Avg SVP Size")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('figures/svp_vs_zeta.pdf', bbox_inches='tight')
    plt.show()

def avg_ded(dt_vals, ded_map):
    mean_bad = ded_map.mean(axis=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(dt_vals, mean_bad, color='red', linestyle='-')
    ax.set_xlabel("death_threshold")
    ax.set_ylabel("Avg DeD size")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('figures/ded_vs_dt.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # load trained maps
    with open("results/trained_pairs_full.pkl","rb") as f:
        data = pickle.load(f)

    zeta_vals     = data['zeta_vals']
    dt_vals       = data['dt_vals']
    incons_map    = data['incons_map']
    svp_size_map  = data['svp_size_map']
    bad_size_map  = data['bad_size_map']

    # draw
    visualize_inconsistency_heatmap(zeta_vals, dt_vals, incons_map)
    visualize_svp_size(zeta_vals, dt_vals, svp_size_map)
    visualize_bad_size(zeta_vals, dt_vals, bad_size_map)
    avg_svp(zeta_vals, svp_size_map)
    avg_ded(dt_vals, bad_size_map)
    plt.show()
