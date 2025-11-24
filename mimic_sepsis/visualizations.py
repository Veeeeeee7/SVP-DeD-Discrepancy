import numpy as np
import matplotlib.pyplot as plt

def plot_svp_size(svp_sizes, path):
    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0.0, 1.0, len(svp_sizes)), svp_sizes, marker='o')
    plt.title('SVP Sizes vs. Zeta')
    plt.xlabel('Zeta')
    plt.ylabel('SVP Size')
    plt.grid(True)
    plt.savefig(path, dpi=500)
    plt.close()

def plot_ded_size(ded_sizes, path):
    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0.0, 1.0, len(ded_sizes)), ded_sizes, marker='o', color='orange')
    plt.title('DeD Sizes vs. Death Threshold')
    plt.xlabel('Death Threshold')
    plt.ylabel('DeD Size')
    plt.grid(True)
    plt.savefig(path, dpi=500)
    plt.close()

def plot_conflict_fraction_heatmap(conflict_fractions, path):
    plt.figure(figsize=(8, 6))
    plt.imshow(conflict_fractions, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Conflict Fraction')
    plt.title('Conflict Fraction Heatmap')
    plt.xlabel('Death Threshold')
    plt.ylabel('Zeta')
    plt.savefig(path, dpi=500)
    plt.close()

def plot_iou_heatmap(ious, path):
    plt.figure(figsize=(8, 6))
    plt.imshow(ious, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='IoU')
    plt.title('IoU Heatmap')
    plt.xlabel('Death Threshold')
    plt.ylabel('Zeta')
    plt.savefig(path, dpi=500)
    plt.close()

if __name__ == "__main__":
    results_dir = 'results/'
    visualizations_dir = 'visualizations/'
    svp_sizes = np.load(results_dir + 'svp_sizes.npy')
    ded_sizes = np.load(results_dir + 'ded_sizes.npy')
    conflict_fractions = np.load(results_dir + 'conflict_fractions.npy')
    ious = np.load(results_dir + 'ious.npy')


    plot_svp_size(svp_sizes, visualizations_dir + 'svp_sizes.png')
    plot_ded_size(ded_sizes, visualizations_dir + 'ded_sizes.png')
    plot_conflict_fraction_heatmap(conflict_fractions, visualizations_dir + 'conflict_fraction_heatmap.png')
    plot_iou_heatmap(ious, visualizations_dir + 'iou_heatmap.png')