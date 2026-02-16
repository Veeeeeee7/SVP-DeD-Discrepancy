import numpy as np
import pandas as pd

results_dir = 'results/'
visualizations_dir = 'visualizations/'
svp_sizes = np.load(results_dir + 'svp_sizes.npy')
ded_sizes = np.load(results_dir + 'ded_sizes.npy')
conflict_fractions = np.load(results_dir + 'conflict_fractions.npy')
ious = np.load(results_dir + 'ious.npy')

# print(conflict_fractions)

points = [
    (211, 23), (271, 5), (323, 19), (444, 24), (490, 14), (531, 5), (540, 14), (728, 24)
]
red_circle_points = []
for point in red_circle_points:
    print(point)