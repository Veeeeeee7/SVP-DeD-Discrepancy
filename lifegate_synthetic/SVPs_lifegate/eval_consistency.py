import pickle
import numpy as np
import os

def evaluate_consistency_when_small_union(results, num_actions=5):
    svp_size_map = results["svp_size_map"]
    bad_size_map = results["bad_size_map"]
    incons_map   = results["incons_map"]

    mask = (svp_size_map + bad_size_map) < num_actions
    tested = np.sum(mask)
    consistent = np.sum((incons_map == 0) & mask)

    if tested == 0:
        print("No states satisfy SVP size + DeD size < total number of actions.")
    else:
        print("Evaluating consistency when SVP size + DeD size < total actions:")
        print(f"- Total parameter pairs tested: {tested}")
        print(f"- Consistent in: {consistent}")
        print(f"- Consistency rate: {consistent / tested:.2%}")

def evaluate_consistency_when_zeta_plus_theta_below_one(results):
    incons_map = results["incons_map"]
    zeta_vals = results['zeta_vals']
    dt_vals = results['dt_vals']

    zeta_grid, theta_grid = np.meshgrid(zeta_vals, dt_vals)
    param_sum = zeta_grid + theta_grid

    mask = param_sum < 1.0
    total = np.sum(mask)
    consistent = np.sum((incons_map == 0) & mask)

    if total == 0:
        print("No parameter pairs satisfy zeta + theta_D < 1.")
    else:
        print("Evaluating consistency when zeta + theta_D < 1:")
        print(f"- Total parameter pairs tested: {total}")
        print(f"- Consistent in: {consistent}")
        print(f"- Consistency rate: {consistent / total:.2%}")

def main():
    results_path = "results/trained_pairs_full.pkl"
    if not os.path.exists(results_path):
        print(f"Error: File '{results_path}' not found.")
        return

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    evaluate_consistency_when_small_union(results)
    evaluate_consistency_when_zeta_plus_theta_below_one(results)
    print(results['combined_size_count'])

if __name__ == "__main__":
    main()
