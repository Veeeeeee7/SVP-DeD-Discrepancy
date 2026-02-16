import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
from tqdm import tqdm
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_policy(df_data):
    """
    Create behavior policy π_b from the dataset.
    """
    # count occurrences of each state-action pair
    SA_count = df_data.groupby(['s:state', 'a:action']).size() \
        .unstack().reindex(index=range(nS), columns=range(nA)).fillna(0)

    # behavior policy
    π_b = SA_count.div(SA_count.sum(axis=1), axis=0)

    # only allow actions frequently used by clinicians
    SA_mask = (SA_count >= 6)

    # for states without any "available" actions, allow the most frequent action
    for s in range(nS): # changed from nS-1
        if SA_mask.loc[s].sum() == 0:
            SA_mask.loc[s, SA_count.loc[s].argmax()] = True

    return π_b, SA_mask, SA_count

def make_transition_matrix(df_data, nA, nS_total, S_survival, S_death):
    """
    Create the empirical transition matrix from the dataset.
    """
    # count occurrences of each transition
    SAS_count = df_data.groupby(['s:state', 'a:action', 's:next_state']).size().reset_index(name='count')

    # Create the transition matrix
    P = np.full((nS_total, nA, nS_total), np.nan)
    for i, row in SAS_count.iterrows():
        P[row['s:state'], row['a:action'], row['s:next_state']] = row['count']

    # Normalize the transition matrix
    P = P / np.nansum(P, axis=2, keepdims=True)

    # Set the transition probabilities for terminal states
    P[S_survival, :, :] = 0
    P[S_survival, :, S_survival] = 1
    P[S_death, :, :] = 0
    P[S_death, :, S_death] = 1

    return P

def make_gymP(P, R, nS, nA, nS_total, S_survival, S_death):
    """
    Convert the transition and reward matrices to the gym format.
    """
    gymP = defaultdict(lambda: defaultdict(list))
    for s in range(nS):
        for a in range(nA):
            for next_s in range(nS_total):
                if not np.isnan(P[s, a, next_s]):
                    prob = P[s, a, next_s]
                    reward = R[s, a, next_s]
                    done = int(next_s in [S_survival, S_death])
                    gymP[s][a].append((prob, next_s, reward, done))
    return gymP


def value_iteration_masked(gymP, nS, nA, SA_mask, gamma, theta=1e-10):
    V = np.zeros(nS)
    for _ in tqdm(itertools.count()):
        V_new = V.copy()
        for s in range(nS):
            ## V[s] = max {a} sum {s', r} P[s', r | s, a] * (r + gamma * V[s'])
            Q_s = np.zeros((nA))
            for a in range(nA):
                Q_s[a] = sum(p * (r + (0 if done else gamma * V[s_])) for p, s_, r, done in gymP[s][a])

            Q_s[~SA_mask[s]] = np.nan
            new_v = np.nanmax(Q_s)
            V_new[s] = new_v
        if np.isclose(np.linalg.norm(V_new - V), theta):
            break
        V = V_new


    pi = np.zeros((nS, nA))
    for s in range(nS):
        Q_s = np.zeros(nA)
        for a in range(nA):
            if a in gymP[s]:
                Q_s[a] = sum(
                    p * (r + (0 if done else gamma * V[s_]))
                    for p, s_, r, done in gymP[s][a]
                )

        Q_s[~SA_mask[s]] = np.nan

        if np.all(np.isnan(Q_s)):
            continue

        best_a = np.nanargmax(Q_s)
        pi[s, :] = 0.0
        pi[s, best_a] = 1.0

    return V, pi


def V2Q(P, V, nA, nS, SA_mask, gamma, mode='svp'):
    Q = np.zeros((nS, nA))
    for s in range(nS):
        for a in P[s]:
            if not SA_mask[s, a]:
                Q[s, a] = np.nan
                continue
            for p, s_, r, done in P[s][a]:
                # Q[s, a] = np.sum(p * (r + gamma * V[s_] * (1 - done)) for p, s_, r, done in P[s][a])
                if mode == 'svp':
                    if s_ == 750:  # survival state
                        Q[s, a] += p * (r + gamma * 1 * (1 - done))
                    elif s_ == 751:  # death state
                        Q[s, a] += p * (r + gamma * -1 * (1 - done))
                    else:
                        Q[s, a] += p * (r + gamma * V[s_] * (1 - done))
                elif mode == 'ded':
                    if s_ == 750:  # survival state
                        Q[s, a] += p * (r + gamma * 0 * (1 - done))
                    elif s_ == 751:  # death state
                        Q[s, a] += p * (r + gamma * -1 * (1 - done))
                    else:
                        Q[s, a] += p * (r + gamma * V[s_] * (1 - done))
    return Q

def svp_masked(P, V_star, nS, nA, SA_mask, gamma, zeta, theta=1e-10, max_iter=1000, min_iter=10):
    is_max_iter = False
    V = V_star.copy().astype(float)
    policies = []
    svp_policy = None
    n_iter = 0

    Q = V2Q(P, V, nA, nS, SA_mask, gamma, mode='svp')
    pi = np.zeros((nS, nA), dtype=float)
    for s in range(nS):
        Q_s = Q[s].copy()

        # set invalid actions to nan
        Q_s[~SA_mask[s]] = np.nan
        pi[s] = (Q_s >= (1 - zeta) * V_star[s]) & SA_mask[s]

    policies.append(pi)

    while True:
        delta = 0.0
        for s in range(nS):
            old_v = V[s]
            Q_s = np.zeros(nA)

            # exploratory policy (compute one-step lookahead state-value function Q)
            for a in P[s]:
                # Q_s[a] = np.sum(p * (r + gamma * V[s_] * (1 - done)) for p, s_, r, done in P[s][a])
                for p, s_, r, done in P[s][a]:
                    if s_ == 750:  # survival state
                        Q_s[a] += p * (r + gamma * 1 * (1 - done))
                    elif s_ == 751:  # death state
                        Q_s[a] += p * (r + gamma * -1 * (1 - done))
                    else:
                        Q_s[a] += p * (r + gamma * V[s_] * (1 - done))
            # set invalid actions to nan
            Q_s[~SA_mask[s]] = np.nan

            # determine cutoff for both near greedy beneficial and harmful actions
            if V_star[s] >= 0:
                Q_cutoff = (1 - zeta) * V_star[s]
            else:
                Q_cutoff = V_star[s] - zeta * abs(V_star[s])

            # find indices of actions that meet the cutoff and are valid
            Pi_S = np.argwhere((Q_s >= Q_cutoff) & SA_mask[s])

            # update state-value function V using the best action from the selected set or worst action if none selected
            if len(Pi_S) > 0:
                new_v = Q_s[Pi_S].min()
            else:
                new_v = Q_s.max()

            V[s] = new_v
            delta = max(delta, np.abs(new_v - old_v))

        # update policy
        Q = V2Q(P, V, nA, nS, SA_mask, gamma, mode='svp')
        pi = np.zeros((nS, nA))
        for s in range(nS):
            if V_star[s] >= 0:
                threshold = (1 - zeta) * V_star[s]
            else:
                threshold = V_star[s] - zeta * abs(V_star[s])
            pi[s] = (Q[s] >= threshold) & SA_mask[s]
        

        n_iter += 1
        if n_iter < min_iter:
            continue

        if ((policies[-1] == pi).all() and delta < theta):
            svp_policy = pi
            iter = n_iter
            break

        is_cycle = False
        for i, past_pi in enumerate(policies):
            if (past_pi == pi).all():
                is_cycle = True
                cycle_start = i
                cycle_policies = policies[cycle_start:]
                core_bool = (cycle_policies[0] > 0)
                for pol in cycle_policies[1:]:
                    core_bool &= (pol > 0)
                svp_policy = core_bool.astype(float)
        if is_cycle:
            iter = n_iter
            break
            
        if n_iter >= max_iter:
            svp_policy = pi
            is_max_iter = True
            iter = n_iter
            break
            
        policies.append(pi)

    # uses optimal actions for states without recommendations
    # _, optimal_policies = value_iteration_masked(P, nS, nA, SA_mask, gamma, theta=theta)
    # for s in range(nS):
    #     recommendation_exist = False
    #     for a in range(nA):
    #         if svp_policy[s][a] != 0 & SA_mask[s][a]: recommendation_exist = True
    #     if not recommendation_exist:
    #         for a in range(nA):
    #             if SA_mask[s][a]:
    #                 svp_policy[s][a] = optimal_policies[s][a]

    return V, svp_policy, is_max_iter, iter

def ded_deadend(Q_d, nS, nA, SA_mask, threshold):
    pi_ded_deadend = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            if Q_d[s, a] <= -threshold and SA_mask[s, a]:
                pi_ded_deadend[s, a] = 1
    return pi_ded_deadend

def compute_conflict_fraction(pi1, pi2, SA_mask, nS, nA):
    conflicts = np.zeros(nS)
    for s in range(nS):
        if any(pi1[s, a] and pi2[s, a] for a in range(nA)):
            conflicts[s] += 1
        elif all(SA_mask[s, a] == False for a in range(nA)):
            conflicts[s] = np.nan
    return np.nanmean(conflicts)

def compute_iou(pi1, pi2, SA_mask, nS, use_0_for_empty=False):
    ious = np.zeros(nS)
    for s in range(nS):
        if all(SA_mask[s, a] == False for a in range(nA)):
            ious[s] = np.nan
            continue
        actions1 = set(np.where(pi1[s] > 0)[0])
        actions2 = set(np.where(pi2[s] > 0)[0])
        intersection = actions1.intersection(actions2)
        union = actions1.union(actions2)
        if len(union) > 0:
            ious[s] = len(intersection) / len(union)
        else:
            if use_0_for_empty:
                ious[s] = 0.0
            else:
                ious[s] = 1.0

    return np.nanmean(ious)

def compute_top_k_conflict_states(pi1, pi2, nS, nA, k):
    conflicts = np.zeros(nS)
    for s in range(nS):
        for a in range(nA):
            if pi1[s, a] and pi2[s, a]:
                conflicts[s] += 1
    top_k_states = np.argsort(conflicts)[-k:][::-1]
    return top_k_states

def compute_conflict_set_for_state(pi1, pi2, state, nA):
    conflict_actions = []
    for a in range(nA):
        if pi1[state, a] and pi2[state, a]:
            conflict_actions.append(a)
    return conflict_actions

if __name__ == "__main__":
    nS = 750
    nA = 25
    S_survival = 750
    S_death = 751
    nS_total = nS + 2

    # Load the data
    data_dir = 'mimic_sepsis_data_2025/'
    train_df = pd.read_csv(data_dir + 'traj_shifted_train.csv', dtype={"a:action": "Int64", 'a:next_action': "Int64"})
    val_df = pd.read_csv(data_dir + 'traj_shifted_val.csv', dtype={"a:action": "Int64", 'a:next_action': "Int64"})
    test_df = pd.read_csv(data_dir + 'traj_shifted_test.csv', dtype={"a:action": "Int64", 'a:next_action': "Int64"})

    # Create behavior policy and transition matrix
    train_π_b, train_SA_mask, train_SA_count = make_policy(train_df)
    train_P = make_transition_matrix(train_df, nA, nS_total, S_survival, S_death)
    val_π_b, val_SA_mask, val_SA_count = make_policy(val_df)
    val_P = make_transition_matrix(val_df, nA, nS_total, S_survival, S_death)

    trajectories = train_df['traj'].unique()
    patient_total = len(trajectories)
    death_total = 0
    state_action_deaths = set()
    state_action_total = nS * nA - np.sum(train_SA_mask.values == False)
    for traj_id in trajectories:
        traj_data = train_df[train_df['traj'] == traj_id]
        last_row = traj_data.iloc[-1]
        last_state = last_row['s:state']
        last_action = last_row['a:action']
        end_state = last_row['s:next_state']
        if end_state == S_death:
            state_action_deaths.add((last_state, last_action))
            death_total += 1
    state_action_deaths = len(state_action_deaths)
    death_rate_per_patient = death_total / patient_total

    print(f'Death rate per state in training data: {death_total}/{patient_total} = {death_rate_per_patient:.4f}')
    # print(f'Death rate per state, action in training data: {state_action_deaths}/{state_action_total} = {state_action_deaths/state_action_total:.4f}')


    # SVP
    zeta = 0.2
    print(f"Choose zeta={zeta} for SVP and death rate threshold={death_rate_per_patient:.4f} for DeD")
    train_R_svp = np.zeros((nS_total, nA, nS_total))
    train_R_svp[:, :, S_survival] = 1
    train_R_svp[:, :, S_death] = -1
    train_P_svp = make_gymP(train_P, train_R_svp, nS, nA, nS_total, S_survival, S_death)
    train_V_star, train_π_star = value_iteration_masked(train_P_svp, nS, nA, train_SA_mask.values, gamma=1.0, theta=1e-10)
    train_V_star_svp, train_π_svp, is_max_iter, iter = svp_masked(train_P_svp, train_V_star, nS, nA, train_SA_mask.values, gamma=1.0, zeta=zeta, theta=1e-10)
    train_size_svp = np.mean(np.sum(train_π_svp, axis=1))
    print(f"SVP average policy size: {train_size_svp}")

    zeta2 = 0.8
    print(f'Second zeta={zeta2}')
    train_V_star_svp2, train_π_svp2, is_max_iter2, iter2 = svp_masked(train_P_svp, train_V_star, nS, nA, train_SA_mask.values, gamma=1.0, zeta=zeta2, theta=1e-10)
    train_size_svp2 = np.mean(np.sum(train_π_svp2, axis=1))
    print(f"SVP (zeta={zeta2}) average policy size: {train_size_svp2}")

    # DeD
    train_R_ded = np.zeros((nS_total, nA, nS_total))
    train_R_ded[:, :, S_death] = -1
    train_P_ded = make_gymP(train_P, train_R_ded, nS, nA, nS_total, S_survival, S_death)
    train_V_star_ded, train_π_star_ded = value_iteration_masked(train_P_ded, nS, nA, train_SA_mask.values, gamma=1.0, theta=1e-10)
    train_Q_ded = V2Q(train_P_ded, train_V_star_ded, nA, nS, train_SA_mask.values, 1.0, mode='ded')
    train_π_ded = ded_deadend(train_Q_ded, nS, nA, train_SA_mask.values, threshold=death_rate_per_patient)
    train_size_ded = np.mean(np.sum(train_π_ded, axis=1))
    print(f"DeD average policy size: {train_size_ded}")

    # Find indices in train_π_svp that conflict with train_π_ded
    conflict_states_svp_ded = []
    for s in range(nS):
        for a in range(nA):
            if train_π_svp[s, a] and train_π_ded[s, a]:
                conflict_states_svp_ded.append([s,a])
                break
    

    print(f"\nConflicting states between SVP (zeta={zeta}) and DeD: {len(conflict_states_svp_ded)} states")
    print(f"Conflict state indices: {conflict_states_svp_ded}")

    # Find indices in train_π_svp2 that conflict with train_π_ded
    conflict_states_svp2_ded = []
    for s in range(nS):
        for a in range(nA):
            if train_π_svp2[s, a] and train_π_ded[s, a]:
                conflict_states_svp2_ded.append([s,a])
                break
    

    print(f"\nConflicting states between SVP (zeta={zeta2}) and DeD: {len(conflict_states_svp2_ded)} states")
    print(f"Conflict state indices: {conflict_states_svp2_ded}")

    # sorting by DeD V-values for visualization
    sort_indices = np.argsort(train_V_star_ded)
    train_Q_ded_ordered = train_Q_ded[sort_indices]
    train_π_svp_ordered = train_π_svp[sort_indices]
    train_π_svp2_ordered = train_π_svp2[sort_indices]

    # Find ordered indices for conflict states in train_π_svp2_ordered
    conflict_indices_svp2 = []
    for s, a in conflict_states_svp2_ded:
        ordered_idx = np.where(sort_indices == s)[0]
        if len(ordered_idx) > 0:
            conflict_indices_svp2.append(ordered_idx[0])
    
    print(f"\nOrdered indices for SVP2 conflict states: {conflict_indices_svp2}")

    hist_list = [i in  conflict_indices_svp2 for i in range(nS)]

    def plot_true_hist_by_bucket(bool_list, bucket_size=50, title=None):
        b = np.asarray(bool_list, dtype=bool)
        n = b.size
        if n == 0:
            raise ValueError("bool_list is empty")

        # number of buckets (ceil)
        nb = (n + bucket_size - 1) // bucket_size

        # count Trues per bucket
        counts = np.array([b[i*bucket_size : (i+1)*bucket_size].sum() for i in range(nb)])

        # x positions = bucket index (0,1,2,...)
        x = np.arange(nb)

        plt.figure()
        plt.bar(x, counts)
        plt.xlabel(f"Bucket index (size={bucket_size})")
        plt.ylabel("# True values in bucket")
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.show()
    plot_true_hist_by_bucket(hist_list, bucket_size=50)

    conflict_fraction = compute_conflict_fraction(train_π_svp, train_π_ded, train_SA_mask.values, nS, nA)
    iou = compute_iou(train_π_svp, train_π_ded, train_SA_mask.values, nS, use_0_for_empty=True)
    print(f"Conflict fraction between SVP and DeD: {conflict_fraction:.14f}")
    print(f"Mean IOU between SVP and DeD: {iou:.14f}")

    conflict_fraction_2 = compute_conflict_fraction(train_π_svp2, train_π_ded, train_SA_mask.values, nS, nA)
    iou_2 = compute_iou(train_π_svp2, train_π_ded, train_SA_mask.values, nS, use_0_for_empty=True)
    print(f"Conflict fraction between SVP (zeta={zeta2}) and DeD: {conflict_fraction_2:.14f}")
    print(f"Mean IOU between SVP (zeta={zeta2}) and DeD: {iou_2:.14f}")



    # analyze consistency for different patient sickness levels
    # print(f"DeD V-values summary statistics:")
    # print(f"  Mean: {np.nanmean(train_V_star_ded)}")
    # print(f"  Median: {np.nanmedian(train_V_star_ded)}")
    # print(f"  Std: {np.nanstd(train_V_star_ded)}")
    # print(f"  Min: {np.nanmin(train_V_star_ded)}")
    # print(f"  Max: {np.nanmax(train_V_star_ded)}")

    # plt.figure(figsize=(10, 6))
    # plt.hist(train_V_star_ded, bins=30, edgecolor='black', alpha=0.7)
    # plt.xlabel('V-value')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of DeD V-values')
    # plt.grid(axis='y', alpha=0.3)
    # plt.savefig('ded_v_values_histogram.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # Print states with conflicts and their ordered indices
    # conflict_states = []
    # for s in range(nS):
    #     for a in range(nA):
    #         if train_π_svp[s, a] and train_π_ded[s, a]:
    #             conflict_states.append(s)
    #             break
    
    # conflict_states = sorted(set(conflict_states))
    # print(f"\nStates with conflicts between SVP and DeD: {len(conflict_states)} states")
    
    # Find ordered indices (position in sorted array)
    # ordered_conflict_indices = []
    # for s in conflict_states:
    #     ordered_idx = np.where(sort_indices == s)[0]
    #     if len(ordered_idx) > 0:
    #         ordered_conflict_indices.append(ordered_idx[0])
    
    # for s, ordered_idx in zip(conflict_states, ordered_conflict_indices):
    #     print(f"  State {s}: ordered index {ordered_idx}, V-value {train_V_star_ded[s]:.4f}")
    #     # Print Q-values for actions chosen by SVP
    #     svp_actions = np.where(train_π_svp[s] > 0)[0]
    #     for a in svp_actions:
    #         print(f"    SVP action {a}: Q-value {train_Q_ded[s, a]:.4f}")

    # quartiles = [10,20,30,40,50,60,70,80,90]
    # dts_V = np.nanpercentile(train_V_star_ded, quartiles)
    # print(f"  {quartiles}%-tiles: {dts_V}")

    # first_quartile_states = np.where(train_V_star_ded <= dts_V[0])[0]
    # second_quartile_states = np.where((train_V_star_ded > dts_V[0]) & (train_V_star_ded <= dts_V[1]))[0]
    # third_quartile_states = np.where((train_V_star_ded > dts_V[1]) & (train_V_star_ded <= dts_V[2]))[0]
    # fourth_quartile_states = np.where(train_V_star_ded > dts_V[2])[0]

    # for i, state_set in enumerate([first_quartile_states, second_quartile_states, third_quartile_states, fourth_quartile_states]):
    #     quartile_V_star_svp = train_V_star_svp[state_set]
    #     quartile_V_star_ded = train_V_star_ded[state_set]
    #     quartile_nS = len(state_set)
    #     quartile_conflict_fraction = compute_conflict_fraction(train_π_svp[state_set], train_π_ded[state_set], train_SA_mask.values, quartile_nS, nA)
    #     quartile_iou = compute_iou(train_π_svp[state_set], train_π_ded[state_set], train_SA_mask.values, quartile_nS, use_0_for_empty=True)
    #     print(f"Quartile {i+1}:")
    #     print(f"  Number of states: {len(state_set)}")
    #     print(f"  Conflict fraction between SVP and DeD: {quartile_conflict_fraction:.14f}")
    #     print(f"  Mean IOU between SVP and DeD: {quartile_iou:.14f}")



    # k_head_states = 10
    # k_tail_states = 10
    # ELLIPSIS_W = 8  # widen the "..." gap (try 8, 12, 20, etc.)

    # nS, nA = train_Q_ded.shape

    # def make_state_keep(n, k_head, k_tail, ellipsis_w):
    #     """
    #     Returns:
    #     keep_states: list of original state indices kept
    #     x_pos: dict original_state -> display_x_position
    #     has_ellipsis: bool
    #     """
    #     if n <= (k_head + k_tail):
    #         keep = list(range(n))
    #         pos = {s: i for i, s in enumerate(keep)}
    #         return keep, pos, False

    #     head = list(range(k_head))
    #     tail = list(range(n - k_tail, n))
    #     keep = head + tail

    #     # head positions: 0..k_head-1
    #     pos = {s: i for i, s in enumerate(head)}
    #     # tail positions start after a wide ellipsis gap
    #     for j, s in enumerate(tail):
    #         pos[s] = k_head + ellipsis_w + j

    #     return keep, pos, True

    # # states to display
    # state_keep, x_pos, state_has_ellipsis = make_state_keep(
    #     nS, k_head_states, k_tail_states, ELLIPSIS_W
    # )

    # disp_n_states = len(state_keep) + (ELLIPSIS_W if state_has_ellipsis else 0)

    # # ---- build display matrix: all actions, only selected states ----
    # # shape: (actions, displayed_states)
    # Q_disp = np.full((nA, disp_n_states), np.nan, dtype=float)

    # for s in state_keep:
    #     xs = x_pos[s]
    #     # train_Q_ded_ordered is (state, action), heatmap expects (action, state)
    #     Q_disp[:, xs] = train_Q_ded_ordered[s, :]

    # # ---- plot ----
    # fig = plt.figure(
    #     figsize=(max(12, disp_n_states * 0.5), max(8, nA * 0.35)),
    #     constrained_layout=True
    # )
    # ax = fig.add_subplot(111)

    # # Create 10 discrete bins
    # bins = np.linspace(-1, 0, 11)  # 10 bins
    # cmap = plt.cm.get_cmap('Reds_r', 10)
    # norm = plt.Normalize(vmin=-1, vmax=0)
    
    # cbar = sns.heatmap(
    #     Q_disp,
    #     cmap=cmap,
    #     norm=norm,
    #     ax=ax,
    #     cbar=True,
    #     cbar_kws={"label": "Q-value"},
    # ).collections[0].colorbar
    # if cbar:
    #     cbar.ax.tick_params(labelsize=12)
    #     cbar.set_label("Q-value", fontsize=18)

    # # ---- mark SVP selections with X markers (Option A) ----
    # svp_states, svp_actions = np.where(train_π_svp_ordered > 0)

    # for s, a in zip(svp_states, svp_actions):
    #     if s in x_pos:  # state is displayed
    #         xs = x_pos[s]
    #         ax.scatter(
    #             xs + 0.5, a + 0.5,   # center of cell
    #             marker="x",
    #             s=80,                # marker size
    #             linewidths=2.0,
    #             color="blue"
    #         )

    # # ---- ticks & labels ----
    # ax.set_xlabel("State", fontsize=18)
    # ax.set_ylabel("Action", fontsize=18)

    # # x ticks: label first 10, one centered "...", and last 10
    # xtick_pos = []
    # xtick_lab = []

    # # head labels
    # head_count = min(k_head_states, nS)
    # for s in range(head_count):
    #     if s in x_pos:
    #         xtick_pos.append(x_pos[s] + 0.5)
    #         xtick_lab.append(f"{train_V_star_ded[sort_indices[s]]:.2f}")

    # # ellipsis label centered in the wide gap
    # if state_has_ellipsis:
    #     ell_center = k_head_states + (ELLIPSIS_W - 1) / 2
    #     xtick_pos.append(ell_center + 0.5)
    #     xtick_lab.append("...")

    #     # tail labels
    #     for s in range(nS - k_tail_states, nS):
    #         if s in x_pos:
    #             xtick_pos.append(x_pos[s] + 0.5)
    #             xtick_lab.append(f"{train_V_star_ded[sort_indices[s]]:.2f}")
    # else:
    #     # no ellipsis case: label all displayed states
    #     xtick_pos = [i + 0.5 for i in range(disp_n_states)]
    #     xtick_lab = [f"{train_V_star_ded[sort_indices[i]]:.2f}" for i in range(disp_n_states)]

    # ax.set_xticks(xtick_pos)
    # ax.set_xticklabels(xtick_lab, rotation=90, fontsize=12)
    # ax.tick_params(axis="x", length=1, pad=1)

    # # y ticks: show ALL actions 1..nA
    # y_pos = np.arange(nA)
    # y_lab = [str(a + 1) for a in y_pos]
    # ax.set_yticks(y_pos + 0.5)
    # ax.set_yticklabels(y_lab, rotation=0, fontsize=12)
    # ax.tick_params(axis="y", length=1, pad=1)

    # # fig.suptitle("DeD Q-values with SVP-selected (s,a) marked", y=1.02)

    # plt.savefig("ded_q_values_with_svp_xmarks.png", dpi=500, bbox_inches="tight")
    # plt.close(fig)

    # # ---- Version 2: Only first 8 states ----
    # k_head_states_v2 = 8
    # k_tail_states_v2 = 0
    
    # state_keep_v2 = list(range(k_head_states_v2))
    # disp_n_states_v2 = len(state_keep_v2)
    
    # # ---- build display matrix for v2 ----
    # Q_disp_v2 = np.full((disp_n_states_v2, nA), np.nan, dtype=float)
    
    # for i, s in enumerate(state_keep_v2):
    #     Q_disp_v2[i, :] = train_Q_ded_ordered[s, :]
    
    # # ---- plot v2 ----
    # fig_v2 = plt.figure(
    #     figsize=(16, 8),
    #     constrained_layout=True
    # )
    # ax_v2 = fig_v2.add_subplot(111)
    
    # cmap_v2 = plt.cm.get_cmap('Reds_r')
    # norm_v2 = plt.Normalize(vmin=-1, vmax=0)
    
    # cbar_v2 = sns.heatmap(
    #     Q_disp_v2,
    #     cmap=cmap_v2,
    #     norm=norm_v2,
    #     ax=ax_v2,
    #     cbar=True,
    #     cbar_kws={"label": "Q-value"},
    # ).collections[0].colorbar
    # if cbar_v2:
    #     cbar_v2.ax.tick_params(labelsize=12)
    #     cbar_v2.set_label("Q-value", fontsize=18)
    
    # # ---- mark SVP selections for v2 ----
    # for s, a in zip(svp_states, svp_actions):
    #     if s in state_keep_v2:
    #         s_idx = state_keep_v2.index(s)
    #         ax_v2.scatter(
    #             a + 0.5, s_idx + 0.5,
    #             marker="x",
    #             s=80,
    #             linewidths=2.0,
    #             color="blue"
    #         )
    
    # ax_v2.set_xlabel("Action", fontsize=18)
    # ax_v2.set_ylabel("V-Values of States with Conflict", fontsize=18)
    
    # # ---- ticks & labels for v2 ----
    # xtick_pos_v2 = np.arange(nA)
    # xtick_lab_v2 = [str(a + 1) for a in xtick_pos_v2]
    # ax_v2.set_xticks(xtick_pos_v2 + 0.5)
    # ax_v2.set_xticklabels(xtick_lab_v2, rotation=0, fontsize=12)
    # ax_v2.tick_params(axis="x", length=1, pad=1)
    
    # ytick_pos_v2 = []
    # ytick_lab_v2 = []
    # for i, s in enumerate(state_keep_v2):
    #     ytick_pos_v2.append(i + 0.5)
    #     ytick_lab_v2.append(f"{train_V_star_ded[sort_indices[s]]:.2f}")
    
    # ax_v2.set_yticks(ytick_pos_v2)
    # ax_v2.set_yticklabels(ytick_lab_v2, rotation=0, fontsize=12)
    # ax_v2.tick_params(axis="y", length=1, pad=1)
    
    # plt.savefig("ded_q_values_with_svp_xmarks_first8.png", dpi=500, bbox_inches="tight")
    # plt.close(fig_v2)


    additional_indices = np.where(sort_indices == 546)[0].tolist() + np.where(sort_indices == 47)[0].tolist()
    points = [
        (211, 23), (323, 19), (444, 24), (490, 14), (531, 5), (540, 14), (728, 24), (177, 13), (52, 19)
    ]
    red_circle_points = []
    for point in points:
        orig_state, action = point
        ordered_idx = np.where(sort_indices == orig_state)[0]
        if len(ordered_idx) > 0:
            red_circle_points.append((ordered_idx[0], action))

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # --- settings ---
    N_MAIN_STATES = 16
    THRESH = 0.0973

    # Manually choose additional ORIGINAL state indices (i.e., indices into train_*_ordered)
    # additional_indices = [50, 123, 402]   # <-- edit this

    # Manually choose specific (state_index, action_index) points to highlight with a RED circle.
    # IMPORTANT: state_index here is an ORIGINAL index into train_*_ordered (same indexing as additional_indices).
    # Example:
    # red_circle_points = [
    #     (0, 3),
    #     (50, 10),
    #     (402, 24),
    # ]  # <-- edit this

    # Optional: keep additional_indices unique in the order typed
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    seen = set()
    additional_indices = [i for i in additional_indices if not (i in seen or seen.add(i))]

    # --- build combined row index list: main (top) + separator row + additional (bottom) ---
    total_states = train_Q_ded_ordered.shape[0]

    main_rows = list(range(min(N_MAIN_STATES, total_states)))
    add_rows = [i for i in additional_indices if 0 <= i < total_states and i not in main_rows]

    SEP = None  # sentinel for the "..." row
    row_sel = main_rows + ([SEP] if len(add_rows) > 0 else []) + add_rows

    # map from original row index -> plotted y coordinate
    orig_to_y = {}
    y = 0
    for r in row_sel:
        if r is SEP:
            y += 1
            continue
        orig_to_y[r] = y
        y += 1

    # --- construct plotting arrays with a NaN separator row (so it draws as blank) ---
    nA = train_Q_ded_ordered.shape[1]
    Q_rows, Pi1_rows, Pi2_rows, yticklabels = [], [], [], []

    for r in row_sel:
        if r is SEP:
            Q_rows.append(np.full((nA,), np.nan))
            Pi1_rows.append(np.zeros((nA,), dtype=int))
            Pi2_rows.append(np.zeros((nA,), dtype=int))
            yticklabels.append("...")
        else:
            Q_rows.append(train_Q_ded_ordered[r, :])
            Pi1_rows.append(train_π_svp_ordered[r, :])
            Pi2_rows.append(train_π_svp2_ordered[r, :])
            yticklabels.append(f"State {sort_indices[r]}")

    Q   = np.vstack(Q_rows)
    Pi1 = np.vstack(Pi1_rows)
    Pi2 = np.vstack(Pi2_rows)

    if not (Q.shape == Pi1.shape == Pi2.shape):
        raise ValueError(f"Shape mismatch: Q {Q.shape}, Pi1 {Pi1.shape}, Pi2 {Pi2.shape}. Expected all equal.")

    nS, nA = Q.shape

    # --- figure/axes with room for colorbar on the right ---
    fig, ax = plt.subplots(figsize=(14, 10))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.25)

    # --- heatmap of DeD Q-values ---
    cmap = plt.get_cmap("Reds_r").copy()
    cmap.set_bad(color="white")  # NaN separator row becomes white
    im = ax.imshow(Q, aspect="auto", interpolation="nearest", cmap=cmap)

    # --- axes ticks/labels (all tick labels fontsize 20) ---
    ax.set_xticks(np.arange(nA))
    ax.set_xticklabels([f"Action {j}" for j in range(nA)], rotation=45, ha="right", fontsize=20)
    ax.set_yticks(np.arange(nS))
    ax.set_yticklabels(yticklabels, fontsize=20)

    # --- colorbar ---
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"DeD $Q$-values", fontsize=32)
    cbar.ax.tick_params(labelsize=20)

    # --- 1) DeD crosses: red where Q < -THRESH ---
    ys_ded, xs_ded = np.where(Q < -THRESH)
    ax.scatter(xs_ded, ys_ded, marker="x", s=175, linewidths=2, c="red")

    # --- 2) SVP1: dark blue circles where Pi1 == 1 ---
    ys1, xs1 = np.where(Pi1 == 1)
    ax.scatter(xs1, ys1, marker="o", s=120, facecolors="none", edgecolors="blue", linewidths=2)

    # --- 3) SVP2: light blue bigger circles where Pi2 == 1 ---
    light_blue = "#6FA8FF"
    ys2, xs2 = np.where(Pi2 == 1)
    ax.scatter(xs2, ys2, marker="o", s=400, facecolors="none", edgecolors=light_blue, linewidths=2)

    # --- 4) Custom black circles at specific (original_state, action) points ---
    rx, ry = [], []
    for (orig_state, action) in red_circle_points:
        if orig_state in orig_to_y and 0 <= action < nA:
            ry.append(orig_to_y[orig_state])
            rx.append(action)

    if len(rx) > 0:
        ax.scatter(
            rx, ry,
            marker="o",
            s=75,
            facecolors="black",
            edgecolors="black",
            linewidths=0
        )

    # --- optional: draw a horizontal divider line through the "..." row ---
    if SEP in row_sel:
        sep_y = row_sel.index(SEP)
        ax.hlines(sep_y, -0.5, nA - 0.5, colors="black", linewidth=1.0, alpha=0.35)

    # --- keep grid cells aligned ---
    ax.set_xlim(-0.5, nA - 0.5)
    ax.set_ylim(nS - 0.5, -0.5)

    plt.tight_layout()
    plt.savefig("mimic_state_level_plot.pdf", dpi=800, bbox_inches="tight")
    plt.show()
    plt.close(fig)