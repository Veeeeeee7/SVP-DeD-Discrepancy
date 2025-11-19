import numpy as np
from lifegate_for_SVPs import LifeGate
from SVPs_algos import value_iter, value_iter_near_greedy, V2Q
import matplotlib.pyplot as plt

# define lifegate simulation
def MDP_lifegate(env, types='regular', deadend_threshold=0.7):
    P = {}
    width, height = env.scr_w, env.scr_h
    for y in range(height):
        for x in range(width):
            s = y * width + x
            P[s] = {}

            # ignore barrier states (cannot get there)
            if [x, y] in env.barriers:
                continue
            for a in env.legal_actions:
                P[s][a] = []
                new_x, new_y = x, y
                reward = 0.0
                done = False

                # if agent is in a dead_end (yellow) state
                if [x, y] in env.dead_ends:

                    # if agent is in a dead_end state next to the death states add possibility of death-dragged death
                    if [x + 1, y] in env.deaths:
                        done = True
                        if types in ['death', 'regular']:
                            reward = -1.0
                    # deadend_threshold% chance to be dragged to death
                    P[s][a].append((deadend_threshold, s + 1, reward, done))
                    P[s][a].append((1 - deadend_threshold, s, 0, False))
                    continue

                # if agent is in death state, guaranteed death with reward -1
                if [new_x, new_y] in env.deaths:
                    if types in ['death', 'regular']:
                        reward = -1.0
                    P[s][a].append((1.0, s, reward, True))
                    continue

                # if agent is in recovery state, guaranteed recovery with reward = 1
                if [new_x, new_y] in env.recoveries:
                    if types in ['recovery', 'regular']:
                        reward = 1.0
                    P[s][a].append((1.0, s, reward, True))
                    continue
                
                # update state based on action
                if a == 1:
                    new_y = y - 1
                elif a == 2:
                    new_y = y + 1
                elif a == 3:
                    new_x = x - 1
                elif a == 4:
                    new_x = x + 1
                
                # prevent agent from moving off the board
                if new_x < 0 or new_y < 0 or new_x >= width or new_y >= height or [new_x, new_y] in env.barriers:
                    new_x, new_y = x, y

                
                # natural drag named "death_drag"
                reward_drag = 0.0
                done_drag = False
                s_next = new_y * width + new_x
                s_drag = s + 1
                # if s_next is in death state, update done and reward
                # if s_next is in death state, there is a death_drag% of being dragged to death and 1-death_drag% of moving to death
                if [new_x, new_y] in env.deaths:
                    done = True
                    if types in ['death', 'regular']:
                        reward = -1.0
                # if s_next is in recovery state, update done and reward
                elif [new_x, new_y] in env.recoveries:
                    done = True
                    if types in ['recovery', 'regular']:
                        reward = 1.0
                # if s_next is next to the death state, update done_drag and reward_drag for after being dragged
                if [x + 1, y] in env.deaths:
                    done_drag = True
                    if types in ['death', 'regular']:
                        reward_drag = -1.0
                
                # add both dragged and not dragged actions
                P[s][a].append((1 - env.death_drag, s_next, reward, done))
                P[s][a].append((env.death_drag, s_drag, reward_drag, done_drag))
    return P


def deadend_policies(Q_d, env_death, env, threshold):
    # Update the death MDP in env_death using the provided env and type 'death'
    env_death.P = MDP_lifegate(env, types='death')
    env_death.nS = env_death.scr_w * env_death.scr_h
    env_death.nA = env_death.nb_actions
    π_bad = np.zeros((env_death.nS, env_death.nA))
    for s in range(env_death.nS):
        for a in range(env_death.nA):
            # Mark action as "bad" if Q_d[s][a] is less than or equal to -threshold
            if Q_d[s][a] <= -threshold:
                π_bad[s][a] = 1
    return π_bad

def rescue_policies(Q_r, env_recovery, env, threshold):
    # Update the recovery MDP in env_recovery using the provided env and type 'recovery'
    env_recovery.P = MDP_lifegate(env, types='recovery')
    env_recovery.nS = env_recovery.scr_w * env_recovery.scr_h
    env_recovery.nA = env_recovery.nb_actions
    π_good = np.zeros((env_recovery.nS, env_recovery.nA))
    for s in range(env_recovery.nS):
        for a in range(env_recovery.nA):
            # Mark action as "good" if Q_r[s][a] is greater than or equal to threshold
            if Q_r[s][a] >= threshold:
                π_good[s][a] = 1
    return π_good

def fraction_conflict(pi_good, pi_bad, valid_states):
    conflicts = np.zeros(len(valid_states))
    for i, s in enumerate(valid_states):
        if any(pi_good[s,a] and pi_bad[s,a] for a in range(pi_bad.shape[1])):
            conflicts[i] += 1
    return np.mean(conflicts)

def compute_intersection_over_union_matrix(policy1, policy2, valid_states, use_0_for_empty=False):
    iou = np.zeros(len(valid_states))
    for i, s in enumerate(valid_states):
        actions1 = set(np.where(policy1[s] > 0)[0])
        actions2 = set(np.where(policy2[s] > 0)[0])
        intersection = actions1.intersection(actions2)
        union = actions1.union(actions2)
        if len(union) > 0:
            iou[i] = len(intersection) / len(union)
        else:
            if use_0_for_empty:
                iou[i] = 0.0
            else:
                iou[i] = 1.0

    return iou

if __name__ == "__main__":
    random_state = np.random.RandomState(42)
    env = LifeGate(state_mode='tabular', rng=random_state, death_drag=0.4, fixed_life=True)
    env.P = MDP_lifegate(env, types='regular')
    env.nS = env.scr_w * env.scr_h
    env.nA = env.nb_actions
    rt_values = np.arange(0, 1.1, 0.1)
    dt_values = np.arange(0, 1.1, 0.1)
    zeta_values = np.arange(0, 1.1, 0.1)

    # *** Compute optimal value function and policy for rescue state discovery ***
    env_r = LifeGate(state_mode='tabular', rng=random_state, death_drag=0.4, fixed_life=True)
    env_r.P = MDP_lifegate(env, types='recovery')
    env_r.nS = env_r.scr_w * env_r.scr_h
    env_r.nA = env_r.nb_actions
    V_r, π_r = value_iter(env_r, gamma=1, theta=1e-10)
    Q_r = V2Q(env_r, V_r, 1)


    # *** Compute optimal value function and policy for deadend state discovery ***
    env_d = LifeGate(state_mode='tabular', rng=random_state, death_drag=0.4, fixed_life=True)
    env_d.P = MDP_lifegate(env, types='death')
    env_d.nS = env_d.scr_w * env_d.scr_h
    env_d.nA = env_d.nb_actions
    V_d, π_d = value_iter(env_d, gamma=1, theta=1e-10)
    Q_d = V2Q(env_d, V_d, 1)

    # *** Compute optimal value function for regular MDP to be used in near-greedy SVP ***
    gamma = 1.0
    theta = 1e-10
    max_iter = 1000
    V_star, _ = value_iter(env, gamma, theta=1e-10)

    svp_ded_rescue_consistency_map = np.zeros((len(zeta_values), len(rt_values)))
    svp_ded_deadend_consistency_map = np.zeros((len(zeta_values), len(dt_values)))
    ded_rescue_ded_deadend_consistency_map = np.zeros((len(rt_values), len(dt_values)))

    svp_ded_rescue_iou = np.zeros((len(zeta_values), len(rt_values)))
    svp_ded_deadend_iou = np.zeros((len(zeta_values), len(dt_values)))
    ded_rescue_ded_deadend_iou = np.zeros((len(rt_values), len(dt_values)))

    all_states = set(range(env.nS))
    death_states = set(y * env.scr_w + x for x, y in env.deaths)
    barrier_states = set(y * env.scr_w + x for x, y in env.barriers)
    lifegate_states = set(y * env.scr_w + x for x, y in env.recoveries)

    valid_states = all_states - death_states - barrier_states - lifegate_states
    for rt in rt_values:
        for zeta in zeta_values:
            π_svp = value_iter_near_greedy(env, gamma, zeta, V_star, theta=theta, max_iter=max_iter)[1]
            π_good = rescue_policies(Q_r, env_r, env, rt)

            svp_ded_rescue_frac_conflict = fraction_conflict(π_svp, π_good, valid_states)
            svp_ded_rescue_iou_matrix = compute_intersection_over_union_matrix(π_svp, π_good, valid_states)

            svp_ded_rescue_consistency_map[int(zeta*10), int(rt*10)] = svp_ded_rescue_frac_conflict
            svp_ded_rescue_iou[int(zeta*10), int(rt*10)] = np.mean(svp_ded_rescue_iou_matrix)
    
    for zeta in zeta_values:
        for dt in dt_values:
            π_svp = value_iter_near_greedy(env, gamma, zeta, V_star, theta=theta, max_iter=max_iter)[1]
            π_deadend = deadend_policies(Q_d, env_d, env, dt)
            
            svp_ded_deadend_frac_conflict = fraction_conflict(π_svp, π_deadend, valid_states)
            svp_ded_deadend_iou_matrix = compute_intersection_over_union_matrix(π_svp, π_deadend, valid_states)

            svp_ded_deadend_consistency_map[int(zeta*10), int(dt*10)] = svp_ded_deadend_frac_conflict
            svp_ded_deadend_iou[int(zeta*10), int(dt*10)] = np.mean(svp_ded_deadend_iou_matrix)


    for rt in rt_values:
        for dt in dt_values:
            π_rescue = rescue_policies(Q_r, env_r, env, rt)
            π_deadend = deadend_policies(Q_d, env_d, env, dt)

            ded_rescue_deadend_frac_conflict = fraction_conflict(π_rescue, π_deadend, valid_states)
            ded_rescue_ded_deadend_iou_matrix = compute_intersection_over_union_matrix(π_rescue, π_deadend, valid_states)

            ded_rescue_ded_deadend_consistency_map[int(rt*10), int(dt*10)] = ded_rescue_deadend_frac_conflict
            ded_rescue_ded_deadend_iou[int(rt*10), int(dt*10)] = np.mean(ded_rescue_ded_deadend_iou_matrix)

    cmap = "viridis"
    vmin, vmax = 0.0, 1.0

    def _plot_heatmap(ax, data, x_vals, y_vals, xlabel, ylabel, title):
        mat = np.ma.masked_invalid(data)
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_xticklabels([f"{v:.1f}" for v in x_vals], rotation=45)
        ax.set_yticklabels([f"{v:.1f}" for v in y_vals])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return im

    plots = [
        (svp_ded_rescue_consistency_map, rt_values, zeta_values, "rescue threshold (rt)", "zeta",
         "SVP vs DED (rescue) - Fraction Conflict", "svp_ded_rescue_fraction_conflict.png"),
        (svp_ded_deadend_consistency_map, dt_values, zeta_values, "deadend threshold (dt)", "zeta",
         "SVP vs DED (deadend) - Fraction Conflict", "svp_ded_deadend_fraction_conflict.png"),
        (ded_rescue_ded_deadend_consistency_map, dt_values, rt_values, "deadend threshold (dt)", "rescue threshold (rt)",
         "DED (rescue) vs DED (deadend) - Fraction Conflict", "ded_rescue_ded_deadend_fraction_conflict.png"),
        (svp_ded_rescue_iou, rt_values, zeta_values, "rescue threshold (rt)", "zeta",
         "SVP vs DED (rescue) - Mean IOU", "svp_ded_rescue_mean_iou.png"),
        (svp_ded_deadend_iou, dt_values, zeta_values, "deadend threshold (dt)", "zeta",
         "SVP vs DED (deadend) - Mean IOU", "svp_ded_deadend_mean_iou.png"),
        (ded_rescue_ded_deadend_iou, dt_values, rt_values, "deadend threshold (dt)", "rescue threshold (rt)",
         "DED (rescue) vs DED (deadend) - Mean IOU", "ded_rescue_ded_deadend_mean_iou.png"),
    ]

    for data, x_vals, y_vals, xlabel, ylabel, title, filename in plots:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = _plot_heatmap(ax, data, x_vals, y_vals, xlabel, ylabel, title)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
