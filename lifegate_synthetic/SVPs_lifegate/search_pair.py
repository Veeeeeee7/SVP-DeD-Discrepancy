import numpy as np
import matplotlib.pyplot as plt
from lifegate_for_SVPs import LifeGate
from SVPs_algos import value_iter, value_iter_near_greedy, V2Q

def MDP_lifegate(env, types='regular', deadend_threshold=0.7):
    P = {}
    width, height = env.scr_w, env.scr_h
    for y in range(height):
        for x in range(width):
            s = y * width + x
            P[s] = {}
            if [x, y] in env.barriers:
                continue
            for a in env.legal_actions:
                P[s][a] = []
                new_x, new_y = x, y
                reward = 0.0
                done = False
                if [x, y] in env.dead_ends:
                    if [x + 1, y] in env.deaths:
                        done = True
                        if types in ['death', 'regular']:
                            reward = -1.0
                    P[s][a].append((deadend_threshold, s + 1, reward, done))
                    P[s][a].append((1 - deadend_threshold, s, 0, False))
                    continue
                if [new_x, new_y] in env.deaths:
                    if types in ['death', 'regular']:
                        reward = -1.0
                    P[s][a].append((1.0, s, reward, True))
                    continue
                if [new_x, new_y] in env.recoveries:
                    if types in ['recovery', 'regular']:
                        reward = 1.0
                    P[s][a].append((1.0, s, reward, True))
                    continue
                if a == 1:
                    new_y = y - 1
                elif a == 2:
                    new_y = y + 1
                elif a == 3:
                    new_x = x - 1
                elif a == 4:
                    new_x = x + 1
                if new_x < 0 or new_y < 0 or new_x >= width or new_y >= height or [new_x, new_y] in env.barriers:
                    new_x, new_y = x, y
                reward_drag = 0.0
                done_drag = False
                s_next = new_y * width + new_x
                s_drag = s + 1
                if [new_x, new_y] in env.deaths:
                    done = True
                    if types in ['death', 'regular']:
                        reward = -1.0
                elif [new_x, new_y] in env.recoveries:
                    done = True
                    if types in ['recovery', 'regular']:
                        reward = 1.0
                if [x + 1, y] in env.deaths:
                    done_drag = True
                    if types in ['death', 'regular']:
                        reward_drag = -1.0
                P[s][a].append((1 - env.death_drag, s_next, reward, done))
                P[s][a].append((env.death_drag, s_drag, reward_drag, done_drag))
    return P

# def MDP_lifegate(env, types='regular', deadend_threshold=0.7):
#     P = {}
#     width, height = env.scr_w, env.scr_h
#     for y in range(height):
#         for x in range(width):
#             s = y * width + x
#             P[s] = {}
#             if [x, y] in env.barriers:
#                 continue
#             for a in env.legal_actions:
#                 P[s][a] = []
#                 new_x, new_y = x, y
#                 reward, done = 0.0, False
#
#                 # dead-end special drag
#                 if [x, y] in env.dead_ends:
#                     if [x+1, y] in env.deaths:
#                         done = True
#                         if types in ('death','regular'):
#                             reward = -1.0
#                     P[s][a].append((deadend_threshold, s+1, reward, done))
#                     P[s][a].append((0.3, s, 0.0, False))
#                     continue
#
#                 # terminal death
#                 if [new_x, new_y] in env.deaths:
#                     if types in ('death','regular'):
#                         reward = -1.0
#                     P[s][a].append((1.0, s, reward, True))
#                     continue
#                 # terminal recovery
#                 if [new_x, new_y] in env.recoveries:
#                     if types in ('recovery','regular'):
#                         reward = +1.0
#                     P[s][a].append((1.0, s, reward, True))
#                     continue
#
#                 # move
#                 if a==1: new_y -= 1
#                 elif a==2: new_y += 1
#                 elif a==3: new_x -= 1
#                 elif a==4: new_x += 1
#
#                 # bounce off
#                 if (new_x<0 or new_y<0 or new_x>=width or new_y>=height
#                   or [new_x,new_y] in env.barriers):
#                     new_x, new_y = x, y
#
#                 s_next = new_y*width + new_x
#                 s_drag = s+1
#                 reward_drag, done_drag = 0.0, False
#
#                 if [new_x,new_y] in env.deaths:
#                     done = True
#                     if types in ('death','regular'):
#                         reward = -1.0
#                 elif [new_x,new_y] in env.recoveries:
#                     done = True
#                     if types in ('recovery','regular'):
#                         reward = +1.0
#
#                 if [x+1, y] in env.deaths:
#                     done_drag = True
#                     if types in ('death','regular'):
#                         reward_drag = -1.0
#
#                 P[s][a].append((1-env.death_drag, s_next, reward, done))
#                 P[s][a].append((env.death_drag,   s_drag, reward_drag, done_drag))
#     return P

def bad_policies(Q_d, env_death, env, threshold):
    env_death.P = MDP_lifegate(env, types='death', deadend_threshold=threshold)
    env_death.nS = env_death.scr_w*env_death.scr_h
    env_death.nA = env_death.nb_actions
    pi_bad = np.zeros((env_death.nS, env_death.nA), dtype=int)
    for s in range(env_death.nS):
        for a in range(env_death.nA):
            if Q_d[s,a] <= -threshold:
                pi_bad[s,a] = 1
    return pi_bad

def fraction_conflict(pi_svp, pi_bad, valid_states):
    conflicts = 0
    for s in range(pi_bad.shape[0]):
        if any(pi_svp[s,a] and pi_bad[s,a] for a in range(pi_bad.shape[1])):
            conflicts += 1
    return conflicts / valid_states

def cleaned(pi, lifegate_states, dead_states, dead_ends):
    for s in range(pi.shape[0]):
        if s in lifegate_states or s in dead_states or s in dead_ends:
            pi[s,:] = 0
    return pi

def visualize_inconsistency_heatmap(zeta_vals, dt_vals, incons_map):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(incons_map, origin='lower', cmap='Reds', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(len(zeta_vals)))
    ax.set_xticklabels([f"{z:.2f}" for z in zeta_vals], rotation=45)
    ax.set_yticks(np.arange(len(dt_vals)))
    ax.set_yticklabels([f"{dt:.2f}" for dt in dt_vals])
    ax.set_xlabel("ζ (zeta)")
    ax.set_ylabel("deadend_threshold")
    ax.set_title("Policy Inconsistency\n(× = fully consistent)")
    plt.colorbar(im, ax=ax, label="Frac. conflicting states")
    for i in range(incons_map.shape[0]):
        for j in range(incons_map.shape[1]):
            if incons_map[i,j]==0:
                ax.text(j,i,'×',ha='center',va='center',color='green',fontsize=16)
    plt.tight_layout()

def visualize_size_heatmaps(zeta_vals, dt_vals, svp_map, bad_map):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
    for ax, data, title in zip((ax1,ax2),(svp_map,bad_map),("Avg SVP size","Avg Bad size")):
        im = ax.imshow(data, origin='lower', cmap='viridis', aspect='auto')
        ax.set_xticks(np.arange(len(zeta_vals)))
        ax.set_xticklabels([f"{z:.2f}" for z in zeta_vals],rotation=45)
        ax.set_yticks(np.arange(len(dt_vals)))
        ax.set_yticklabels([f"{dt:.2f}" for dt in dt_vals])
        ax.set_xlabel("ζ")
        ax.set_ylabel("dt")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="actions per state")
    plt.tight_layout()

def search_and_plot(env, env_death, gamma,
                    zeta_vals, dt_vals,
                    barrier_states, lifegate_states, dead_states, dead_ends,
                    theta=1e-10, max_iter=1000):
    n_dt, n_z = len(dt_vals), len(zeta_vals)
    incons_map = np.zeros((n_dt,n_z))
    svp_size_map = np.zeros((n_dt,n_z))
    bad_size_map = np.zeros((n_dt,n_z))

    total_states = env.scr_w * env.scr_h
    valid_states = total_states - len(barrier_states)

    for i, dt in enumerate(dt_vals):
      for j, zeta in enumerate(zeta_vals):
        # build MDPs
        env.P = MDP_lifegate(env, types='regular')
        env.nS,env.nA = env.scr_w*env.scr_h, env.nb_actions
        env_death.P = MDP_lifegate(env, types='death', deadend_threshold=dt)
        env_death.nS,env_death.nA = env_death.scr_w*env_death.scr_h, env_death.nb_actions

        # SVP
        V_star, _    = value_iter(env, gamma, theta=theta)
        V_svp, pi_svp, _, _ = value_iter_near_greedy(
                          env, gamma, zeta, V_star,
                          theta=theta, max_iter=max_iter)
        pi_svp = cleaned(pi_svp, lifegate_states, dead_states, dead_ends)

        # bad
        V_d, _ = value_iter(env_death, gamma, theta=theta)
        Q_d = V2Q(env_death, V_d, gamma)
        pi_bad = bad_policies(Q_d, env_death, env, dt)

        # fill maps
        incons_map[i,j] = fraction_conflict(pi_svp, pi_bad, valid_states)
        svp_size_map[i,j] = pi_svp.sum(axis=1).mean()
        bad_size_map[i,j] = pi_bad.sum(axis=1).mean()

    visualize_size_heatmaps(zeta_vals, dt_vals, svp_size_map, bad_size_map)
    plt.show()
    # visualize_inconsistency_heatmap(zeta_vals, dt_vals, incons_map)
    # plt.show()

if __name__ == "__main__":
    rs = np.random.RandomState(1234)
    env = LifeGate(state_mode='tabular', rng=rs, death_drag=0.4, fixed_life=True)
    env_death = LifeGate(state_mode='tabular', rng=rs, death_drag=0.4, fixed_life=True)

    barrier_states  = [0,1,2,3,4,51,52,53,54]
    lifegate_states = [5,6,7]
    dead_states     = [8,9,19,29,39,49,59,69,79,89,99]
    dead_ends       = [45,46,47,48,55,56,57,58,65,66,67,68,75,76,77,78,85,86,87,88,95,96,97,98]

    zeta_vals = np.linspace(0,1,10)
    dt_vals   = np.linspace(0,1,10)

    search_and_plot(env, env_death,
                    gamma=1,
                    zeta_vals=zeta_vals,
                    dt_vals=dt_vals,
                    barrier_states=barrier_states,
                    lifegate_states=lifegate_states,
                    dead_states=dead_states,
                    dead_ends=dead_ends)
