import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import shape

from lifegate_for_SVPs import LifeGate
from SVPs_algos import value_iter, value_iter_near_greedy_prob, V2Q, value_iter_near_greedy
import matplotlib.patches as patches


def MDP_lifegate(env, types='regular', deadend_threshold=0.7):
    P = {}
    width, height = env.scr_w, env.scr_h
    for y in range(height):
        for x in range(width):

            s = y * width + x
            P[s] = {}

            # Barriers got no list
            if  [x, y] in env.barriers:
                continue

            for a in env.legal_actions:
                P[s][a] = []
                new_x, new_y = x, y
                reward = 0.0
                done = False

                # Dead_ends state can always move to right without getting out of bound.
                if [x, y] in env.dead_ends:
                    if [x + 1, y] in env.deaths:
                        done = True
                        if types == 'death' or types == 'regular':
                            reward = -1.0
                    P[s][a].append((deadend_threshold, s + 1, reward, done))
                    P[s][a].append((0.3, s, 0, False))
                    continue

                # If the current state is termination, quick set up and continue.
                if [new_x, new_y] in env.deaths:
                    if types == 'death' or types == 'regular':
                        reward = -1.0
                    P[s][a].append((1.0, s, reward, True))
                    continue
                if [new_x, new_y] in env.recoveries:
                    if types == 'recovery' or types == 'regular':
                        reward = 1.0
                    P[s][a].append((1.0, s, reward, True))
                    continue

                # Move the player.
                if a == 1:
                    new_y = y - 1
                elif a == 2:
                    new_y = y + 1
                elif a == 3:
                    new_x = x - 1
                elif a == 4:
                    new_x = x + 1

                # If the player get out of bound, move it back to the previous state.
                if new_x < 0 or new_y < 0 or new_x >= width or new_y >= height or [new_x, new_y] in env.barriers:
                    new_x, new_y = x, y

                # Set up for regular states, regular states also have a natural risk to be dragged to the right.
                reward_drag = 0.0
                done_drag = False
                s_next = new_y * width + new_x
                s_drag = s + 1

                if [new_x, new_y] in env.deaths:
                    done = True
                    if types == 'death' or types == 'regular':
                        reward = -1.0
                elif [new_x, new_y] in env.recoveries:
                    done = True
                    if types == 'recovery' or types == 'regular':
                        reward = 1.0

                if [x + 1, y] in env.deaths:
                    done_drag = True
                    if types == 'death' or types == 'regular':
                        reward_drag = -1.0

                P[s][a].append((1 - env.death_drag, s_next, reward, done))
                P[s][a].append((env.death_drag, s_drag, reward_drag, done_drag))
    return  P


def visualize_v_grid(grid, title="V-Grid Heatmap", cmap="RdBu", vmin=None, vmax=None):
    plt.imshow(grid, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax)
    plt.colorbar(label="Value")
    plt.title(title)
    plt.axis("off")
    plt.show()


def visualize_svp(π, barrier_states, lifegate_states, dead_end, deads_states, title="Set-Valued Policy"):
    grid_size = 10
    fig, ax = plt.subplots(figsize=(6, 6))

    # Offsets for each action: how far the arrow will go from the center (dx, dy)
    arrow_offsets = {
        0: (0, 0),
        1: (0, -0.4),  # up
        2: (0, 0.4),   # down
        3: (-0.4, 0),  # left
        4: (0.4, 0),   # right
    }

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.invert_yaxis()  # put y=0 at top

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, grid_size, 1))
    ax.set_yticks(np.arange(-0.5, grid_size, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='major', color='black', linewidth=1)

    # Arrow style
    arrow_style = dict(arrowstyle="->", color="green", lw=2)

    # Loop over each cell (y,x)
    for y in range(grid_size):
        for x in range(grid_size):
            s = y * grid_size + x  # Flattened index
            if s in barrier_states: continue
            for a in range(5):
                if π[s, a] == 1:
                    dx, dy = arrow_offsets[a]
                    # If action=0 is "no-op," you could skip drawing or draw a dot:
                    if a == 0:
                        # For example, draw a small dot for "no move"
                        ax.plot(x, y, 'o', markersize=4, color="green")
                    else:
                        # End point is the start plus offset
                        x_end = x + dx
                        y_end = y + dy
                        # Draw an arrow from (x, y) to (x_end, y_end)
                        ax.annotate(
                            "",
                            xy=(x_end, y_end),
                            xytext=(x, y),
                            arrowprops=arrow_style
                        )
    for s in barrier_states:
        y = s // grid_size
        x = s % grid_size
        rect = patches.Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            facecolor="gray", edgecolor="none"
        )
        ax.add_patch(rect)

    for s in lifegate_states:
        y = s // grid_size
        x = s % grid_size
        rect = patches.Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            facecolor="blue", edgecolor="none"
        )
        ax.add_patch(rect)

    for s in dead_end:
        y = s // grid_size
        x = s % grid_size
        rect = patches.Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            facecolor="yellow", edgecolor="none"
        )
        ax.add_patch(rect)

    for s in deads_states:
        y = s // grid_size
        x = s % grid_size
        rect = patches.Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            facecolor="red", edgecolor="none"
        )
        ax.add_patch(rect)

    plt.title(title)
    plt.tight_layout()
    plt.show()


def visualize_ded(Q_d, barrier_states, lifegate_states, dead_end, deads_states, title="Dead-End Policy"):
    grid_size = 10
    fig, ax = plt.subplots(figsize=(6, 6))

    # Offsets for each action: how far the arrow will go from the center (dx, dy)
    arrow_offsets = {
        0: (0, 0),
        1: (0, -0.4),  # up
        2: (0, 0.4),   # down
        3: (-0.4, 0),  # left
        4: (0.4, 0),   # right
    }

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.invert_yaxis()  # put y=0 at top

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, grid_size, 1))
    ax.set_yticks(np.arange(-0.5, grid_size, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='major', color='black', linewidth=1)

    # Arrow style
    arrow_style = dict(arrowstyle="->", color="red", lw=2)

    # Loop over each cell (y,x)
    for y in range(grid_size):
        for x in range(grid_size):
            s = y * grid_size + x  # Flattened index
            for a in range(5):
                if Q_d[s, a] <= -0.7:
                    dx, dy = arrow_offsets[a]
                    # If action=0 is "no-op," you could skip drawing or draw a dot:
                    if a == 0:
                        # For example, draw a small dot for "no move"
                        ax.plot(x, y, 'ko', markersize=4)
                    else:
                        # End point is the start plus offset
                        x_end = x + dx
                        y_end = y + dy
                        # Draw an arrow from (x, y) to (x_end, y_end)
                        ax.annotate(
                            "",
                            xy=(x_end, y_end),
                            xytext=(x, y),
                            arrowprops=arrow_style
                        )
    for s in barrier_states:
        y = s // grid_size
        x = s % grid_size
        rect = patches.Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            facecolor="gray", edgecolor="none"
        )
        ax.add_patch(rect)

    for s in lifegate_states:
        y = s // grid_size
        x = s % grid_size
        rect = patches.Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            facecolor="blue", edgecolor="none"
        )
        ax.add_patch(rect)

    for s in dead_end:
        y = s // grid_size
        x = s % grid_size
        rect = patches.Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            facecolor="yellow", edgecolor="none"
        )
        ax.add_patch(rect)

    for s in deads_states:
        y = s // grid_size
        x = s % grid_size
        rect = patches.Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            facecolor="red", edgecolor="none"
        )
        ax.add_patch(rect)

    plt.title(title)
    plt.tight_layout()
    plt.show()

def bad_policies(Q_d, env_death, env, threshold):
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

def policy_conflict(π_svp, π_bad):
    # A conflict exists if for any state s and action a, both π_svp and π_bad recommend that action.
    for s in range(len(π_bad)):
        for a in range(len(π_bad[s])):
            # Conflict only if both policies indicate a recommendation (i.e. both are 1)
            if π_svp[s][a] == 1 and π_bad[s][a] == 1:
                print([s, a])
                return True
    return False

def search_parameters(env, env_death, gamma, zeta_values, threshold_values, theta=1e-10, max_iter=1000):
    valid_pairs = []
    lifegate_states = [5, 6, 7]
    deads_states = [8, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    dead_ends = [45, 46, 47, 48, 55, 56, 57, 58, 65, 66, 67, 68, 75, 76, 77, 78, 85, 86, 87, 88, 95, 96, 97, 98]
    # Loop over candidate deadend thresholds and ζ values.
    for dt in threshold_values:
        for zeta in zeta_values:
            # Update the regular MDP with the candidate deadend_threshold.
            env.P = MDP_lifegate(env, types='regular', deadend_threshold=dt)
            env.nS = env.scr_w * env.scr_h
            env.nA = env.nb_actions
            # Also update the death MDP.
            env_death.P = MDP_lifegate(env, types='death')
            env_death.nS = env_death.scr_w * env_death.scr_h
            env_death.nA = env_death.nb_actions

            # Compute the optimal value function using standard value iteration.
            # (Assuming value_iter returns (V, pi) – we only need V_star.)
            V_star, _ = value_iter(env, gamma, theta=theta)
            # Now run the near-greedy value iteration to get the set-valued policy (SVP).
            V, π_svp = value_iter_near_greedy(env, gamma, zeta, V_star, theta=theta, max_iter=max_iter)
            π_svp = cleaned(π_svp, lifegate_states, deads_states, dead_ends)

            # For the death MDP, compute V_d and then get Q_d.
            V_d, π_d = value_iter(env=env_death, gamma=1, theta=theta)
            Q_d = V2Q(env_death, V_d, gamma)
            # Use the candidate deadend threshold dt (not the entire threshold list) for bad policies.
            π_bad = bad_policies(Q_d, env_death, env, dt)

            # Check for conflict: i.e., an action is recommended by both SVP and the bad policy.
            conflict = policy_conflict(π_svp, π_bad)
            if not conflict:
                valid_pairs.append([zeta, dt])
                print(f"Valid pair found: ζ = {zeta:.3f}, deadend_threshold = {dt:.3f}")
            else:
                print(f"Pair ζ = {zeta:.3f}, deadend_threshold = {dt:.3f} FAILS")
    return valid_pairs


def cleaned(π_SVP, lifegate_states, deads_states, dead_ends):
    for s in range(len(π_SVP)):
        if s in lifegate_states or s in deads_states or s in dead_ends:
            for a in range(len(π_SVP[s])):
                π_SVP[s, a] = 0
    return π_SVP

def main():
    # random seed
    random_state = np.random.RandomState(1234)

    barrier_states = [0, 1, 2, 3, 4, 51, 52, 53, 54]
    lifegate_states = [5, 6, 7]
    dead_ends = [45, 46, 47, 48, 55, 56, 57, 58, 65, 66, 67, 68, 75, 76, 77, 78, 85, 86, 87, 88, 95, 96, 97, 98]
    deads_states = [8, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

    # Initial setup for set-valued-policies environment.
    env = LifeGate(state_mode='tabular', rng=random_state, death_drag=0.4, fixed_life=True)
    env.P = MDP_lifegate(env)
    env.nS = env.scr_w * env.scr_h  # number of states (10*10)
    env.nA = env.nb_actions  # number of actions (should be 5)
    # 1) Compute the optimal value function and policies using value iteration.
    V_star, π_star = value_iter(env=env, gamma=1)
    # 2) Compute the worst value function and near-optimal policies using near greedy
    V_SVP, π_SVP = value_iter_near_greedy(env=env, gamma=1,
                                               V_star=V_star, zeta=0.01, theta=1e-10, max_iter=1000)
    π_SVP = cleaned(π_SVP, lifegate_states, deads_states, dead_ends)

    # Initial setup for lifegate environment.
    env_lifegate = LifeGate(state_mode='tabular', rng=random_state, death_drag=0.4, fixed_life=True)
    env_lifegate.P = MDP_lifegate(env, types='recovery')
    env_lifegate.nS = env_lifegate.scr_w * env_lifegate.scr_h
    env_lifegate.nA = env_lifegate.nb_actions
    # 3) Compute the V_r and r-policies.
    V_r, π_r = value_iter(env=env_lifegate, gamma=1)

    # Initial setup for death environment.
    env_death = LifeGate(state_mode='tabular', rng=random_state, death_drag=0.4, fixed_life=True)
    env_death.P = MDP_lifegate(env, types='death')
    env_death.nS = env_death.scr_w * env_death.scr_h
    env_death.nA = env_death.nb_actions
    # 4) Compute the V_d and d-policies.
    V_d, π_d = value_iter(env=env_death, gamma=1)


    # Visualize the three V-functions.
    V_SVP_Grid = np.zeros((env.scr_w, env.scr_h))
    V_r_Grid = np.zeros((env.scr_w, env.scr_h))
    V_d_Grid = np.zeros((env.scr_w, env.scr_h))
    for y in range(env.scr_h):
        for x in range(env.scr_w):
            V_SVP_Grid[y, x] = V_SVP[y * env.scr_w + x]
            V_r_Grid[y, x] = V_r[y * env.scr_w + x]
            V_d_Grid[y, x] = V_d[y * env.scr_w + x]

    visualize_v_grid(V_SVP_Grid, title="V_SVP Grid", vmin=V_SVP.min(), vmax=V_SVP.max())
    visualize_v_grid(V_r_Grid, title="V_R Grid", vmin=V_r.min(), vmax=V_r.max())
    visualize_v_grid(V_d_Grid, title="V_D Grid", vmin=V_d.min(), vmax=V_d.max())

    Q_d = V2Q(env_death, V_d, 1)
    visualize_ded(Q_d, barrier_states, lifegate_states, dead_ends, deads_states, 'Dead-End Policy')

    visualize_svp(π_SVP, barrier_states, lifegate_states, dead_ends, deads_states, 'Set-Valued Policy')

    # Define candidate parameter ranges
    zeta_values = np.linspace(0.0, 0.5, 6)  # e.g., 0, 0.1, 0.2, 0.3, 0.4, 0.5
    deadend_threshold_values = np.linspace(0.5, 0.9, 5)  # e.g., 0.5, 0.6, 0.7, 0.8, 0.9

    # Create an instance of your LifeGate environment.
    random_state = np.random.RandomState(1234)
    env = LifeGate(state_mode='tabular', rng=random_state, death_drag=0.4, fixed_life=True)
    env_death = LifeGate(state_mode='tabular', rng=random_state, death_drag=0.4, fixed_life=True)

    # Run the parameter search.
    valid_pairs = search_parameters(env, env_death, 1, zeta_values, deadend_threshold_values)

if __name__ == '__main__':
    main()
