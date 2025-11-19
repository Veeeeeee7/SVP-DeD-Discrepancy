import numpy as np
import pickle
from lifegate_for_SVPs import LifeGate
from SVPs_algos import value_iter, value_iter_near_greedy, V2Q

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


def train_and_save_results(filename="results/trained_results.pkl"):
    # Define special states
    barrier_states = [0, 1, 2, 3, 4, 51, 52, 53, 54]
    lifegate_states = [5, 6, 7]
    dead_ends = [45, 46, 47, 48, 55, 56, 57, 58, 65, 66, 67, 68, 75, 76, 77, 78, 85, 86, 87, 88, 95, 96, 97, 98]
    deads_states = [8, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

    random_state = np.random.RandomState(1234)
    # Create the environment for training (regular MDP)
    env = LifeGate(state_mode='tabular', rng=random_state, death_drag=0.4, fixed_life=True)
    env.P = MDP_lifegate(env, types='regular')
    env.nS = env.scr_w * env.scr_h
    env.nA = env.nb_actions

    # *** Compute the near-greedy SVP using optimal value function V_star ***
    V_star, π_star = value_iter(env, gamma=1, theta=1e-10)
    V_SVP, π_SVP, is_max_iter, iter = value_iter_near_greedy(env, gamma=1, zeta=0, V_star=V_star, theta=1e-10, max_iter=1000)
    π_SVP = cleaned(π_SVP, lifegate_states, deads_states, dead_ends)
    if is_max_iter: print("SVP does not converge")
    else: print("SVP converge")
    print(iter)

    # *** Compute optimal value function and policy for rescue state discovery ***
    env_r = LifeGate(state_mode='tabular', rng=random_state, death_drag=0.4, fixed_life=True)
    env_r.P = MDP_lifegate(env, types='recovery')
    env_r.nS = env_r.scr_w * env_r.scr_h
    env_r.nA = env_r.nb_actions
    V_r, π_r = value_iter(env_r, gamma=1, theta=1e-10)


    # *** Compute optimal value function and policy for deadend state discovery ***
    env_d = LifeGate(state_mode='tabular', rng=random_state, death_drag=0.4, fixed_life=True)
    env_d.P = MDP_lifegate(env, types='death')
    env_d.nS = env_d.scr_w * env_d.scr_h
    env_d.nA = env_d.nb_actions
    V_d, π_d = value_iter(env_d, gamma=1, theta=1e-10)
    Q_d = V2Q(env_d, V_d, 1)

    # Pack results into a dictionary.
    results = {
        "V_star": V_star,
        "π_star": π_star,
        "V_SVP": V_SVP,
        "π_SVP": π_SVP,
        "V_r": V_r,
        "π_r": π_r,
        "V_d": V_d,
        "π_d": π_d,
        "Q_d": Q_d,
    }

    # Save results to a file.
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print("Training complete. Results saved to", filename)



if __name__ == "__main__":
    train_and_save_results()
