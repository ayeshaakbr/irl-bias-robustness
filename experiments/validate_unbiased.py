"""
VALIDATION SLICE — the non-negotiable first milestone.

Both IRL methods must recover the true reward from OPTIMAL (unbiased)
demonstrations before we trust anything they say under bias.

We check recovery two ways, because the two methods return different objects:
 - Max-Ent returns calibrated theta -> compare induced reward / policy.
 - Apprenticeship returns a reward DIRECTION -> compare via induced policy
   and via cosine alignment with true theta direction.

The policy-match metric (does the recovered-reward-optimal policy match the
true-optimal policy) is the common currency, and is what we'll build the real
comparison metric on -- it sidesteps reward-shaping ambiguity.
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gridworld import Gridworld
from maxent_irl import maxent_irl
from apprenticeship import apprenticeship_learning, discounted_feature_expectations


def make_simple_gridworld(size=5, gamma=0.9, seed=0):
    """
    One-hot state features (n_features = n_states). True reward: goal cell in the
    corner has high weight, small negative step cost elsewhere.
    One-hot features let us test exact recovery cleanly in the validation slice.
    """
    n_states = size * size
    feature_map = np.eye(n_states)          # one-hot: feature(s) = e_s
    theta_true = np.full(n_states, -0.1)    # small step cost everywhere
    goal = n_states - 1                     # bottom-right corner
    theta_true[goal] = 1.0
    return Gridworld(size, feature_map, theta_true, gamma=gamma), goal


def generate_optimal_demos(env, policy, start_states, horizon):
    """Roll out the (deterministic) optimal policy to produce trajectories."""
    trajs = []
    for s0 in start_states:
        s = s0
        traj = [s]
        for _ in range(horizon - 1):
            s = env.transitions[s, policy[s]]
            traj.append(s)
        trajs.append(traj)
    return trajs


def empirical_feature_counts(env, trajs):
    """Mean (undiscounted) feature vector across all visited states."""
    total = np.zeros(env.n_features)
    n = 0
    for traj in trajs:
        for s in traj:
            total += env.feature_map[s]
            n += 1
    return total / n


def discounted_expert_fe(env, trajs, horizon):
    """Discounted feature expectations from demonstrations (for apprenticeship)."""
    fe = np.zeros(env.n_features)
    for traj in trajs:
        discount = 1.0
        for s in traj:
            fe += discount * env.feature_map[s]
            discount *= env.gamma
    return fe / len(trajs)


def policy_match(p1, p2):
    """Fraction of states where two policies agree."""
    return np.mean(p1 == p2)


def run_validation():
    np.random.seed(0)
    size, gamma, horizon = 5, 0.9, 15
    env, goal = make_simple_gridworld(size=size, gamma=gamma)

    # Ground-truth optimal policy
    V_true, Q_true, pi_true = env.value_iteration()

    # Demonstrations from every cell (full coverage for the clean validation)
    start_states = list(range(env.n_states))
    start_dist = np.ones(env.n_states) / env.n_states
    demos = generate_optimal_demos(env, pi_true, start_states, horizon)

    print("=" * 60)
    print("VALIDATION: recover true reward from OPTIMAL demos")
    print("=" * 60)

    # --- Max-Ent IRL ---
    demo_fc = empirical_feature_counts(env, demos)
    theta_me = maxent_irl(env, demo_fc, start_dist, horizon, lr=0.5, n_iters=300)
    rewards_me = env.feature_map @ theta_me
    _, _, pi_me = env.value_iteration(rewards=rewards_me)
    match_me = policy_match(pi_true, pi_me)

    # --- Apprenticeship ---
    expert_fe = discounted_expert_fe(env, demos, horizon)
    w_al = apprenticeship_learning(env, expert_fe, start_dist, horizon, n_iters=100)
    rewards_al = env.feature_map @ w_al
    _, _, pi_al = env.value_iteration(rewards=rewards_al)
    match_al = policy_match(pi_true, pi_al)

    # Cosine alignment of recovered reward direction with truth
    def cos(a, b):
        return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

    print(f"\nMax-Ent   policy match vs true optimal : {match_me:.3f}")
    print(f"Max-Ent   reward-direction cosine      : {cos(theta_me, env.theta_true):.3f}")
    print(f"Apprentice policy match vs true optimal: {match_al:.3f}")
    print(f"Apprentice reward-direction cosine     : {cos(w_al, env.theta_true):.3f}")

    print("\nInterpretation:")
    print("  policy match near 1.0 => method recovers reward that induces")
    print("  (near-)optimal behaviour. This is the gate. Cosine is secondary")
    print("  (reward-shaping ambiguity makes exact theta recovery not expected).")
    return match_me, match_al


if __name__ == "__main__":
    run_validation()
