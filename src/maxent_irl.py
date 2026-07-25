"""
Maximum Entropy IRL (Ziebart et al. 2008).

Core idea: find reward weights theta such that the expected feature counts of
trajectories, under the max-entropy distribution induced by theta, match the
empirical feature counts of the demonstrations. Gradient of the objective is:

    grad = empirical_feature_counts - expected_state_visitation_freq @ feature_map

We ascend that gradient. The expected state visitation frequency (SVF) is computed
by the standard Ziebart forward algorithm given the current soft policy.

This is the algorithm whose *generative model assumes a Boltzmann-rational
demonstrator* — the key fact behind the confound we discussed. Note that here.
"""

import numpy as np


def compute_expected_svf(env, theta, horizon, start_dist):
    """Expected state visitation frequencies under the soft-optimal policy for theta."""
    rewards = env.feature_map @ theta
    n_states, n_actions = env.n_states, 4

    # --- Backward pass: soft value iteration -> stochastic policy ---
    V = np.full(n_states, -np.inf)
    # Soft VI (finite horizon, matches Ziebart's backward pass)
    V = np.zeros(n_states)
    for _ in range(horizon):
        Q = rewards[:, None] + env.gamma * V[env.transitions]
        V = _softmax_rows(Q)
    Q = rewards[:, None] + env.gamma * V[env.transitions]
    policy = np.exp(Q - V[:, None])          # soft policy pi(a|s)
    policy /= policy.sum(axis=1, keepdims=True)

    # --- Forward pass: propagate visitation ---
    svf = np.zeros(n_states)
    mu = start_dist.copy()                    # visitation at t=0
    svf += mu
    for _ in range(horizon - 1):
        mu_next = np.zeros(n_states)
        for s in range(n_states):
            if mu[s] < 1e-12:
                continue
            for a in range(n_actions):
                mu_next[env.transitions[s, a]] += mu[s] * policy[s, a]
        mu = mu_next
        svf += mu
    # svf currently sums to `horizon` (one full probability mass added per
    # timestep). Normalise to per-step-average visitation frequency so it's
    # on the same scale as demo_feature_counts (empirical_feature_counts
    # divides by total state-visits, i.e. also a per-step average) -- the
    # gradient demo_fc - exp_fc is only meaningful if both sides use the
    # same convention.
    return svf / horizon


def _softmax_rows(Q):
    """Row-wise log-sum-exp (soft value)."""
    m = Q.max(axis=1, keepdims=True)
    return (m.squeeze(1) + np.log(np.exp(Q - m).sum(axis=1)))


def maxent_irl(env, demo_feature_counts, start_dist, horizon,
               lr=0.1, n_iters=200, verbose=False):
    """
    demo_feature_counts: empirical mean feature vector of demonstrations (n_features,)
    Returns recovered theta.
    """
    theta = np.zeros(env.n_features)
    for it in range(n_iters):
        exp_svf = compute_expected_svf(env, theta, horizon, start_dist)
        exp_feature_counts = exp_svf @ env.feature_map
        grad = demo_feature_counts - exp_feature_counts
        theta += lr * grad
        if verbose and it % 50 == 0:
            print(f"  [maxent] iter {it} |grad|={np.linalg.norm(grad):.4f}")
    return theta
