"""
Apprenticeship Learning via IRL (Abbeel & Ng 2004), projection method.

Core idea: iteratively find a reward (weight vector) under which the expert
looks maximally better than the current mixture of learner policies, then add
the best-response policy to the mixture. The weight vector w points from the
current mixture's feature expectations toward the expert's.

Unlike Max-Ent, this makes *no probabilistic assumption* about how the
demonstrator generated its actions -- it only cares about matching expected
discounted feature counts. That asymmetry is exactly what the experiment probes.

We return the final weight vector w as the "recovered reward direction".
Note: AL recovers a reward *direction* (unit vector), not a calibrated scale --
this matters for how we design the comparison metric later.
"""

import numpy as np


def discounted_feature_expectations(env, policy, start_dist, horizon):
    """Expected discounted feature counts for a deterministic policy."""
    mu = start_dist.copy()
    fe = np.zeros(env.n_features)
    discount = 1.0
    for _ in range(horizon):
        fe += discount * (mu @ env.feature_map)
        mu_next = np.zeros(env.n_states)
        for s in range(env.n_states):
            if mu[s] < 1e-12:
                continue
            mu_next[env.transitions[s, policy[s]]] += mu[s]
        mu = mu_next
        discount *= env.gamma
    return fe


def apprenticeship_learning(env, expert_fe, start_dist, horizon,
                            n_iters=50, tol=1e-6, verbose=False):
    """
    expert_fe: expert's expected discounted feature counts (n_features,)
    Returns recovered weight vector w (reward direction).
    """
    # Initialise with an arbitrary policy's feature expectations
    _, _, policy0 = env.value_iteration()
    fe0 = discounted_feature_expectations(env, policy0, start_dist, horizon)

    fe_bar = fe0.copy()          # current mixture feature expectation
    mixture = [fe0]

    w = expert_fe - fe_bar
    for it in range(n_iters):
        # Projection step: w points from mixture toward expert
        w = expert_fe - fe_bar
        t = np.linalg.norm(w)
        if t < tol:
            break
        w_unit = w / t
        # Best response: optimal policy under reward = w_unit . features
        rewards = env.feature_map @ w_unit
        _, _, policy = env.value_iteration(rewards=rewards)
        fe = discounted_feature_expectations(env, policy, start_dist, horizon)

        # Projection update of the mixture feature expectation (Abbeel&Ng projection)
        a = fe - fe_bar
        b = expert_fe - fe_bar
        step = (a @ b) / (a @ a + 1e-12)
        fe_bar = fe_bar + step * a
        mixture.append(fe)
        if verbose and it % 10 == 0:
            print(f"  [apprentice] iter {it} margin t={t:.4f}")
    return w / (np.linalg.norm(w) + 1e-12)
