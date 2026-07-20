"""
Evaluation metric: VALUE REGRET (a.k.a. expected value difference / policy regret).

Given a policy pi_hat induced by a recovered reward, its value regret is how much
TRUE-reward value it loses relative to the true-optimal policy, measured under the
true reward and true dynamics:

    regret(pi_hat) = E_startdist[ V*_true(s) - V^{pi_hat}_true(s) ]

Key properties (why we chose this over action-matching):
 - TIE-ROBUST: a policy that picks a different-but-equally-optimal action has
   regret 0, not a spurious penalty (this was the bug the validation slice exposed).
 - SHAPING-ROBUST: rewards differing by potential-based shaping induce the same
   optimal policy, so they get the same (zero) regret. Raw reward-distance does not
   have this property.
 - CONTINUOUS: unlike binary optimality, it measures HOW MUCH recovery degrades
   under bias -- essential for characterising the degradation curve.

Normalisation: we report NORMALISED regret in [0, 1]:
    regret_norm = regret / (V*_true - V_worst_true)
so results are comparable across environments with different reward scales.
regret_norm = 0 means fully optimal; 1 means as bad as the worst policy.
"""

import numpy as np


def policy_value_under_true(env, policy, horizon, start_dist):
    """Expected discounted TRUE-reward value of following `policy` for `horizon` steps."""
    mu = start_dist.copy()
    value = 0.0
    discount = 1.0
    for _ in range(horizon):
        value += discount * (mu @ env.true_rewards)
        mu_next = np.zeros(env.n_states)
        for s in range(env.n_states):
            if mu[s] < 1e-12:
                continue
            mu_next[env.transitions[s, policy[s]]] += mu[s]
        mu = mu_next
        discount *= env.gamma
    return value


def value_regret(env, recovered_policy, horizon, start_dist, normalise=True):
    """
    Normalised value regret of a recovered policy. 0 = optimal, higher = worse.
    """
    _, _, pi_star = env.value_iteration()          # true-optimal policy
    v_star = policy_value_under_true(env, pi_star, horizon, start_dist)
    v_hat = policy_value_under_true(env, recovered_policy, horizon, start_dist)

    raw_regret = v_star - v_hat
    if not normalise:
        return raw_regret

    # Worst-case value for normalisation: policy that minimises true value.
    # Compute via value iteration on the NEGATED true reward.
    worst_env_rewards = -env.true_rewards
    V = np.zeros(env.n_states)
    for _ in range(2000):
        Q = worst_env_rewards[:, None] + env.gamma * V[env.transitions]
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < 1e-8:
            V = V_new; break
        V = V_new
    Q = worst_env_rewards[:, None] + env.gamma * V[env.transitions]
    pi_worst = Q.argmax(axis=1)
    v_worst = policy_value_under_true(env, pi_worst, horizon, start_dist)

    denom = v_star - v_worst
    if abs(denom) < 1e-9:
        return 0.0                                 # degenerate: all policies equal
    return raw_regret / denom
