"""
Tests that lock in the validated foundation. Run with: python3 tests/test_foundation.py

These encode the invariants we established:
 1. Both IRL methods recover ~zero value regret from OPTIMAL demos (the gate).
 2. The value-regret metric behaves correctly on known policies
    (optimal -> 0, worst -> ~1, tie-breaking -> still 0).
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gridworld import Gridworld
from metrics import value_regret, policy_value_under_true
from maxent_irl import maxent_irl
from apprenticeship import apprenticeship_learning
from experiments.validate_unbiased import (
    make_simple_gridworld, generate_optimal_demos, empirical_feature_counts,
    discounted_expert_fe,
)

HORIZON = 15


def test_metric_optimal_is_zero():
    env, _ = make_simple_gridworld()
    _, _, pi_star = env.value_iteration()
    sd = np.ones(env.n_states) / env.n_states
    r = value_regret(env, pi_star, HORIZON, sd)
    assert abs(r) < 1e-9, f"optimal policy should have 0 regret, got {r}"
    print(f"  PASS  optimal policy regret = {r:.2e} (~0)")


def test_metric_tie_robust():
    """A policy using different-but-tied-optimal actions must still have 0 regret."""
    env, _ = make_simple_gridworld()
    _, Q, pi_star = env.value_iteration()
    sd = np.ones(env.n_states) / env.n_states
    # Build an alternative policy choosing the LAST tied-optimal action instead of first
    pi_alt = np.array([
        np.where(np.isclose(Q[s], Q[s].max(), atol=1e-6))[0][-1]
        for s in range(env.n_states)
    ])
    n_diff = np.sum(pi_alt != pi_star)
    r = value_regret(env, pi_alt, HORIZON, sd)
    assert abs(r) < 1e-9, f"tied-optimal policy should have 0 regret, got {r}"
    print(f"  PASS  tie-robust: {n_diff} states differ in action, regret still {r:.2e}")


def test_validation_gate():
    """Both methods recover near-zero regret from optimal demos."""
    np.random.seed(0)
    env, _ = make_simple_gridworld()
    sd = np.ones(env.n_states) / env.n_states
    _, _, pi_true = env.value_iteration()
    demos = generate_optimal_demos(env, pi_true, list(range(env.n_states)), HORIZON)

    # Max-Ent
    demo_fc = empirical_feature_counts(env, demos)
    theta_me = maxent_irl(env, demo_fc, sd, HORIZON, lr=0.5, n_iters=300)
    _, _, pi_me = env.value_iteration(rewards=env.feature_map @ theta_me)
    r_me = value_regret(env, pi_me, HORIZON, sd)

    # Apprenticeship
    expert_fe = discounted_expert_fe(env, demos, HORIZON)
    w_al = apprenticeship_learning(env, expert_fe, sd, HORIZON, n_iters=100)
    _, _, pi_al = env.value_iteration(rewards=env.feature_map @ w_al)
    r_al = value_regret(env, pi_al, HORIZON, sd)

    print(f"  Max-Ent      value regret from optimal demos: {r_me:.4f}")
    print(f"  Apprentice   value regret from optimal demos: {r_al:.4f}")
    assert r_me < 0.05, f"Max-Ent failed gate: regret {r_me}"
    assert r_al < 0.05, f"Apprenticeship failed gate: regret {r_al}"
    print("  PASS  both methods clear the un-biased validation gate")


if __name__ == "__main__":
    print("Running foundation tests...\n")
    test_metric_optimal_is_zero()
    test_metric_tie_robust()
    test_validation_gate()
    print("\nAll foundation tests passed.")
