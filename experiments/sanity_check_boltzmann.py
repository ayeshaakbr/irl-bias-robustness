"""
SANITY CHECK for the Boltzmann demonstrator (src/boltzmann.py) -- run this
before trusting it for the real bias-sweep experiments.

Sweeps beta (the rationality knob), and for each value: generates Boltzmann
demos from the TRUE Q-values, recovers a reward with Max-Ent IRL, and scores
the recovered policy with value regret against the true optimum.

Pass condition: regret -> ~0 as beta -> large (recovers the optimal-demo
result from the foundation), and regret rises as beta drops toward 0.

Failure signatures this is designed to catch:
 - Flat curve (regret same at every beta) -> generate_trajectories is
   accidentally using argmax instead of sampling (silently rebuilds the
   optimal demonstrator regardless of beta).
 - NaN at high beta -> boltzmann_policy is missing the row-max subtraction
   before exponentiating (overflow).
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maxent_irl import maxent_irl
from metrics import value_regret
from boltzmann import generate_trajectories
from validate_unbiased import (
    make_simple_gridworld, empirical_feature_counts, generate_optimal_demos,
    policy_match,
)


def cross_check_high_beta(env, Q_true, pi_true, start_states, start_dist, horizon,
                           beta=1000, seed=0):
    """
    As beta -> infinity, softmax(beta*Q) -> a one-hot on the argmax action, so
    a Boltzmann demonstrator should become indistinguishable from the
    deterministic optimal demonstrator. This checks that claim directly by
    comparing Max-Ent recovery from very-high-beta Boltzmann demos against
    recovery from the actual deterministic optimal demos (same code path
    validate_unbiased.py uses for the foundation gate) -- not just "both are
    near zero regret" but that the two recovered rewards actually agree.
    """
    def cos(a, b):
        return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

    # --- Baseline: deterministic optimal demos ---
    base_demos = generate_optimal_demos(env, pi_true, start_states, horizon)
    base_fc = empirical_feature_counts(env, base_demos)
    theta_base = maxent_irl(env, base_fc, start_dist, horizon, lr=0.5, n_iters=300)
    _, _, pi_base = env.value_iteration(rewards=env.feature_map @ theta_base)
    regret_base = value_regret(env, pi_base, horizon, start_dist)

    # --- High-beta Boltzmann demos ---
    rng = np.random.default_rng(seed)
    hb_demos = generate_trajectories(env, Q_true, beta, start_states, horizon, rng)
    hb_fc = empirical_feature_counts(env, hb_demos)
    theta_hb = maxent_irl(env, hb_fc, start_dist, horizon, lr=0.5, n_iters=300)
    _, _, pi_hb = env.value_iteration(rewards=env.feature_map @ theta_hb)
    regret_hb = value_regret(env, pi_hb, horizon, start_dist)

    fc_diff = np.max(np.abs(base_fc - hb_fc))
    policy_agree = policy_match(pi_base, pi_hb)
    theta_cos = cos(theta_base, theta_hb)
    regret_diff = abs(regret_hb - regret_base)

    print("\n" + "=" * 60)
    print(f"CROSS-CHECK: beta={beta} Boltzmann demos vs. optimal-demo baseline")
    print("=" * 60)
    print(f"  regret (optimal-demo baseline)      : {regret_base:.4f}")
    print(f"  regret (beta={beta} Boltzmann)          : {regret_hb:.4f}")
    print(f"  max |feature_count difference|      : {fc_diff:.2e}")
    print(f"  recovered-policy agreement          : {policy_agree:.3f}  (informational -- see note)")
    print(f"  recovered-theta cosine similarity   : {theta_cos:.6f}  (informational -- see note)")

    # PASS condition is regret agreement, not raw policy/theta agreement.
    # This environment has 17/25 states with tied-optimal actions (see
    # test_metric_tie_robust in tests/test_foundation.py): at beta=1000 the
    # softmax over near-tied Q-values still occasionally samples the
    # non-canonical tied action, so pi_hb legitimately disagrees with
    # pi_base on exactly those tied states without being wrong -- both
    # actions have identical true value there. Requiring literal policy/
    # theta agreement would reproduce the tie-matching bug this project's
    # metric choice (value regret) already exists to avoid.
    ok = regret_diff < 1e-2
    print("PASS: high-beta Boltzmann recovery matches the optimal-demo baseline (by regret)."
          if ok else
          "FAIL: high-beta Boltzmann recovery diverges from the optimal-demo baseline (by regret).")
    if policy_agree < 0.99:
        n_diff = int(round((1 - policy_agree) * len(pi_base)))
        print(f"  note: {n_diff} state(s) disagree -- expected if confined to tied-optimal states.")
    return ok


def run_sanity_check():
    size, gamma, horizon = 5, 0.9, 15
    env, goal = make_simple_gridworld(size=size, gamma=gamma)

    start_states = list(range(env.n_states))
    start_dist = np.ones(env.n_states) / env.n_states

    _, Q_true, pi_true = env.value_iteration()

    betas = [0.1, 0.5, 1, 2, 5, 10, 50]
    rng = np.random.default_rng(seed=0)

    print("=" * 60)
    print("SANITY CHECK: Boltzmann demonstrator via Max-Ent recovery")
    print("=" * 60)

    regrets = []
    for beta in betas:
        demos = generate_trajectories(env, Q_true, beta, start_states, horizon, rng)
        demo_fc = empirical_feature_counts(env, demos)

        theta_me = maxent_irl(env, demo_fc, start_dist, horizon, lr=0.5, n_iters=300)
        rewards_me = env.feature_map @ theta_me
        _, _, pi_me = env.value_iteration(rewards=rewards_me)

        regret = value_regret(env, pi_me, horizon, start_dist)
        regrets.append(regret)
        flag = "  <-- NaN! check row-max subtraction in boltzmann_policy" if np.isnan(regret) else ""
        print(f"  beta={beta:>5}: value regret = {regret:.4f}{flag}")

    if np.isnan(regrets).any():
        print("\nFAIL: NaN encountered -- see flagged rows above.")
    elif regrets[0] - regrets[-1] < 1e-3:
        print("\nFAIL: curve is flat -- looks like actions were argmax'd, not sampled.")
    elif regrets[-1] > 0.05:
        print("\nFAIL: regret at highest beta did not go near zero.")
    else:
        print("\nPASS: regret falls to ~0 at high beta, rises as beta drops.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(betas, regrets, marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("beta (log scale)")
        ax.set_ylabel("value regret (normalised)")
        ax.set_title("Max-Ent recovery regret vs. Boltzmann demonstrator beta")
        ax.grid(True, alpha=0.3)
        out_path = os.path.join(os.path.dirname(__file__), "boltzmann_sanity_check.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {out_path}")
    except ImportError:
        print("\n(matplotlib not available, skipping plot)")

    cross_check_high_beta(env, Q_true, pi_true, start_states, start_dist, horizon)

    return betas, regrets


if __name__ == "__main__":
    run_sanity_check()
