"""
SANITY CHECK for the myopia demonstrator (src/myopia.py) -- run before
trusting it, same spirit as sanity_check_boltzmann.py.

Sweeps planning_horizon H (bounded lookahead depth), and for each value:
rolls out the finite-horizon policy, recovers a reward with Max-Ent IRL, and
scores the recovered policy with value regret. NOTE: gamma-based myopia
(gamma_demo < gamma_true) was tried first and rejected -- in this
environment reward(s) doesn't depend on action, so discounting never
reorders which direction looks best, only H (whether the goal is even
visible within the plan) does. See src/myopia.py docstring for the full
diagnosis. Confirmed directly before trusting it: sweeping H alone (without
Max-Ent) already shows a smooth, monotonic degradation from 25/25 states
disagreeing with true-optimal at H=0 down to 0/25 at H=8 (the grid's
diameter) -- a real curve, not the gamma-based version's cliff.

Pass condition: regret -> ~0 as H grows past the grid's diameter (myopia
bias vanishes, demos become the deterministic-optimal baseline), and regret
rises smoothly as H shrinks (the demonstrator's blind spot grows).

This script also builds the SEVERITY-MATCHED comparison with the Boltzmann
sweep: instead of plotting recovered regret against the raw bias parameter
(beta or H -- not comparable to each other), it plots against the
DEMONSTRATOR's own value regret (src/metrics.py: value_regret /
stochastic_value_regret), which is on the same normalised 0-1 scale
regardless of which bias model produced it. That's what makes "myopia
severity 0.4" and "Boltzmann severity 0.4" mean the same thing, and is what
will let the eventual headline experiment ask "at matched overall
demonstrator degradation, does any Max-Ent/Apprenticeship asymmetry
persist?" instead of comparing apples (beta) to oranges (H).
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maxent_irl import maxent_irl
from metrics import value_regret, stochastic_value_regret
from myopia import finite_horizon_policy, generate_trajectories as myopia_rollout
from boltzmann import boltzmann_policy, generate_trajectories as boltzmann_rollout
from validate_unbiased import make_simple_gridworld, empirical_feature_counts


def check_horizon_smoothness(env, horizon, planning_horizons):
    """
    Guard against myopia's own version of the Boltzmann trap: verify the
    H-sweep is a genuine curve (state-disagreement count strictly shrinking,
    covering intermediate values) before spending time on the expensive
    Max-Ent recovery loop. A step function here (e.g. all-or-nothing at one
    H) would mean this grid is too small for ANY myopia mechanism to give
    smooth degradation, which is a bigger conversation than a code fix.
    """
    _, _, pi_true = env.value_iteration()
    sd = np.ones(env.n_states) / env.n_states
    diffs = []
    for H in planning_horizons:
        pi_h = finite_horizon_policy(env, H)
        n_diff = int(np.sum(pi_h != pi_true))
        diffs.append(n_diff)
    print("  H-sweep state-disagreement counts:", diffs)
    is_monotonic = all(diffs[i] >= diffs[i + 1] for i in range(len(diffs) - 1))
    n_intermediate = sum(0 < d < env.n_states for d in diffs)
    smooth = is_monotonic and n_intermediate >= 3
    print("  PASS: smooth, monotonic degradation across H." if smooth else
          "  FAIL: myopia-via-horizon is also a step function on this grid.")
    return smooth


def run_myopia_sweep(env, start_states, start_dist, horizon, planning_horizons):
    regrets, severities = [], []
    for H in planning_horizons:
        pi_demo = finite_horizon_policy(env, H)

        # Demonstrator's own severity: regret of the finite-horizon policy
        # ITSELF under the true reward (deterministic -> plain value_regret).
        severity = value_regret(env, pi_demo, horizon, start_dist)

        demos = myopia_rollout(env, pi_demo, start_states, horizon)
        demo_fc = empirical_feature_counts(env, demos)
        theta = maxent_irl(env, demo_fc, start_dist, horizon, lr=0.5, n_iters=300)
        _, _, pi_hat = env.value_iteration(rewards=env.feature_map @ theta)
        regret = value_regret(env, pi_hat, horizon, start_dist)

        regrets.append(regret)
        severities.append(severity)
        flag = "  <-- NaN!" if np.isnan(regret) else ""
        print(f"  H={H:>2}: demonstrator severity={severity:.4f}"
              f"  recovered regret={regret:.4f}{flag}")
    return severities, regrets


def run_boltzmann_sweep(env, Q_true, start_states, start_dist, horizon, betas, seed=0):
    rng = np.random.default_rng(seed)
    regrets, severities = [], []
    for beta in betas:
        policy_probs = boltzmann_policy(Q_true, beta)
        severity = stochastic_value_regret(env, policy_probs, horizon, start_dist)

        demos = boltzmann_rollout(env, Q_true, beta, start_states, horizon, rng)
        demo_fc = empirical_feature_counts(env, demos)
        theta = maxent_irl(env, demo_fc, start_dist, horizon, lr=0.5, n_iters=300)
        _, _, pi_hat = env.value_iteration(rewards=env.feature_map @ theta)
        regret = value_regret(env, pi_hat, horizon, start_dist)

        regrets.append(regret)
        severities.append(severity)
    return severities, regrets


def run_sanity_check():
    size, gamma_true, horizon = 5, 0.9, 15
    env, goal = make_simple_gridworld(size=size, gamma=gamma_true)
    start_states = list(range(env.n_states))
    start_dist = np.ones(env.n_states) / env.n_states
    _, Q_true, _ = env.value_iteration()

    planning_horizons = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15]

    print("=" * 60)
    print("SANITY CHECK: myopia demonstrator via Max-Ent recovery")
    print("=" * 60)
    print("Step 1 -- confirm the H-sweep itself is a curve, not a cliff:")
    smooth = check_horizon_smoothness(env, horizon, planning_horizons)
    if not smooth:
        print("\nAborting recovery sweep: fix the horizon mechanism or grid size first.")
        return None

    print("\nStep 2 -- Max-Ent recovery across the H-sweep:")
    mi_severities, mi_regrets = run_myopia_sweep(env, start_states, start_dist, horizon, planning_horizons)

    if np.isnan(mi_regrets).any():
        print("\nFAIL: NaN encountered.")
    elif mi_regrets[0] - mi_regrets[-1] < 1e-3:
        print("\nFAIL: curve is flat -- finite_horizon_policy likely not varying with H.")
    elif mi_regrets[-1] > 0.05:
        print("\nFAIL: regret at large H did not go near zero.")
    else:
        print("\nPASS: regret falls to ~0 as H grows, rises as H shrinks.")

    # --- Severity-matched comparison against the Boltzmann sweep ---
    betas = [0.1, 0.5, 1, 2, 5, 10, 50]
    bz_severities, bz_regrets = run_boltzmann_sweep(env, Q_true, start_states, start_dist, horizon, betas)

    print("\n" + "=" * 60)
    print("SEVERITY-MATCHED COMPARISON (x-axis = demonstrator's own regret)")
    print("=" * 60)
    print("  Myopia:")
    for s, r in zip(mi_severities, mi_regrets):
        print(f"    severity={s:.4f}  recovered_regret={r:.4f}")
    print("  Boltzmann:")
    for s, r in zip(bz_severities, bz_regrets):
        print(f"    severity={s:.4f}  recovered_regret={r:.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        ax = axes[0]
        ax.plot(planning_horizons, mi_regrets, marker="o")
        ax.set_xlabel("planning horizon H")
        ax.set_ylabel("value regret (normalised)")
        ax.set_title("Max-Ent recovery regret vs. myopia planning horizon")
        ax.axvline(8, color="gray", linestyle="--", alpha=0.5, label="grid diameter")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        mi_order = np.argsort(mi_severities)
        bz_order = np.argsort(bz_severities)
        ax.plot(np.array(mi_severities)[mi_order], np.array(mi_regrets)[mi_order],
                marker="o", label="Myopia")
        ax.plot(np.array(bz_severities)[bz_order], np.array(bz_regrets)[bz_order],
                marker="s", label="Boltzmann")
        ax.set_xlabel("demonstrator's own value regret (severity, shared scale)")
        ax.set_ylabel("Max-Ent recovered-policy regret")
        ax.set_title("Severity-matched comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), "myopia_sanity_check.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {out_path}")
    except ImportError:
        print("\n(matplotlib not available, skipping plot)")

    return mi_severities, mi_regrets, bz_severities, bz_regrets


if __name__ == "__main__":
    run_sanity_check()
