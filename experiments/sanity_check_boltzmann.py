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
from validate_unbiased import make_simple_gridworld, empirical_feature_counts


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

    return betas, regrets


if __name__ == "__main__":
    run_sanity_check()
