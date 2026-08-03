"""
THE discriminating diagnostic: regret vs. sample size (demos per start
state), at a few fixed Boltzmann severities, for Max-Ent recovery.

Endpoints already measured: population (N=infinity) gives ~0 regret at
every beta tested; N=1 demo/start gives ~1.0 regret at beta=0.1/0.3/0.5.
This fills in the shape between those endpoints, at a few beta values, to
tell apart:
  (a) bias severity manifests as a SAMPLE-COMPLEXITY effect -- higher bias
      needs much more N to reach the same recovery quality, but the
      SHAPE/RATE differs meaningfully across beta (steeper N-requirement
      at low beta than high beta)
  (b) bias barely matters once you have "enough" data, and that "enough"
      is roughly the SAME modest N regardless of beta -- i.e. the curves
      for different beta look similar in shape/rate, just start from
      different N=1 points.

Averages over multiple seeds per (beta, N) cell since small-N regret is
itself a noisy random variable.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from boltzmann import boltzmann_policy, generate_trajectories as boltzmann_rollout
from metrics import value_regret, stochastic_value_regret
from maxent_irl import maxent_irl
from validate_unbiased import make_simple_gridworld, empirical_feature_counts

env, goal = make_simple_gridworld(size=5, gamma=0.9)
start_dist = np.ones(env.n_states) / env.n_states
horizon = 15
_, Q_true, pi_true = env.value_iteration()

betas = [0.1, 0.5, 1.0, 2.0]
Ns = [1, 2, 5, 10, 20, 50, 100, 200]
n_seeds = 8

results = {}
for beta in betas:
    pp = boltzmann_policy(Q_true, beta)
    severity = stochastic_value_regret(env, pp, horizon, start_dist)
    row = []
    for N in Ns:
        regrets = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 * seed + int(beta * 10))
            start_states = list(range(env.n_states)) * N
            demos = boltzmann_rollout(env, Q_true, beta, start_states, horizon, rng)
            demo_fc = empirical_feature_counts(env, demos)
            theta = maxent_irl(env, demo_fc, start_dist, horizon, lr=0.5, n_iters=300)
            _, _, pi_hat = env.value_iteration(rewards=env.feature_map @ theta)
            regrets.append(value_regret(env, pi_hat, horizon, start_dist))
        mean_r, std_r = np.mean(regrets), np.std(regrets)
        row.append((N, mean_r, std_r))
        print(f"  beta={beta:<4} severity={severity:.3f}  N={N:<4} mean_regret={mean_r:.4f}  std={std_r:.4f}")
    results[beta] = {"severity": severity, "curve": row}
    print()

import json
out_path = os.path.join(os.path.dirname(__file__), "regret_vs_samplesize_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"Saved to {out_path}")
