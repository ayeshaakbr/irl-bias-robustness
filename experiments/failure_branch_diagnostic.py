"""
Artifact check for the bimodal Max-Ent failure branch (beta=0.1, finite N).

Hypothesis to test: failure is explained by a GENERAL, environment-agnostic
informational quantity -- the empirical demo feature count's distance from
the TRUE (analytic, population) demonstrator expectation -- because that
distance is what determines whether the noisy sample "looks like" a
different, wrong-but-internally-consistent policy that Max-Ent then
faithfully (and correctly, given what it was handed) fits to.

Two checks, both must point the same way for this to be a general
informational story rather than a grid-specific degeneracy:
  1. Does ||demo_fc - analytic_fe|| (deviation from truth) separate
     success (regret~0) from failure (regret~1) cases?
  2. On FAILURE draws specifically: is demo_fc actually closer to the
     RECOVERED policy's own feature expectation than to the analytic
     Boltzmann expectation? If so, Max-Ent isn't malfunctioning -- the data
     it received really does look more like a different policy, and it
     fit exactly what was in front of it. That's a general property of
     fitting-under-noise, not a 5x5-gridworld-specific artifact.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from boltzmann import boltzmann_policy, generate_trajectories as boltzmann_rollout
from metrics import value_regret
from maxent_irl import maxent_irl
from apprenticeship import discounted_feature_expectations
from validate_unbiased import make_simple_gridworld, empirical_feature_counts
from maxent_battery import undiscounted_feature_expectation

env, goal = make_simple_gridworld(size=5, gamma=0.9)
start_dist = np.ones(env.n_states) / env.n_states
horizon = 15
_, Q_true, pi_true = env.value_iteration()

beta = 0.1
pp = boltzmann_policy(Q_true, beta)
analytic_fe = undiscounted_feature_expectation(env, pp, start_dist, horizon)

N = 200
n_seeds = 40
rows = []
for seed in range(n_seeds):
    rng = np.random.default_rng(5000 * seed + 7)
    start_states = list(range(env.n_states)) * N
    demos = boltzmann_rollout(env, Q_true, beta, start_states, horizon, rng)
    demo_fc = empirical_feature_counts(env, demos)

    theta = maxent_irl(env, demo_fc, start_dist, horizon, lr=0.5, n_iters=300)
    _, _, pi_hat = env.value_iteration(rewards=env.feature_map @ theta)
    regret = value_regret(env, pi_hat, horizon, start_dist)

    dev_from_truth = np.linalg.norm(demo_fc - analytic_fe)

    # feature expectation of the RECOVERED policy, same (undiscounted,
    # per-step-average) convention as demo_fc, for the "closer to recovered
    # policy than to truth" check
    recovered_policy_probs = np.zeros((env.n_states, 4))
    recovered_policy_probs[np.arange(env.n_states), pi_hat] = 1.0
    recovered_fe = undiscounted_feature_expectation(env, recovered_policy_probs, start_dist, horizon)
    dist_to_recovered = np.linalg.norm(demo_fc - recovered_fe)
    dist_to_truth = dev_from_truth

    rows.append({
        "seed": seed, "regret": round(regret, 3),
        "dev_from_analytic_truth": round(dev_from_truth, 4),
        "dist_to_recovered_policy_fe": round(dist_to_recovered, 4),
        "dist_to_true_demonstrator_fe": round(dist_to_truth, 4),
        "closer_to_recovered_than_truth": dist_to_recovered < dist_to_truth,
    })

success = [r for r in rows if r["regret"] < 0.3]
failure = [r for r in rows if r["regret"] > 0.7]
ambiguous = [r for r in rows if 0.3 <= r["regret"] <= 0.7]

print(f"n_seeds={n_seeds}: success={len(success)}  failure={len(failure)}  ambiguous={len(ambiguous)}")
print()
print("CHECK 1: does deviation-from-truth separate success from failure?")
print(f"  mean ||demo_fc - analytic_fe||  success cases: {np.mean([r['dev_from_analytic_truth'] for r in success]):.4f}")
print(f"  mean ||demo_fc - analytic_fe||  failure cases: {np.mean([r['dev_from_analytic_truth'] for r in failure]):.4f}")
if success and failure:
    s_max = max(r['dev_from_analytic_truth'] for r in success)
    f_min = min(r['dev_from_analytic_truth'] for r in failure)
    print(f"  max deviation among successes: {s_max:.4f}   min deviation among failures: {f_min:.4f}")
    print(f"  clean separation (no overlap)?  {s_max < f_min}")

print()
print("CHECK 2: on FAILURE draws, is demo_fc closer to the recovered (wrong) policy's own FE than to the true demonstrator's FE?")
for r in failure:
    print(f"  seed={r['seed']:3d} regret={r['regret']:.3f}  dist_to_recovered={r['dist_to_recovered_policy_fe']:.4f}"
          f"  dist_to_true_demonstrator={r['dist_to_true_demonstrator_fe']:.4f}  closer_to_recovered={r['closer_to_recovered_than_truth']}")

frac_closer = np.mean([r["closer_to_recovered_than_truth"] for r in failure]) if failure else None
print(f"\n  fraction of failures where demo_fc is closer to recovered-policy-FE than to true-demonstrator-FE: {frac_closer}")

import json
with open(os.path.join(os.path.dirname(__file__), "failure_branch_diagnostic_results.json"), "w") as f:
    json.dump(rows, f, indent=2, default=str)
