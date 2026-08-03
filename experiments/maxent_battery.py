"""
Run Max-Ent IRL recovery through the SAME corrected battery used for AL:
Boltzmann sweep + FIXED (tie-aware) myopia sweep, analytic feature
expectations, at the same severities already measured for AL. This is the
missing half of the comparison -- everything in the AL 2x2 battery was
AL-only.

Max-Ent's demo_feature_counts convention is UNDISCOUNTED per-step-average
visitation (matches compute_expected_svf: svf accumulated with no gamma
factor, normalised by /horizon at the end) -- NOT the discounted
convention used by discounted_feature_expectations / AL. Needed a
separate analytic function for this, validated against a large-sample
empirical estimate before trusting it (same discipline as everything else
this session).
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from boltzmann import boltzmann_policy, generate_trajectories as boltzmann_rollout
from myopia import finite_horizon_policy_probs, generate_trajectories_stochastic
from metrics import value_regret, stochastic_value_regret
from maxent_irl import maxent_irl
from validate_unbiased import make_simple_gridworld, empirical_feature_counts

size, gamma, horizon = 5, 0.9, 15
env, goal = make_simple_gridworld(size=size, gamma=gamma)
start_dist = np.ones(env.n_states) / env.n_states
_, Q_true, pi_true = env.value_iteration()


def undiscounted_feature_expectation(env, policy_probs, start_dist, horizon):
    """Analytic (exact) UNDISCOUNTED per-step-average feature expectation --
    the population version of what empirical_feature_counts estimates via
    sampling. Matches compute_expected_svf's convention exactly (no gamma
    factor in the forward pass, normalised by /horizon)."""
    mu = start_dist.copy()
    total = np.zeros(env.n_features)
    n_states, n_actions = policy_probs.shape
    for _ in range(horizon):
        total += mu @ env.feature_map
        mu_next = np.zeros(n_states)
        for s in range(n_states):
            if mu[s] < 1e-12:
                continue
            for a in range(n_actions):
                p = policy_probs[s, a]
                if p < 1e-12:
                    continue
                mu_next[env.transitions[s, a]] += mu[s] * p
        mu = mu_next
    return total / horizon


# --- validate the new analytic function against large-sample empirical estimate ---
print("=" * 90)
print("VALIDATION: analytic undiscounted FE vs large-sample empirical estimate")
print("=" * 90)
rng = np.random.default_rng(0)
beta_check = 0.5
pp_check = boltzmann_policy(Q_true, beta_check)
analytic_fe = undiscounted_feature_expectation(env, pp_check, start_dist, horizon)
start_states_many = list(range(env.n_states)) * 200   # 200 demos per start state
demos = boltzmann_rollout(env, Q_true, beta_check, start_states_many, horizon, rng)
empirical_fe = empirical_feature_counts(env, demos)
diff = np.linalg.norm(analytic_fe - empirical_fe)
print(f"  beta={beta_check}: ||analytic - empirical(200 demos/start)|| = {diff:.5f}  "
      f"(analytic norm={np.linalg.norm(analytic_fe):.4f})")
print("  PASS: analytic function matches large-sample empirical estimate" if diff < 0.02
      else "  FAIL: analytic function does not match sampling -- investigate before trusting it")

if diff >= 0.02:
    print("\nABORTING battery -- analytic FE function not validated.")
    sys.exit(1)

print()
print("=" * 90)
print("MAX-ENT RECOVERY -- Boltzmann sweep (matched to AL battery severities)")
print("=" * 90)
maxent_boltzmann = []
for beta in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
    pp = boltzmann_policy(Q_true, beta)
    severity = stochastic_value_regret(env, pp, horizon, start_dist)
    demo_fc = undiscounted_feature_expectation(env, pp, start_dist, horizon)
    theta = maxent_irl(env, demo_fc, start_dist, horizon, lr=0.5, n_iters=300)
    _, _, pi_hat = env.value_iteration(rewards=env.feature_map @ theta)
    regret = value_regret(env, pi_hat, horizon, start_dist)
    maxent_boltzmann.append((beta, severity, regret))
    print(f"  beta={beta:<5} severity={severity:.3f}  maxent_recovered_regret={regret:.4f}")

print()
print("=" * 90)
print("MAX-ENT RECOVERY -- myopia (FIXED, tie-aware) sweep")
print("=" * 90)
maxent_myopia = []
for H in [0, 1, 2, 3, 4, 5, 6]:
    probs = finite_horizon_policy_probs(env, H)
    severity = stochastic_value_regret(env, probs, horizon, start_dist)
    demo_fc = undiscounted_feature_expectation(env, probs, start_dist, horizon)
    theta = maxent_irl(env, demo_fc, start_dist, horizon, lr=0.5, n_iters=300)
    _, _, pi_hat = env.value_iteration(rewards=env.feature_map @ theta)
    regret = value_regret(env, pi_hat, horizon, start_dist)
    maxent_myopia.append((H, severity, regret))
    print(f"  H={H:<5} severity={severity:.3f}  maxent_recovered_regret={regret:.4f}")

import json
out_path = os.path.join(os.path.dirname(__file__), "maxent_battery_results.json")
with open(out_path, "w") as f:
    json.dump({"boltzmann": maxent_boltzmann, "myopia_fixed": maxent_myopia}, f, indent=2)
print(f"\nSaved to {out_path}")
