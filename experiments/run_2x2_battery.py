"""
Clean 2x2 battery: {projection, hard-margin(separability)} x {Boltzmann, myopia}.

Analytic (population-level) feature expectations throughout -- isolates the
geometric mechanism from sampling noise. Local diagnostic only, not part of
the committed experiment runner.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from boltzmann import boltzmann_policy
from myopia import finite_horizon_policy_probs
from metrics import value_regret, stochastic_value_regret
from apprenticeship import discounted_feature_expectations
from validate_unbiased import make_simple_gridworld
from diagnose_al_wellposedness import stochastic_feature_expectations, apprenticeship_learning_traced
from hard_margin_al import hard_margin_al_traced

size, gamma, horizon = 5, 0.9, 15
env, goal = make_simple_gridworld(size=size, gamma=gamma)
start_dist = np.ones(env.n_states) / env.n_states
_, Q_true, pi_true = env.value_iteration()

PROJ_ITERS = 8000
QP_ITERS = 60

results = []

def run_cell(label, expert_fe, severity):
    proj = apprenticeship_learning_traced(env, expert_fe, start_dist, horizon,
                                           n_iters=PROJ_ITERS, tol=1e-9)
    proj_floor = proj["margins"][-1] / proj["margins"][-2000] if len(proj["margins"]) > 2000 and proj["margins"][-2000] > 0 else None
    proj_status = "converges" if proj["final_margin"] < 1e-3 else (
        "plateau" if proj_floor is not None and proj_floor > 0.5 else "slow/uncertain")

    qp = hard_margin_al_traced(env, expert_fe, start_dist, horizon, n_iters=QP_ITERS)

    row = {
        "label": label, "severity": severity,
        "proj_final_margin": proj["final_margin"], "proj_status": proj_status,
        "qp_separable": qp["separable"], "qp_matched": qp["matched"],
        "qp_final_margin": qp["final_margin"], "qp_n_mixture": qp["n_mixture"],
    }
    results.append(row)
    qp_m = row["qp_final_margin"]
    qp_m_str = f'{qp_m:.4f}' if (qp_m is not None and np.isfinite(qp_m)) else str(qp_m)
    print(f"{label:22s} sev={severity:.3f}  proj_final={proj['final_margin']:.6f} ({proj_status:12s})  "
          f"qp_separable={qp['separable']!s:5s}  qp_final_margin={qp_m_str}")

print("=" * 100)
print("BOLTZMANN sweep")
print("=" * 100)
for beta in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
    pp = boltzmann_policy(Q_true, beta)
    severity = stochastic_value_regret(env, pp, horizon, start_dist)
    efe = stochastic_feature_expectations(env, pp, start_dist, horizon)
    run_cell(f"boltzmann beta={beta}", efe, severity)

print()
print("=" * 100)
print("MYOPIA (fixed, tie-aware) sweep")
print("=" * 100)
for H in [0, 1, 2, 3, 4, 5, 6]:
    probs = finite_horizon_policy_probs(env, H)
    severity = stochastic_value_regret(env, probs, horizon, start_dist)
    efe = stochastic_feature_expectations(env, probs, start_dist, horizon)
    run_cell(f"myopia H={H}", efe, severity)

print()
print("=" * 100)
print("TRUE-OPTIMAL (severity 0, sanity anchor)")
print("=" * 100)
efe_opt = discounted_feature_expectations(env, pi_true, start_dist, horizon)
run_cell("true-optimal", efe_opt, 0.0)

# save results locally
import json
out_path = os.path.join(os.path.dirname(__file__), "battery_2x2_results.json")
with open(out_path, "w") as f:
    json.dump([{k: (v if not isinstance(v, (np.floating,)) or np.isfinite(v) else str(v)) for k, v in r.items()} for r in results], f, indent=2, default=str)
print(f"\nSaved to {out_path}")
