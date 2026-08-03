"""
Pre-committed pass/fail battery for the new structured-feature gridworld
(src/gridworld_structured.py). Weights and layout were fixed BEFORE this
script was run; nothing here is adjusted based on the results below.

Reuses all existing generic IRL/metric machinery unchanged (apprenticeship.py,
maxent_irl.py, metrics.py, boltzmann.py) -- only the environment differs.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gridworld_structured import make_structured_gridworld
from apprenticeship import apprenticeship_learning, discounted_feature_expectations
from maxent_irl import maxent_irl
from metrics import value_regret, stochastic_value_regret
from boltzmann import boltzmann_policy, generate_trajectories as boltzmann_rollout

env, feature_names, goal_state, hazard_cells, terrain_cells = make_structured_gridworld(size=7, gamma=0.9)
start_dist = np.ones(env.n_states) / env.n_states
horizon = 25
_, Q_true, pi_true = env.value_iteration()

print(f"Structured gridworld: {env.size}x{env.size}, {env.n_states} states, "
      f"{env.n_features} features {feature_names}, theta_true={env.theta_true.tolist()}")
print(f"hazard cells: {len(hazard_cells)}   terrain cells: {len(terrain_cells)}")
print()


def undiscounted_feature_expectation(env, policy_probs, start_dist, horizon):
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


def deterministic_probs(env, policy):
    probs = np.zeros((env.n_states, 4))
    probs[np.arange(env.n_states), policy] = 1.0
    return probs


results_summary = []

# ---------------------------------------------------------------------
print("=" * 90)
print("CHECK 1: Foundation gate -- both methods recover ~0 regret from optimal demos")
print("=" * 90)
expert_fe_disc = discounted_feature_expectations(env, pi_true, start_dist, horizon)
w_al = apprenticeship_learning(env, expert_fe_disc, start_dist, horizon, n_iters=200)
_, _, pi_al = env.value_iteration(rewards=env.feature_map @ w_al)
regret_al = value_regret(env, pi_al, horizon, start_dist)

expert_fe_undisc = undiscounted_feature_expectation(env, deterministic_probs(env, pi_true), start_dist, horizon)
theta_me = maxent_irl(env, expert_fe_undisc, start_dist, horizon, lr=0.5, n_iters=300)
_, _, pi_me = env.value_iteration(rewards=env.feature_map @ theta_me)
regret_me = value_regret(env, pi_me, horizon, start_dist)

pass1 = regret_al < 0.05 and regret_me < 0.05
print(f"  AL regret={regret_al:.4f}   MaxEnt regret={regret_me:.4f}   -> {'PASS' if pass1 else 'FAIL'}")
results_summary.append(("1. Foundation gate", f"AL={regret_al:.4f}, ME={regret_me:.4f}", pass1))

# ---------------------------------------------------------------------
print()
print("=" * 90)
print("CHECK 2: Tie sanity -- fraction of states with tied-optimal actions (true reward)")
print("=" * 90)
tol = 1e-6
q_max = Q_true.max(axis=1, keepdims=True)
n_tied_actions = (np.abs(Q_true - q_max) < tol).sum(axis=1)
frac_tied_states = np.mean(n_tied_actions > 1)
pass2 = frac_tied_states < 0.20
print(f"  states with >1 tied-optimal action: {int(np.sum(n_tied_actions>1))}/{env.n_states} = {frac_tied_states:.3f}"
      f"   (old one-hot grid: 17/25 = 0.680)   -> {'PASS' if pass2 else 'FAIL'}")
results_summary.append(("2. Tie sanity", f"{frac_tied_states:.3f} tied", pass2))

# ---------------------------------------------------------------------
print()
print("=" * 90)
print("CHECK 3: Continuity, not bimodality -- per-seed regret distribution, fixed high bias")
print("=" * 90)
beta_fixed = 0.2
pp_fixed = boltzmann_policy(Q_true, beta_fixed)
severity_fixed = stochastic_value_regret(env, pp_fixed, horizon, start_dist)
print(f"  beta={beta_fixed}, demonstrator severity={severity_fixed:.3f}")

from validate_unbiased import empirical_feature_counts  # generic, works for any feature_map

Ns = [1, 5, 10, 20, 50, 100]
n_seeds = 16
all_cell_regrets = {}
for N in Ns:
    regrets = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(9000 * seed + 3)
        start_states = list(range(env.n_states)) * N
        demos = boltzmann_rollout(env, Q_true, beta_fixed, start_states, horizon, rng)
        demo_fc = empirical_feature_counts(env, demos)
        theta = maxent_irl(env, demo_fc, start_dist, horizon, lr=0.5, n_iters=300)
        _, _, pi_hat = env.value_iteration(rewards=env.feature_map @ theta)
        regrets.append(round(value_regret(env, pi_hat, horizon, start_dist), 3))
    all_cell_regrets[N] = regrets
    bins = np.zeros(10)
    for r in regrets:
        bins[min(int(r * 10), 9)] += 1
    hist_str = " ".join(f"{int(b)}" for b in bins)
    print(f"  N={N:<4} regrets={regrets}")
    print(f"        histogram [0.0-1.0 in 10 bins]: {hist_str}")

# PASS if, at SOME N, there's a populated middle (values in (0.2,0.8) present in >=2 seeds)
middle_populated = any(sum(1 for r in regrets if 0.2 < r < 0.8) >= 2 for regrets in all_cell_regrets.values())
pass3 = middle_populated
print(f"  -> {'PASS' if pass3 else 'FAIL'} (populated middle found: {middle_populated})")
results_summary.append(("3. Continuity not bimodality", "see histograms above", pass3))

# ---------------------------------------------------------------------
print()
print("=" * 90)
print("CHECK 4: Non-trivial population recovery -- analytic FE, Max-Ent across beta sweep")
print("=" * 90)
betas = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
pop_regrets = []
for beta in betas:
    pp = boltzmann_policy(Q_true, beta)
    severity = stochastic_value_regret(env, pp, horizon, start_dist)
    demo_fc = undiscounted_feature_expectation(env, pp, start_dist, horizon)
    theta = maxent_irl(env, demo_fc, start_dist, horizon, lr=0.5, n_iters=300)
    _, _, pi_hat = env.value_iteration(rewards=env.feature_map @ theta)
    regret = value_regret(env, pi_hat, horizon, start_dist)
    pop_regrets.append(regret)
    print(f"  beta={beta:<5} severity={severity:.3f}  population_maxent_regret={regret:.4f}")

pass4 = any(r > 0.02 for r in pop_regrets)
print(f"  -> {'PASS' if pass4 else 'FAIL'} (max population regret across sweep: {max(pop_regrets):.4f})")
results_summary.append(("4. Non-trivial population recovery", f"max_regret={max(pop_regrets):.4f}", pass4))

# ---------------------------------------------------------------------
print()
print("=" * 90)
print("CHECK 5: Extraction stability -- gap between recovered-policy FE and fitted data")
print("=" * 90)
gaps = []
for N in [1, 5]:
    for seed in range(16):
        rng = np.random.default_rng(9000 * seed + 3)
        start_states = list(range(env.n_states)) * N
        demos = boltzmann_rollout(env, Q_true, beta_fixed, start_states, horizon, rng)
        demo_fc = empirical_feature_counts(env, demos)
        theta = maxent_irl(env, demo_fc, start_dist, horizon, lr=0.5, n_iters=300)
        _, _, pi_hat = env.value_iteration(rewards=env.feature_map @ theta)
        regret = value_regret(env, pi_hat, horizon, start_dist)
        recovered_fe = undiscounted_feature_expectation(env, deterministic_probs(env, pi_hat), start_dist, horizon)
        gap = np.linalg.norm(demo_fc - recovered_fe)
        gaps.append((N, seed, regret, gap))

max_gap = max(g[3] for g in gaps)
median_gap = float(np.median([g[3] for g in gaps]))
high_regret_gaps = [g[3] for g in gaps if g[2] > 0.5]
print(f"  n_draws={len(gaps)}  median_gap={median_gap:.4f}  max_gap={max_gap:.4f}")
if high_regret_gaps:
    print(f"  gaps specifically on high-regret (>0.5) draws: {[round(g,4) for g in high_regret_gaps]}")
else:
    print("  (no draws had regret > 0.5 in this sample)")
pass5 = max_gap < 0.3
print(f"  -> {'PASS' if pass5 else 'FAIL'} (old one-hot grid showed 0.68-0.71 gaps)")
results_summary.append(("5. Extraction stability", f"max_gap={max_gap:.4f}, median={median_gap:.4f}", pass5))

# ---------------------------------------------------------------------
print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"{'Check':<40} {'Metric':<35} {'Result'}")
for name, metric, passed in results_summary:
    print(f"{name:<40} {metric:<35} {'PASS' if passed else 'FAIL'}")
