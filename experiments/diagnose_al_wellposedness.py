"""
DIAGNOSTIC: does Apprenticeship Learning (projection method) stay well-posed
under the myopia demonstrator, the way it fails to under Boltzmann?

This is the discriminating test for the interior-point-genericity mechanism:
a Boltzmann-rational demonstrator's feature expectation is (generically) an
INTERIOR point of the achievable-by-deterministic-policies polytope, because
it is itself a probability-weighted mixture over actions at every state. A
myopic (finite-horizon) demonstrator is fully DETERMINISTIC, so its feature
expectation should be a VERTEX (or at least much nearer one) of that same
polytope -- an extremal, achievable-by-a-single-policy point.

If the mechanism is right: AL should stay well-posed (margin -> ~0, bounded
w) under myopia even though myopia is a real, off-model bias -- because
"biased" and "interior" are not the same thing, and AL's pathology is
specifically about the latter. If AL ALSO breaks under myopia, the
interior-point mechanism is wrong or incomplete and "AL breaks under
imperfect demonstrators generally" is the (weaker, more confound-vulnerable)
fallback claim.

Uses ANALYTIC (population-level) feature expectations for both demonstrators
-- no trajectory sampling -- so this isolates the geometric mechanism from
sampling noise, which the prior Boltzmann-only diagnostic already ruled out
as the explanation for AL's failure there.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from metrics import value_regret, stochastic_value_regret
from myopia import finite_horizon_policy
from boltzmann import boltzmann_policy
from apprenticeship import discounted_feature_expectations
from validate_unbiased import make_simple_gridworld


def stochastic_feature_expectations(env, policy_probs, start_dist, horizon):
    """Analytic (exact) discounted feature expectation of a STOCHASTIC policy.
    Mirrors metrics.stochastic_policy_value_under_true but for feature counts
    instead of value -- exact expectation, no rollout sampling."""
    mu = start_dist.copy()
    fe = np.zeros(env.n_features)
    discount = 1.0
    n_states, n_actions = policy_probs.shape
    for _ in range(horizon):
        fe += discount * (mu @ env.feature_map)
        mu_next = np.zeros(env.n_states)
        for s in range(n_states):
            if mu[s] < 1e-12:
                continue
            for a in range(n_actions):
                p = policy_probs[s, a]
                if p < 1e-12:
                    continue
                mu_next[env.transitions[s, a]] += mu[s] * p
        mu = mu_next
        discount *= env.gamma
    return fe


def apprenticeship_learning_traced(env, expert_fe, start_dist, horizon,
                                    n_iters=200, tol=1e-8):
    """Same algorithm as src/apprenticeship.apprenticeship_learning, but
    returns the full margin trace and raw (pre-normalisation) w-norm trace
    instead of just the final normalised direction, so we can see HOW it
    converges (or doesn't), not just where it ends up."""
    _, _, policy0 = env.value_iteration()
    fe0 = discounted_feature_expectations(env, policy0, start_dist, horizon)

    fe_bar = fe0.copy()
    margins = []
    raw_norms = []

    for it in range(n_iters):
        w = expert_fe - fe_bar
        t = np.linalg.norm(w)
        margins.append(t)
        raw_norms.append(t)
        if t < tol:
            break
        w_unit = w / t
        rewards = env.feature_map @ w_unit
        _, _, policy = env.value_iteration(rewards=rewards)
        fe = discounted_feature_expectations(env, policy, start_dist, horizon)

        a = fe - fe_bar
        b = expert_fe - fe_bar
        denom = a @ a
        if denom < 1e-12:
            # best response is IDENTICAL to current mixture point -> stuck.
            # This is itself a diagnostic signal (mixture can't move toward
            # expert with the vertices generated so far).
            break
        step = (a @ b) / denom
        fe_bar = fe_bar + step * a

    return {
        "margins": margins,
        "final_margin": margins[-1],
        "n_iters_used": len(margins),
        "converged": margins[-1] < 1e-3,
        "margin_decreasing": len(margins) > 1 and margins[-1] < margins[0] * 0.5,
    }


def summarize(label, result):
    m = result["margins"]
    trend = "-> ~0 (well-posed)" if result["converged"] else (
        "decreasing but not converged" if result["margin_decreasing"] else
        "FLAT / STUCK (ill-posed signature)")
    print(f"  {label:38s} iters={result['n_iters_used']:4d}  "
          f"margin[0]={m[0]:.4f}  margin[-1]={result['final_margin']:.6f}   {trend}")


def run():
    size, gamma, horizon = 5, 0.9, 15
    env, goal = make_simple_gridworld(size=size, gamma=gamma)
    start_states = list(range(env.n_states))
    start_dist = np.ones(env.n_states) / env.n_states
    _, Q_true, pi_true = env.value_iteration()

    print("=" * 78)
    print("DIAGNOSTIC: AL (projection method) well-posedness, Boltzmann vs myopia")
    print("Analytic feature expectations (no sampling noise)")
    print("=" * 78)

    print("\n-- Boltzmann sweep (stochastic demonstrator, analytic FE) --")
    betas = [0.1, 0.3, 0.5, 1, 2, 5, 10, 50]
    for beta in betas:
        policy_probs = boltzmann_policy(Q_true, beta)
        severity = stochastic_value_regret(env, policy_probs, horizon, start_dist)
        expert_fe = stochastic_feature_expectations(env, policy_probs, start_dist, horizon)
        result = apprenticeship_learning_traced(env, expert_fe, start_dist, horizon)
        summarize(f"beta={beta:<6} severity={severity:.3f}", result)

    print("\n-- Myopia sweep (deterministic demonstrator, analytic FE) --")
    horizons_demo = [0, 1, 2, 3, 4, 5, 6, 8, 15]
    for H in horizons_demo:
        pi_demo = finite_horizon_policy(env, H)
        severity = value_regret(env, pi_demo, horizon, start_dist)
        expert_fe = discounted_feature_expectations(env, pi_demo, start_dist, horizon)
        result = apprenticeship_learning_traced(env, expert_fe, start_dist, horizon)
        summarize(f"H={H:<6} severity={severity:.3f}", result)

    print("\n" + "=" * 78)
    print("Interpretation: compare rows at MATCHED severity across the two")
    print("sweeps. If Boltzmann rows show 'FLAT/STUCK' at moderate-high")
    print("severity while myopia rows at similar severity show '-> ~0', that")
    print("is the discriminating contrast the mechanism predicts.")
    print("=" * 78)


if __name__ == "__main__":
    run()
