"""
Hard-margin (max-margin) Apprenticeship Learning, implemented as a linear
SEPARABILITY test rather than a raw unconstrained QP.

Why a separability LP instead of the textbook "minimize ||w||^2 s.t.
w.(mu_E - mu_j) >= 1" QP directly: that QP is scale-sensitive (the "1" on
the RHS is an arbitrary unit against an unconstrained ||w||), so an
infeasible instance can make a poorly-behaved solver's ||w|| run away
towards whatever numerical ceiling it hits, and a BUGGY solver would
produce exactly the same runaway-norm symptom on a feasible-but-badly-
scaled instance -- i.e. "w diverges" is not by itself distinguishable from
"my solver is broken" without more care. This was flagged explicitly
before writing any solver code (risk of manufacturing a fake divergence
under time pressure).

Instead: fix ||w||_inf <= 1 (a bounded, scale-free normalisation) and ask
the crisp question "what is the best achievable margin m at this
normalisation" via a linear program:

    maximize m
    subject to  (mu_E - mu_j) . w >= m   for every mixture policy j
                -1 <= w_i <= 1
                m unconstrained in sign (m <= 0 IS a valid LP outcome --
                it just means the best you can do is not separate at all)

This is exactly the Abbeel & Ng max-margin objective's FEASIBILITY
question, decided by a solver (HiGHS via scipy.optimize.linprog) with a
crisp success/failure result instead of a hand-rolled unconstrained
optimizer. m* > 0 (positive achievable margin) means expert_fe IS
separable from the current mixture -- a genuine solution exists. m* <= 0
means no direction separates it -- genuine infeasibility, not a numerical
artifact, because linprog's optimality certificate is exact for the given
bounded LP.

The outer loop mirrors src/apprenticeship.py's projection method
structure: maintain a growing set of best-response policies' feature
expectations, re-solve the separability LP each round using the best w
found so far to generate the next best-response policy.
"""

import sys, os
import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from apprenticeship import discounted_feature_expectations
from metrics import _worst_case_policy


def separating_margin(expert_fe, mixture_fes, w_bound=1.0):
    """
    LP: maximize m s.t. (expert_fe - mu_j).w >= m for all j, -w_bound <= w <= w_bound.
    Returns (m_star, w_star). m_star > 0 => separable (feasible hard-margin
    solution exists at this normalisation). m_star <= 0 => not separable
    (genuine infeasibility -- exact LP result, not a convergence artifact).
    """
    mixture_fes = np.asarray(mixture_fes)
    n_features = expert_fe.shape[0]
    diffs = expert_fe[None, :] - mixture_fes           # (n_mix, n_features)
    n_mix = diffs.shape[0]

    # variables: [w (n_features), m (1)]. maximize m == minimize -m.
    c = np.zeros(n_features + 1)
    c[-1] = -1.0
    # constraint: diffs @ w - m >= 0  =>  -diffs @ w + m <= 0
    A_ub = np.hstack([-diffs, np.ones((n_mix, 1))])
    b_ub = np.zeros(n_mix)
    bounds = [(-w_bound, w_bound)] * n_features + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        # LP itself failed to solve (should not happen for a bounded, always-
        # feasible-at-m<=0 problem) -- surface this distinctly from m*<=0.
        return None, None
    w_star = res.x[:n_features]
    m_star = res.x[-1]
    return m_star, w_star


def hard_margin_al_traced(env, expert_fe, start_dist, horizon, n_iters=50, w_bound=1.0):
    """
    Outer loop: maintain growing mixture of best-response policies' feature
    expectations, track the separability margin m* at each round.

    Mixture seed: the WORST-case policy (metrics._worst_case_policy), not
    env.value_iteration()'s default (TRUE-optimal) output. Seeding with the
    true-optimal policy makes the true-optimal demonstrator's own test
    degenerate (distance-0 self-comparison at iteration 0, since both would
    default to the same call) -- caught by validating against the known-
    good case first, exactly the discipline this was supposed to enforce.
    """
    policy0 = _worst_case_policy(env)
    fe0 = discounted_feature_expectations(env, policy0, start_dist, horizon)
    mixture = [fe0]

    margins = []
    matched = False
    for it in range(n_iters):
        # A mixture point exactly (within tolerance) at expert_fe is SUCCESS
        # (the expert was found as an achievable best response), not a
        # self-comparison failure -- exclude self-matches from the
        # constraint set, or the trivial "distance to itself is 0" would
        # masquerade as non-separability.
        others = [mu for mu in mixture if not np.allclose(mu, expert_fe, atol=1e-8)]
        if len(others) < len(mixture):
            matched = True
            margins.append(np.inf)   # exact match: strictly better than any finite margin
            break
        m_star, w_star = separating_margin(expert_fe, others, w_bound=w_bound)
        margins.append(m_star)
        if m_star is None:
            break
        rewards = env.feature_map @ w_star
        _, _, policy = env.value_iteration(rewards=rewards)
        fe = discounted_feature_expectations(env, policy, start_dist, horizon)
        if any(np.allclose(fe, existing, atol=1e-9) for existing in mixture):
            break
        mixture.append(fe)

    return {
        "margins": margins,
        "final_margin": margins[-1] if margins else None,
        "n_mixture": len(mixture),
        "matched": matched,
        "separable": matched or (margins and margins[-1] is not None and margins[-1] > 1e-6),
    }
