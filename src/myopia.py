"""
Myopia demonstrator (confound-control bias model).

Max-Ent IRL's generative model assumes a Boltzmann-rational demonstrator, so
an apparent Max-Ent advantage under Boltzmann bias could just be a modelling
artifact (Max-Ent "gets" the noise model, the other method doesn't) rather
than a genuine robustness difference. Myopia exists to remove that
possibility: it is OFF-MODEL for BOTH methods being compared.
 - Off-model for Max-Ent: Max-Ent assumes a stochastic softmax-over-Q
   demonstrator; a finite-horizon planner is fully deterministic and its
   suboptimality doesn't come from action noise at all.
 - Off-model for Apprenticeship: Abbeel & Ng's projection method assumes the
   demonstrator IS (near-)optimal for the true infinite-horizon objective;
   a bounded-horizon planner is a coherent optimiser, just for a truncated
   objective, so it violates that assumption too, in a different way than
   Boltzmann noise does.
Any Max-Ent-vs-Apprenticeship asymmetry that survives under myopia is much
less likely to be an artifact of either method's own generative assumption.

MECHANISM (finite-horizon lookahead, not discount-factor myopia):
Originally speced as gamma_demo < gamma_true (plan with a smaller discount).
That was tried and rejected: in this environment reward(s) is attributed to
the CURRENT state and does not vary by action, so Q(s,a) = reward(s) +
gamma*V(s') differs across actions ONLY through the bootstrapped V(s') term.
Since every non-goal state has the identical step cost, discounting never
changes the RELATIVE ranking of "closer to goal" vs "farther from goal" for
any gamma_demo > 0 -- discounting shrinks the future term uniformly, it
doesn't reorder it. The gamma-based demonstrator was therefore identical to
the true-optimal policy for every gamma_demo tested down to ~0.01, and only
became degenerate at the literal gamma_demo=0 boundary: a step function, not
a bias severity axis.

Bounded planning horizon does not have that problem: it directly limits
which states the goal is even VISIBLE from. A demonstrator that only plans
H steps ahead cannot see reward more than H steps away, so states farther
than H steps from the goal genuinely lose all directional signal (every
action looks equally good, all leading to H-step futures that never see
reward) -- this creates a real, spatially-graded blind spot that grows and
shrinks smoothly as H is swept, verified empirically in
experiments/sanity_check_myopia.py rather than assumed.
"""

import numpy as np


def finite_horizon_policy_probs(env, planning_horizon, gamma=None, tol=1e-9):
    """
    Stochastic finite-horizon policy: pi(a|s) = uniform over the actions
    that ACTUALLY tie for Q(s,.) under the truncated (value-assumed-0-beyond-
    horizon) backup; one-hot when there is a unique best action.

    Why this exists (bug found during AL well-posedness diagnosis): the
    original deterministic finite_horizon_policy used Q.argmax(axis=1),
    which silently picks the FIRST action index on a tie. Because reward(s)
    is action-invariant here, EVERY state beyond the horizon has Q(s,a)
    EXACTLY equal across all four actions (provably -- the goal signal
    cannot propagate past H hops in H backup rounds, so every successor's
    truncated value is 0, for every action, at every such state). argmax
    therefore picked action 0 (UP) at EVERY beyond-horizon state, regardless
    of position -- including states where UP walks into a wall. That is not
    a "myopic" choice, it is an indexing artifact: verified directly
    (see experiments/diagnose_al_wellposedness.py) that this deterministic
    policy is NOT achievable as the best response to ANY state-indexed
    reward (an LP feasibility check, Ng & Russell style), i.e. it isn't
    even a coherent policy for this MDP's restricted (state-only) reward
    class, let alone a meaningful bias mechanism.

    Fix: beyond-horizon states are genuinely INDIFFERENT under the
    truncated objective -- the honest way to represent indifference is a
    uniform mixture over the tied actions, not an arbitrary deterministic
    pick. This does mean the myopia demonstrator is no longer purely
    deterministic (it's deterministic within the visibility radius,
    uniform-random beyond it) -- a real change to the bias model's
    character, not a cosmetic patch. See
    experiments/myopia_tiebreak_finding.md for the full writeup of why
    this doesn't restore "myopia = best-response vertex" (it doesn't --
    genuine indifference-handling reintroduces mixing for the same reason
    Boltzmann bias does) but does fix the spatial-incoherence bug.
    """
    if gamma is None:
        gamma = env.gamma
    n_states, n_actions = env.transitions.shape
    V = np.zeros(n_states)
    for _ in range(planning_horizon):
        Q = env.true_rewards[:, None] + gamma * V[env.transitions]
        V = Q.max(axis=1)
    Q = env.true_rewards[:, None] + gamma * V[env.transitions]
    q_max = Q.max(axis=1, keepdims=True)
    is_best = np.abs(Q - q_max) < tol
    probs = is_best / is_best.sum(axis=1, keepdims=True)
    return probs


def generate_trajectories_stochastic(env, policy_probs, start_states, horizon, rng):
    """Roll out the (possibly tie-mixed) finite-horizon policy, sampling
    actions at each visit the way boltzmann.generate_trajectories does --
    same rationale: argmax over probs would silently collapse ties back to
    the old deterministic-first-index behavior."""
    n_actions = policy_probs.shape[1]
    trajs = []
    for s0 in start_states:
        s = s0
        traj = [s]
        for _ in range(horizon - 1):
            a = rng.choice(n_actions, p=policy_probs[s])
            s = env.transitions[s, a]
            traj.append(s)
        trajs.append(traj)
    return trajs


def finite_horizon_policy(env, planning_horizon, gamma=None):
    """
    Deterministic policy of an agent that only plans `planning_horizon`
    steps ahead on the TRUE reward (assumes value is exactly 0 beyond that
    horizon), then acts greedily on that truncated plan. Uses the TRUE
    discount gamma_true by default -- the bias here is bounded foresight,
    not a wrong discount rate.

    planning_horizon=0 means "see nothing beyond the immediate reward of
    the current state" -- since reward(s) doesn't vary by action here, this
    degenerates to a tie across all four actions everywhere (equivalent to
    the old gamma_demo=0 edge case). Larger planning_horizon values
    progressively reveal more of the grid; once planning_horizon covers the
    grid's diameter, every state can see the goal and this converges to the
    fully-optimal policy.

    NOT the demonstrator generator (see finite_horizon_policy_probs /
    generate_trajectories_stochastic for that) -- this deterministic,
    argmax-tie-broken version is kept only because the H-sweep
    disagreement-count smoothness check (experiments/sanity_check_myopia.py)
    doesn't depend on WHICH tied action gets picked, only on whether a
    state disagrees with true-optimal at all. Using this for actual
    demonstrations reintroduces the fixed tie-break artifact.
    """
    if gamma is None:
        gamma = env.gamma
    V = np.zeros(env.n_states)
    for _ in range(planning_horizon):
        Q = env.true_rewards[:, None] + gamma * V[env.transitions]
        V = Q.max(axis=1)
    Q = env.true_rewards[:, None] + gamma * V[env.transitions]
    return Q.argmax(axis=1)


def generate_trajectories(env, policy, start_states, horizon):
    """
    Deterministic rollout of `policy` from each start state. Myopia has no
    stochastic component at rollout time (unlike the Boltzmann
    demonstrator) -- all the bias is already baked into which policy the
    bounded-horizon plan produced; execution of that policy is exact.
    """
    trajs = []
    for s0 in start_states:
        s = s0
        traj = [s]
        for _ in range(horizon - 1):
            s = env.transitions[s, policy[s]]
            traj.append(s)
        trajs.append(traj)
    return trajs
