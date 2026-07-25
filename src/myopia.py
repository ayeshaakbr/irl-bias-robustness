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
