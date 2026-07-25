"""
Boltzmann-rational demonstrator (the primary bias model).

Max-Ent IRL's generative model ASSUMES the demonstrator picks actions via
    P(a|s) = softmax(beta * Q_true[s, a])
beta is the rationality knob: beta -> infinity collapses this onto the
deterministic optimal policy (the case the foundation already validates);
beta -> 0 makes actions uniform random regardless of Q. Because this is
also the exact model Max-Ent assumes internally, it's Max-Ent's "home turf"
bias -- which is why Phase 2 adds the myopia demonstrator as an off-model
check for both methods.
"""

import numpy as np


def boltzmann_policy(Q, beta):
    """
    Stochastic policy pi(a|s) = softmax(beta * Q[s, :]).

    Row-max subtraction before exponentiating: softmax is shift-invariant
    (softmax(x) == softmax(x - c) for any per-row constant c), so subtracting
    each row's max doesn't change the result. It does keep every exponent
    <= 0, which is what stops beta * Q from overflowing exp() to inf/nan at
    high beta (nan / nan in the normalisation is silent -- no exception).
    """
    scaled = beta * Q
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def generate_trajectories(env, Q, beta, start_states, horizon, rng):
    """
    Roll out the Boltzmann policy from each start state for `horizon` steps.

    Actions are sampled with rng.choice against the policy's probability
    row, not argmax. argmax(pi(.|s)) just picks the highest-probability
    action every time, which for a well-formed Boltzmann policy is always
    the true-optimal action -- so it would silently reproduce the
    deterministic optimal demonstrator no matter what beta is set to.
    Sampling is what actually makes beta control demonstration quality.

    rng: an np.random.default_rng(seed) instance, passed in by the caller
    so trajectory sampling stays reproducible per-seed without depending on
    global numpy random state.
    """
    policy = boltzmann_policy(Q, beta)
    n_actions = Q.shape[1]
    trajs = []
    for s0 in start_states:
        s = s0
        traj = [s]
        for _ in range(horizon - 1):
            a = rng.choice(n_actions, p=policy[s])
            s = env.transitions[s, a]
            traj.append(s)
        trajs.append(traj)
    return trajs
