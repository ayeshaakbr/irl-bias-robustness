"""
Gridworld MDP for IRL experiments.

An N x N grid. States are cells, indexed 0..N*N-1. Actions are up/down/left/right.
Movement is deterministic here (we keep it simple for the first validation slice;
stochasticity can be added later). Bumping a wall keeps you in place.

Each state has a feature vector; the true reward is a linear function of features:
    reward(s) = feature(s) . theta_true
This linear-in-features setup is the standard IRL assumption (both Abbeel & Ng
and Ziebart assume it) and is what lets us talk about "recovering theta".
"""

import numpy as np

# Action encoding
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


class Gridworld:
    def __init__(self, size, feature_map, theta_true, gamma=0.9):
        """
        size:        grid is size x size
        feature_map: array (n_states, n_features), one feature vector per state
        theta_true:  array (n_features,), true reward weights
        gamma:       discount factor
        """
        self.size = size
        self.n_states = size * size
        self.feature_map = np.asarray(feature_map, dtype=float)
        self.n_features = self.feature_map.shape[1]
        self.theta_true = np.asarray(theta_true, dtype=float)
        self.gamma = gamma

        assert self.feature_map.shape[0] == self.n_states
        assert self.theta_true.shape[0] == self.n_features

        self.transitions = self._build_transitions()
        self.true_rewards = self.feature_map @ self.theta_true  # (n_states,)

    def state_to_rc(self, s):
        return divmod(s, self.size)

    def rc_to_state(self, r, c):
        return r * self.size + c

    def _build_transitions(self):
        """Return P[s, a] -> s' (deterministic). Wall bumps stay in place."""
        P = np.zeros((self.n_states, len(ACTIONS)), dtype=int)
        for s in range(self.n_states):
            r, c = self.state_to_rc(s)
            for a in ACTIONS:
                dr, dc = ACTION_DELTAS[a]
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    P[s, a] = self.rc_to_state(nr, nc)
                else:
                    P[s, a] = s  # bump wall, stay
        return P

    def value_iteration(self, rewards=None, gamma=None, tol=1e-8, max_iter=10000):
        """
        Standard value iteration. Returns (V, Q, policy).
        rewards: per-state reward vector; defaults to true rewards.
        gamma:   allows overriding discount (used later for the myopia demonstrator).
        """
        if rewards is None:
            rewards = self.true_rewards
        if gamma is None:
            gamma = self.gamma

        V = np.zeros(self.n_states)
        for _ in range(max_iter):
            Q = rewards[:, None] + gamma * V[self.transitions]  # (n_states, n_actions)
            V_new = Q.max(axis=1)
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new
        Q = rewards[:, None] + gamma * V[self.transitions]
        policy = Q.argmax(axis=1)
        return V, Q, policy
