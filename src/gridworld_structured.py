"""
Structured-feature gridworld -- built to test whether the bimodal
Max-Ent-under-bias failure mode found in the 5x5 one-hot grid
(experiments/failure_branch_diagnostic.py) is a general phenomenon or an
artifact of that environment's specific degeneracies (heavy tie structure,
one-hot features giving the true reward essentially one dominant signal
dimension).

Does NOT modify src/gridworld.py. Reuses the existing Gridworld class
(transitions, value_iteration) unchanged -- only the feature_map and
theta_true construction differ. Weights below were chosen by a-priori
reasoning before running any battery check, and were not adjusted
afterward to make anything pass.

Layout (size x size grid, goal at bottom-right corner):
  - Feature 0: negative normalised Manhattan distance to goal, in [-1, 0].
    Smooth gradient toward the goal.
  - Feature 1: hazard indicator (0/1) for a block of cells placed off the
    direct path from most start states -- creates a real "detour or cross
    it" trade-off rather than a hard no-go, IF the weight magnitude is
    modest relative to the distance feature's full range.
  - Feature 2: rough-terrain indicator (0/1), a DIFFERENT block of cells
    (doesn't overlap the hazard region), a second, separate trade-off axis.
  - Feature 3: bias (constant 1) -- a flat per-step cost, independent of
    position, so "shorter paths" have a small independent pressure beyond
    the smooth distance gradient.

Weight choice (a priori, not tuned against results):
  w_dist=1.0, w_hazard=-0.3, w_terrain=-0.15, w_bias=-0.02
  Reasoning: hazard penalty (-0.3) is meaningfully smaller than the
  distance feature's full range (1.0), so crossing hazard is worth it if
  it saves a large detour, but not worth it for a trivial 1-cell saving --
  a genuine trade-off rather than a hard constraint. Terrain penalty is
  half of hazard's, a milder version of the same idea. Bias is small,
  a weak tiebreaker toward shorter paths in general.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gridworld import Gridworld


def make_structured_gridworld(size=7, gamma=0.9):
    n_states = size * size
    goal_r, goal_c = size - 1, size - 1
    max_dist = (size - 1) + (size - 1)

    # hazard block: rows 1-3, cols 3-5 (clamped to grid) -- upper-middle-right
    hazard_cells = {(r, c) for r in range(1, 4) for c in range(3, 6)
                    if r < size and c < size}
    # terrain block: rows 4-5, cols 0-2 (clamped) -- lower-left, no overlap with hazard
    terrain_cells = {(r, c) for r in range(4, 6) for c in range(0, 3)
                     if r < size and c < size}

    n_features = 4
    feature_map = np.zeros((n_states, n_features))
    for s in range(n_states):
        r, c = divmod(s, size)
        dist = abs(goal_r - r) + abs(goal_c - c)
        feature_map[s, 0] = -dist / max_dist
        feature_map[s, 1] = 1.0 if (r, c) in hazard_cells else 0.0
        feature_map[s, 2] = 1.0 if (r, c) in terrain_cells else 0.0
        feature_map[s, 3] = 1.0

    theta_true = np.array([1.0, -0.3, -0.15, -0.02])

    env = Gridworld(size, feature_map, theta_true, gamma=gamma)
    feature_names = ["neg_dist_to_goal", "hazard", "terrain", "bias"]
    goal_state = goal_r * size + goal_c
    return env, feature_names, goal_state, hazard_cells, terrain_cells
