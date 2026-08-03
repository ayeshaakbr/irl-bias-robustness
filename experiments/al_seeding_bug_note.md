# Bug note: `apprenticeship_learning` self-seeding degeneracy

Status: local note, NOT pushed. Documenting a real bug found in the
existing, unmodified `src/apprenticeship.py` so it isn't lost before the
next person touches that file.

## The bug

`apprenticeship_learning` (src/apprenticeship.py) seeds its mixture with:

```python
_, _, policy0 = env.value_iteration()
```

`env.value_iteration()` with no `rewards` argument defaults to
`env.true_rewards` — i.e. `policy0` is always the true-optimal policy,
regardless of what the expert demonstrator actually is.

This is harmless when the expert differs from the true-optimal policy
(any biased demonstrator: Boltzmann, myopia at any H<8 tested this
session) — the initial margin is nonzero and the projection loop proceeds
normally. It is NOT harmless when testing the FOUNDATION GATE specifically
(expert = true-optimal policy, i.e. "recover from optimal demonstrations"):
in that case `policy0 == expert` exactly, so `fe0 == expert_fe` exactly,
so the very first computed margin is exactly 0, the loop's
`if t < tol: break` fires on iteration 0, and the function returns
`w / (np.linalg.norm(w) + 1e-12)` where `w` is a zero (or floating-point-
noise) vector — an essentially meaningless reward direction.

## How it was found and confirmed

Found via the structured-gridworld (`src/gridworld_structured.py`)
foundation-gate check: AL regret from optimal demonstrations was 0.7689
(should be ~0). Ruled out iteration budget as the cause directly: reran
at n_iters = 200, 500, 2000, 8000 -- bit-identical regret (0.7689) every
time, confirming a genuine stuck point, not slow convergence. Confirmed
the mechanism directly: `policy0 == pi_true` is `True`, and
`||expert_fe - fe0|| == 0.0` exactly at iteration 0.

Same class of bug as the one already found and fixed (via a different
seed choice) in this session's own `experiments/hard_margin_al.py` --
except that fix was only applied to the new file, never to the original
`apprenticeship_learning`.

## Why this does not contaminate any finding in this project

Every substantive AL result produced this session (the 2x2 battery: QP/
separability infeasibility tracking severity, projection-method
convergence under Boltzmann vs. plateau under myopia) used a BIASED
demonstrator as the expert -- never the exact true-optimal policy. The
degeneracy only triggers on the trivial self-match case, which none of
the reported findings depend on.

## Left unfixed, pending a decision

Two candidate fixes, not applied without a decision:
1. Seed with a fixed adversarial/worst-case policy instead of the default
   true-optimal one (the fix already used in
   `experiments/hard_margin_al.py` via `metrics._worst_case_policy`).
2. Special-case: if `expert_fe` matches `fe0` within tolerance at
   initialization, treat it as an immediate, correct "matched" success
   rather than falling through to a meaningless normalized-zero vector.

Either fixes the foundation gate; (1) is probably the more general fix
since it also avoids the same trap in any future test where the expert
happens to coincide with whatever `env.value_iteration()`'s default
produces.
