# Myopia tie-break bug, and why myopia can't serve as AL's "vertex" control

Status: local working note, NOT pushed to the repo. Captures a same-session
finding so it isn't lost, same discipline as the gamma-myopia writeup in
the main README, until it's reviewed and ready to land there for real.

## The bug

`finite_horizon_policy` (src/myopia.py) used `Q.argmax(axis=1)`. Because
this environment's reward is attributed to the *state*, not the
state-action pair (`reward(s)`, not `reward(s,a)`), every state beyond the
planning horizon has **Q(s,a) exactly equal across all four actions** --
not approximately, exactly, provably: the goal signal cannot propagate
further than H hops in H rounds of backward induction, so every
successor's truncated value is 0 for every action, at every state whose
true distance to the goal exceeds H.

`argmax` breaks that tie by picking the first action index (UP) every
time. Checked directly for H=3: all 15 disagreeing-with-optimal states
take the identical action UP, including states where UP walks into a
wall. This is not bounded-foresight behavior, it's an indexing artifact.

**Proof it's not even a coherent policy for this MDP**, not just visually
odd: built the Ng & Russell-style IRL feasibility LP (does there exist a
state-indexed reward w under which this exact policy is the unique best
response?) and ran it against the H-sweep:

| H | disagrees w/ optimal | is best-response-achievable (vertex)? |
|---|---|---|
| 0 | 25/25 | False |
| 1 | 22/25 | False |
| 2 | 19/25 | False |
| 3 | 15/25 | False |
| 4 | 10/25 | False |
| 5 | 6/25  | False |
| 6 | 3/25  | False |
| 8 | 0/25  | True (= true-optimal policy, not biased anymore) |

Every H that represents real bias fails. Sanity check: true-optimal policy
passes trivially (True), confirming the LP itself is correct.

## The fix

Added `finite_horizon_policy_probs` + `generate_trajectories_stochastic`
(mirroring boltzmann.py's `policy_probs` + `rng.choice` pattern): beyond
the horizon, actions that genuinely tie get a uniform mixture instead of
an arbitrary deterministic pick. This is the textbook-correct way to
represent genuine indifference -- not a patch, a different (more honest)
model of what "no information beyond the horizon" means.

Old deterministic `finite_horizon_policy` is kept only for the H-sweep
disagreement-count smoothness plot, which doesn't care which tied action
gets picked, only whether a state disagrees with optimal at all. It
should not be used to generate demonstrations.

## Re-validation after the fix

Severity (stochastic_value_regret) sweep is still smooth and monotonic,
and is uniformly *lower* than the old buggy version at every H (makes
sense: "always walk into whatever's north, including walls" is a worse
policy than a fair random walk):

| H | old severity | new severity |
|---|---|---|
| 0 | 1.000 | 0.940 |
| 1 | 0.819 | 0.646 |
| 2 | 0.662 | 0.458 |
| 3 | 0.480 | 0.270 |
| 4 | 0.285 | 0.133 |
| 5 | 0.152 | 0.061 |
| 6 | 0.067 | 0.020 |
| 7 | 0.020 | 0.003 |
| 8+ | 0.000 | 0.000 |

Smooth, monotonic, passes the same pass condition as before. The bug fix
does not break myopia's role as a severity axis -- it changes the severity
SCALE (a given H is now "less severe" in the old buggy numbers than it
really is), which matters for any earlier result that used the buggy
severity values to pick severity-matched comparison points.

## Why this kills "myopia = vertex, discriminates AL infeasibility"

The AL mechanism hypothesis this session was chasing predicted: QP
infeasible on Boltzmann (interior/mixture target), feasible on myopia
(vertex/deterministic target). That discriminating design assumed myopia
gives a clean deterministic target. It doesn't, for a structural reason,
not an implementation one:

- The *buggy* deterministic version fails the is-vertex LP outright (shown
  above) -- arbitrary-but-deterministic is not the same as
  rational-hence-achievable-as-a-best-response.
- The *fixed*, honest version reintroduces genuine mixing for exactly the
  states that mattered (beyond-horizon indifference), because uniform
  mixing is the textbook-correct way to model indifference. That mixing
  makes the fixed myopia demonstrator's feature expectation a mixture
  too -- not the same generative structure as Boltzmann
  (deterministic-then-uniform vs. globally-softmax), but not a clean
  vertex either.

There may be no way to get a meaningfully-biased, purely-deterministic,
best-response-achievable myopic demonstrator in an environment with
state-only reward -- state-only reward is too restrictive a class to
rationalize arbitrary deterministic ties, and honest ties aren't
deterministic. This is a negative methodological result worth keeping,
not a failure: it's the same kind of finding as the gamma-myopia
degeneracy, caught by the same discipline (verify, don't assume).

**Consequence for QP validation:** myopia can't fill the "known-vertex,
should-succeed" role. The true-optimal policy already fills it (proven
above, is_vertex=True, and it's already the project's existing unbiased
validation gate) -- no new fixture needed.

## Thread opened, not yet explained at time of writing

Under the FIXED myopia demonstrator, projection-method AL was run out to
8000 iterations (analytic feature expectations):

- H=4, H=5 (lower severity, less of the state space in the indifferent
  region): margin keeps shrinking through 8000 iterations, Boltzmann-like.
- H=2, H=3 (higher severity, more indifferent states): margin is
  bit-identical from iteration ~1000 through iteration 8000 (ratio
  margin[-1]/margin[-2000] = 1.0 exactly) -- a genuine floor, not just slow
  convergence. Not a simple 2-cycle (best-response policies at early
  iterations are all distinct; the projection step's denominator stays
  well above the degenerate threshold through iteration 29).

See the 2x2 battery results (this same directory) for how this thread was
resolved.
