# AL well-posedness under demonstrator bias: full diagnostic writeup

Status: local working note, NOT pushed to the repo. Synthesizes the whole
autonomous diagnostic session (myopia tie-break fix, QP validation, full
2x2 battery) into one record before anything gets written into the paper.

## Headline: the original mechanism hypothesis is falsified by clean data

The hypothesis chased for most of this session was: **AL's hard-margin
formulation is infeasible specifically against Boltzmann (a stochastic,
generically-interior-point demonstrator) and specifically feasible against
myopia (a deterministic, generically-vertex demonstrator)** -- i.e. the
Boltzmann/myopia distinction itself explains QP feasibility.

The full 2x2 battery (analytic feature expectations, both AL formulations,
both bias models, sweep of severities) does not support this. See
`battery_2x2_results.json` for raw numbers; table below.

| demonstrator | severity | QP separable? | projection final margin | projection behavior |
|---|---|---|---|---|
| Boltzmann beta=0.1 | 0.921 | **No** | 0.000000 | converges |
| Boltzmann beta=0.3 | 0.868 | **No** | 0.000000 | converges |
| Boltzmann beta=0.5 | 0.794 | **No** | 0.000000 | converges |
| Boltzmann beta=1.0 | 0.561 | **No** | 0.000000 | converges |
| Boltzmann beta=2.0 | 0.215 | **No** | 0.000000 | converges |
| Boltzmann beta=5.0 | 0.012 | Yes | 0.000442 | converges |
| myopia H=0 | 0.940 | **No** | 0.000000 | converges |
| myopia H=1 | 0.646 | **No** | 0.046782 | **plateau** |
| myopia H=2 | 0.458 | **No** | 0.035814 | **plateau** |
| myopia H=3 | 0.270 | **No** | 0.027120 | **plateau** |
| myopia H=4 | 0.133 | Yes | 0.009885 | plateau (small) |
| myopia H=5 | 0.061 | Yes | 0.004129 | plateau (small) |
| myopia H=6 | 0.020 | Yes | 0.001259 | plateau (small) |
| true-optimal | 0.000 | Yes (matched) | 0.000000 | converges |

Two genuinely different, more precise findings replace the original
hypothesis:

### Finding A: QP infeasibility tracks SEVERITY, not bias-model identity

Both Boltzmann and myopia become QP-infeasible once severity crosses
roughly ~0.1-0.3 in this environment, and both are feasible near severity
~0. This is a real, validated result (the QP-equivalent separability LP
passed its own validation gates: matches the true-optimal case exactly,
shows genuine non-trivial infeasibility -- 33 distinct probed vertices,
monotonic margin decay to exactly zero, not a degenerate artifact -- on
Boltzmann beta=0.5). But it is a severity effect common to both
mechanisms, not evidence that Boltzmann's stochasticity specifically
causes it. The interior-point/vertex geometric story doesn't discriminate
the way hoped, because (per the myopia tie-break finding) an honestly-
modeled myopic demonstrator is ALSO a mixture over enough of the state
space to behave like an interior point at meaningful severities.

**This is still a real, useful, and honestly broader finding**: AL's
strict-separation formulation becomes infeasible against demonstrators
with moderate-or-worse bias, for at least two structurally different bias
mechanisms tested. It's a claim about the formulation's fragility to bias
IN GENERAL (in this environment), not a Boltzmann-specific mechanism.

### Finding B: the projection method shows the OPPOSITE asymmetry, unexplained

Projection-method AL converges (given enough iterations, up to 8000
tested) to near-zero margin against Boltzmann at EVERY severity tested,
including the most severe (beta=0.1, severity 0.921). Against myopia, it
genuinely PLATEAUS (bit-identical margin over thousands of iterations,
confirmed not a 2-cycle) at every H representing meaningful bias (H=1
through 6, severity 0.02-0.65) -- worse behavior at LOWER severity than
Boltzmann's WORST tested point.

This is a clean, reproducible, opposite-direction asymmetry from the
original hypothesis, and I do not have a mechanism for it. Flagging it
rather than chasing it further, per the standing instruction not to
rabbit-hole -- this is a good candidate for the next diagnostic session,
not something to resolve or explain away right now.

## What actually survived from the original session's evidence

- AL's max-margin/hard-separation formulation genuinely fails (infeasible,
  not just slow) against demonstrators with meaningful bias severity. This
  was true in the original lost-session Boltzmann-only diagnostic and
  remains true here, independently re-derived via a validated LP rather
  than recovered from memory.
- AL's projection-method formulation does NOT fail against Boltzmann --
  it converges, just slowly (needs orders of magnitude more iterations
  than any practical budget, e.g. the runner's default of 50). This
  contradicts "the projection method also fails against Boltzmann" as
  originally characterized; the correct description is a compute-cost
  asymmetry, not a breakdown, exactly the distinction flagged earlier in
  this session before any of this evidence existed.
- Max-Ent's behavior under either bias model was NOT tested this session
  (out of scope for the AL-focused diagnostic) -- the original
  README's provisional single-method observation about Max-Ent-under-
  myopia remains exactly as uncertain as before, and is now ALSO
  affected by the myopia tie-break fix (severities have changed scale;
  any existing Max-Ent-vs-myopia numbers computed against the OLD buggy
  demonstrator should be treated as stale).

## Implications for the paper

**Method chapter:** the AL well-posedness finding is still headline-
worthy, but the mechanism paragraph must change. Cannot write "Boltzmann's
interior-point feature expectation makes AL's max-margin formulation
infeasible while myopia's vertex-hood keeps it well-posed" -- that's
false per Finding A. Can write: "AL's max-margin formulation becomes
infeasible once demonstrator bias exceeds a moderate severity threshold,
observed under two structurally different bias mechanisms (Boltzmann,
finite-horizon myopia); AL's iterative projection formulation remains
solvable under Boltzmann bias at every severity tested but requires
orders of magnitude more computation than a practical budget provides,
and empirically plateaus (does not converge within any iteration budget
tested) under myopic bias -- an asymmetry between the two AL variants'
behavior under the two bias models that is not yet mechanistically
understood."

**Scoped claim, updated:** the original scoping ("the formulations
tested break against Boltzmann; generalization is a limitation") should
become "AL's tested formulations show bias-severity-dependent failure
(hard-margin: infeasibility; projection: either slow convergence or a
genuine non-convergent floor, depending on which bias model) -- the
common thread is fragility to demonstrator bias generally in this
environment, not a Boltzmann-specific mechanism." This is a defensible,
even-more-honest version of the scoped claim, consistent with the
project's characterization-not-prediction discipline.

**Discussion chapter / four-outcome taxonomy:** this finding doesn't
cleanly fit "clean asymmetry / regime-dependent asymmetry / no asymmetry
/ Boltzmann-artifact asymmetry" as originally previewed in the
Introduction, because the AL-vs-MaxEnt story is now itself two different
sub-findings (hard-margin infeasibility that's severity-general;
projection-method plateau that's myopia-specific and unexplained) rather
than one clean mechanism. The Introduction's outcome-taxonomy preview may
need a note that the actual finding is more granular than a single
axis -- worth discussing before finalizing that section's wording, not a
decision to make unilaterally here.

**What still needs to happen before ANY of this is paper-ready:**
1. Max-Ent needs to be run through the SAME battery (both bias models,
   fixed myopia demonstrator, analytic + eventually sampled FE) so the
   paper's actual comparative claim (Max-Ent vs AL) has evidence, not just
   AL's internal behavior.
2. The projection-method myopia plateau (Finding B) needs a mechanism or
   at least a more careful characterization before it's mentioned as more
   than an open observation.
3. Everything here used analytic feature expectations. The real runner
   uses sampled trajectories -- sampling noise was "ruled out" in the
   original lost session for the Boltzmann-only result, but has not been
   re-checked against the corrected myopia demonstrator or the QP-
   equivalent LP test.
4. None of this is committed to the actual repo yet (by instruction --
   everything in this file and the accompanying scripts lives only in the
   local clone).
