# IRL Bias-Robustness Comparison — Validated Foundation

Comparing **Maximum-Entropy IRL** (Ziebart et al. 2008) and **Apprenticeship
Learning** (Abbeel & Ng 2004) on robustness to demonstrator bias in small
discrete environments.

## Status: foundation + both bias demonstrators validated, ready for Phase 3 (experiment runner)

What exists and is tested:
- `src/gridworld.py` — Gridworld MDP, value iteration (with overridable discount).
- `src/maxent_irl.py` — Max-Ent IRL. **Assumes a Boltzmann-rational demonstrator**
  — this is the generative-model assumption behind the confound the paper controls for.
  Fixed a normalization bug where the expected state-visitation-frequency computation
  summed over the whole horizon while the empirical demo feature counts averaged per
  visit — a 1/horizon scale mismatch that made the IRL gradient never converge (it
  drifted at a constant rate). Invisible on optimal demos (differential signal
  dominated the drift); catastrophic on noisy demos. Fixed by normalizing the SVF
  by horizon.
- `src/apprenticeship.py` — Abbeel & Ng projection method. Makes **no** probabilistic
  assumption about the demonstrator; recovers a reward *direction*.
- `src/boltzmann.py` — **primary bias model.** Stochastic policy
  `P(a|s) = softmax(beta * Q[s])`, plus a trajectory sampler. `beta` is the
  rationality knob: low = noisy/near-random, high = near-deterministic-optimal.
  Sanity-checked in `experiments/sanity_check_boltzmann.py` (beta sweep, regret
  falls to ~0 at high beta and rises smoothly as beta drops) and cross-checked
  against the deterministic optimal-demo baseline at beta=1000.
- `src/myopia.py` — **confound-control bias model.** See "Myopia: gamma vs.
  finite-horizon" below — this is not what the original plan specified, and the
  reason it changed is itself a methodological point worth keeping.
- `src/metrics.py` — **Value regret** (normalised expected value difference). The
  core evaluation metric. Tie-robust and shaping-robust — chosen over action-matching
  and raw reward-distance for reasons documented in the file. Also provides
  `stochastic_value_regret` / `stochastic_policy_value_under_true`, the
  generalisation to stochastic policies needed for severity-matching (below).
- `tests/test_foundation.py` — locks in: metric correctness, tie-robustness, the
  un-biased validation gate (both methods recover ~0 regret from optimal demos),
  and Boltzmann recovery (high-beta near-zero regret, low-beta meaningfully worse,
  no NaNs — guards against silently argmax'ing instead of sampling, or overflowing
  the softmax).

Run the tests: `python3 tests/test_foundation.py`

## Key decision already made (propagate to the Method chapter)
Evaluation metric = **value regret**, not action-agreement. Action-agreement
conflates genuine recovery error with arbitrary tie-breaking (17/25 states in the
5x5 grid have tied-optimal actions). Value regret measures how much true-reward value
the recovered policy loses — continuous, so it captures the *shape* of degradation
under bias, not just pass/fail. This was discovered empirically via the validation
gate; write it up as a methodological choice, not an afterthought.

## Severity-matching (propagate to the Method chapter)
Raw bias parameters aren't comparable across bias models — `beta=0.5` and a
planning horizon of `H=3` mean nothing next to each other. The shared currency is
the **demonstrator's own value regret** under the true reward: how much true value
does a demonstrator following this bias actually lose, before any recovery is even
attempted. This is computed identically regardless of which bias model produced the
demonstrator (`value_regret` for deterministic demonstrators like myopia,
`stochastic_value_regret` for stochastic ones like Boltzmann), so "myopia severity
0.4" and "Boltzmann severity 0.4" mean the same thing. This is what makes the
headline regret-vs-severity comparison principled instead of comparing apples to
oranges.

## Myopia: gamma vs. finite-horizon (propagate to the Method chapter)
The original plan specified myopia as `gamma_demo < gamma_true` (plan with a smaller
discount, then act). That was implemented and rejected after sanity-checking it:

- **(a) Why gamma-based myopia is degenerate here:** this environment attributes
  reward to the *current* state, not to the action taken (`reward(s)`, not
  `reward(s,a)` or `reward(s')`), so every non-goal action from a given state incurs
  the identical step cost. `Q(s,a) = reward(s) + gamma*V(s')` therefore varies across
  actions *only* through the bootstrapped `V(s')` term. Discounting shrinks that
  future term uniformly — it never reorders which neighbor looks better. Confirmed
  directly: the gamma-based demonstrator's policy was bit-for-bit identical to the
  true-optimal policy for every `gamma_demo` tested down to ~0.01, and only became
  degenerate at the literal `gamma_demo=0` boundary (where reward becomes fully
  action-invariant and ties resolve to argmax's first index). A step function, not a
  bias-severity axis.
- **(b) Fix — finite-horizon lookahead:** myopia is instead operationalised as
  bounded planning depth: plan `H` steps ahead (truncated value iteration, value
  assumed 0 beyond the horizon), then act greedily. Arguably this is more literally
  "myopic" anyway (bounded foresight, the ordinary-language meaning) than a discount
  factor ever was.
- **(c) Verified, not assumed, to be smooth:** before spending compute on Max-Ent
  recovery, the horizon sweep alone was checked for state-disagreement-vs-optimal
  counts across `H=0..15`: `25 → 22 → 19 → 15 → 10 → 6 → 3 → 1 → 0` (H=0 through the
  grid's diameter, H=8). Smooth and monotonic — a real curve, unlike the gamma
  version. See `experiments/sanity_check_myopia.py`.
- **(d) The off-model property is unchanged and is the entire point.** Myopia exists
  because Max-Ent's generative model assumes a Boltzmann-rational (stochastic)
  demonstrator — so a Max-Ent advantage measured under Boltzmann bias could just be
  Max-Ent "knowing" the right noise model, not genuine robustness. A finite-horizon
  planner is fully deterministic, so it is off-model for Max-Ent in a completely
  different way than Boltzmann bias is (no action noise at all, just a truncated
  objective). It is *also* off-model for Apprenticeship Learning, which assumes the
  demonstrator is (near-)optimal for the true infinite-horizon objective — a
  bounded-horizon planner is a coherent optimiser, just for the wrong (truncated)
  objective, violating that assumption too, in yet another different way. Any
  Max-Ent-vs-Apprenticeship asymmetry that survives under myopia is much less likely
  to be an artifact of either method's own generative assumption. Changing the
  *mechanism* (gamma → horizon) does not change this role.

## Provisional observation — hold loosely until Phase 3
`experiments/sanity_check_myopia.py` overlays the Boltzmann and myopia sweeps on the
shared severity axis. At matched severity (~0.5), myopia drives **Max-Ent's**
recovered regret much higher than Boltzmann bias does at the same severity level.
This is consistent with the home-turf/off-model validity concern above — but it is
**not yet evidence about the research question**, for reasons that matter enough to
write down explicitly rather than let this quietly turn into an assumption:

- It's a single configuration: one grid, no seeds, one severity point region,
  **Max-Ent only**. The research question is comparative (Max-Ent vs. AL under
  matched bias); a single-method observation cannot speak to it.
- It's confounded in exactly the way the myopia check exists to resolve. "Myopia
  hurts Max-Ent more than Boltzmann does" is uninterpretable on its own — it could
  mean Max-Ent has a Boltzmann home-turf advantage, or it could just mean myopia is
  a harder bias for *any* method to recover through. Distinguishing those requires
  Apprenticeship Learning in the picture. Until then this is suggestive, not
  informative.
- **Guard against this shaping Phase 3.** Having glimpsed a direction (clean
  asymmetry, outcome-flavored), the risk is unconsciously building the runner,
  choosing severity points, or reading plots in ways that confirm it. The project's
  framing is characterisation, not prediction, specifically to stay honest across
  all four possible outcomes (including "asymmetry is a Boltzmann artifact" and "no
  asymmetry at all") — this is the moment that discipline actually gets tested.

Status: early single-method signal, direction = Max-Ent more sensitive to myopia
than to matched-severity Boltzmann bias. To be confirmed or refuted by the
comparative run, not treated as a working hypothesis.

## What YOU build next (in order)
1. ~~Boltzmann demonstrator~~ — done, sanity-checked, cross-checked against the
   optimal-demo baseline, locked in by `test_boltzmann_recovery`.
2. ~~Myopia demonstrator~~ — done (finite-horizon, see above), sanity-checked for
   smooth degradation.
3. **Experiment runner — build the SPINE first, robustness rings second.** The
   matrix of possible axes (beta sweep, H sweep, severity-matching, two methods,
   multiple seeds, eventually multiple grids) is large enough that "dump the whole
   matrix and figure out the plot later" produces an unreadable results chapter.
   Build, in this order:
   - **The spine (build and get right first):** regret vs. matched bias severity,
     one line per method (Max-Ent, Apprenticeship), **primary bias model
     (Boltzmann)**, canonical grid, **error bands over seeds**. This is the paper's
     headline figure.
   - **The confound panel (build second, right next to the spine):** the same
     regret-vs-severity plot, one line per method, but under **myopia**. This is
     what actually resolves the provisional observation above — does the asymmetry
     (if any) survive?
   - **Robustness rings (build last, only after the spine works):** everything
     else — additional grids, the raw beta-vs-H decomposition, etc. These support
     the headline result; they are not the headline result.
   - Loop over {method} x {bias model} x {bias level} x {seed}, write results to a
     CSV keyed so the spine and confound panel can each be sliced out cleanly.
4. **Analysis/plots** — render the spine and confound panel from the CSV. Confound
   check: does any asymmetry found under Boltzmann survive under myopia, or does it
   evaporate (myopia-is-just-harder-for-everyone)? Report whichever of the four
   outcomes actually shows up — don't reach for the one from the provisional
   observation above.

## Design decisions locked in planning (don't re-litigate without reason)
- Thesis: how do the two methods compare in robustness to matched demonstrator bias,
  and does any divergence support a principled method choice? (Characterise, don't
  predict — survives all four outcomes incl. "asymmetry is a Boltzmann artifact".)
- Two bias models: Boltzmann (primary) + myopia (confound control, finite-horizon
  mechanism — see above).
- Small discrete environments justified by *inspectability*, not convenience.
- Central validity threat: Max-Ent assumes Boltzmann demonstrators, so a Max-Ent
  advantage under Boltzmann bias could be an artifact — hence the myopia check.
- Bias severity is measured via the demonstrator's own value regret (shared scale
  across bias models), not via raw bias parameters — see "Severity-matching" above.

## Open items carried from planning
- `gap-verification-v2`: confirmatory database scan (Semantic Scholar/OpenAlex) for
  any existing *comparative* matched-bias robustness study, before writing the
  Introduction's gap claim.
- Verify Max-Ent IRL's stated self-motivation (robustness-to-noise vs ambiguity-
  resolution) against the primary source, for the hypothesis framing.
