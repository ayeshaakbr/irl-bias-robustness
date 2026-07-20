# IRL Bias-Robustness Comparison — Validated Foundation

Comparing **Maximum-Entropy IRL** (Ziebart et al. 2008) and **Apprenticeship
Learning** (Abbeel & Ng 2004) on robustness to demonstrator bias in small
discrete environments.

## Status: foundation validated, ready to build the bias experiments

What exists and is tested:
- `src/gridworld.py` — Gridworld MDP, value iteration (with overridable discount,
  which the myopia demonstrator will need).
- `src/maxent_irl.py` — Max-Ent IRL. **Assumes a Boltzmann-rational demonstrator**
  — this is the generative-model assumption behind the confound the paper controls for.
- `src/apprenticeship.py` — Abbeel & Ng projection method. Makes **no** probabilistic
  assumption about the demonstrator; recovers a reward *direction*.
- `src/metrics.py` — **Value regret** (normalised expected value difference). The
  core evaluation metric. Tie-robust and shaping-robust — chosen over action-matching
  and raw reward-distance for reasons documented in the file.
- `tests/test_foundation.py` — locks in: metric correctness, tie-robustness, and the
  un-biased validation gate (both methods recover ~0 regret from optimal demos).

Run the tests: `python3 tests/test_foundation.py`

## Key decision already made (propagate to the Method chapter)
Evaluation metric = **value regret**, not action-agreement. Action-agreement
conflates genuine recovery error with arbitrary tie-breaking (17/25 states in the
5x5 grid have tied-optimal actions). Value regret measures how much true-reward value
the recovered policy loses — continuous, so it captures the *shape* of degradation
under bias, not just pass/fail. This was discovered empirically via the validation
gate; write it up as a methodological choice, not an afterthought.

## What YOU build next (in order)
1. **Boltzmann demonstrator** — generate trajectories via softmax(beta * Q) on the
   true reward. `beta` is the rationality knob (low = noisy, high = near-optimal).
   This gives you the primary bias model AND the beta sweep for free.
   - First check: as beta -> infinity, recovered regret -> 0 (recovers the
     optimal-demo result). As beta -> 0, demos become random, regret should rise.
2. **Myopia demonstrator** — plan with gamma_demo < gamma_true (use the gamma override
   already in `value_iteration`), then act. The off-model-for-both validity check.
3. **Experiment runner** — loop over {method} x {bias model} x {bias level} x {seeds},
   write results to CSV. Keep the primary result axis clean: regret vs bias level,
   one line per method, on the canonical grid. Everything else is a robustness ring.
4. **Analysis/plots** — the headline figure is regret-vs-bias-severity, one line per
   method. Confound check: does any asymmetry survive under myopia?

## Design decisions locked in planning (don't re-litigate without reason)
- Thesis: how do the two methods compare in robustness to matched demonstrator bias,
  and does any divergence support a principled method choice? (Characterise, don't
  predict — survives all four outcomes incl. "asymmetry is a Boltzmann artifact".)
- Two bias models: Boltzmann (primary) + myopia (confound control).
- Small discrete environments justified by *inspectability*, not convenience.
- Central validity threat: Max-Ent assumes Boltzmann demonstrators, so a Max-Ent
  advantage under Boltzmann bias could be an artifact — hence the myopia check.

## Open items carried from planning
- `gap-verification-v2`: confirmatory database scan (Semantic Scholar/OpenAlex) for
  any existing *comparative* matched-bias robustness study, before writing the
  Introduction's gap claim.
- Verify Max-Ent IRL's stated self-motivation (robustness-to-noise vs ambiguity-
  resolution) against the primary source, for the hypothesis framing.
