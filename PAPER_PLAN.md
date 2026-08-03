# Paper Plan — IRL Bias-Robustness Comparison (option-2: cautionary/methodological)

Status: locked Chapter Plan + INSIGHT collection from an `academic-paper`
plan-mode session. Committed locally for durability (this file, not chat
memory, is the record). Target: NeurIPS-style workshop paper, ~4-8 pages,
IMRaD, "Applied AI Research" track, portfolio piece. Breadth paper in a
two-paper arc; paper #2 goes deeper on one method with richer bias models.

## How this paper's direction changed

Originally planned as a head-to-head robustness comparison (Max-Ent IRL vs.
Apprenticeship Learning under matched demonstrator bias). A long diagnostic
process — chronicled in `experiments/*.md` and `experiments/*.py` in this
directory — found that the comparison, as originally posed, rests on
assumptions two independent small-gridworld testbeds do not support:
population-level Max-Ent recovery is largely bias-insensitive in both, the
apparent degradation is a finite-sample/extraction artifact, and only one
piece of the original mechanism story (AL's strict-separation formulation
failing against interior/stochastic targets) survived full scrutiny. The
paper pivots to what the evidence actually supports: a systematic account
of why this comparison is harder to pose validly than it looks, plus a
reusable checklist, plus the one AL finding that held up.

This is a considered decision, not a fallback — see the diagnostic trail
below for why.

## Contribution statement (locked)

> We identify five necessary properties a testbed must satisfy to validly
> compare IRL methods' robustness to demonstrator bias, and show that
> violating them produces specific, characterizable failures — a checklist
> of ways a testbed can silently sabotage this comparison, not a
> certificate that passing it guarantees validity. This checklist is
> motivated by a striking pattern observed independently in two
> structurally different small discrete gridworlds: population-level
> Max-Ent recovery is largely insensitive to demonstrator bias severity,
> with the apparent robustness gap arising instead from finite-sample and
> policy-extraction effects. Within the one comparison that survives these
> confounds, we demonstrate that Apprenticeship Learning's strict-separation
> formulation becomes infeasible against stochastic (interior-point)
> demonstrators, consistent with the geometric fact that interior points
> admit no strict separator.

Discipline behind every clause: claim exactly what was shown, nothing
generalized past two testbeds, no word (e.g. "provably") claiming a formal
result that wasn't produced.

## Chapter Plan

### 1. Introduction (+ Related Work folded in, workshop page budget)
- Open with the contribution statement's structure in paragraph one —
  checklist + demonstrated phenomenon — not a slow "we tried X, found
  problems" reveal. The reframe IS paragraph one.
- State the original motivating question honestly (how do Max-Ent/AL
  compare under bias) as the entry point, pivot immediately, one sentence.
- Gap sentence: hedge-proof "we are not aware of..." form (survives the
  still-pending lit scan regardless of paper direction).
- Preview exactly three deliverables, in this order: the checklist
  (reusable), the population-insensitivity demonstration (the anchor),
  the AL infeasibility result (the clean secondary finding).
- Write for a specialist reader, one orienting sentence per method.

### 2. Method
- Two methods (Max-Ent IRL, Apprenticeship Learning), one orienting
  sentence each.
- Two bias models (Boltzmann primary, finite-horizon myopia
  confound-control) + the off-model logic, framed explicitly as apparatus
  built for the diagnostic, not for a robustness curve.
- Value regret metric + severity-matching (unchanged from original plan,
  still fully valid, tie-robust/shaping-robust justification intact).
- **The five-check testbed-validity battery** (named, numbered, elevated
  to primary contribution, cross-referenced from abstract):
  1. Foundation gate — both methods recover ~0 regret from optimal demos.
  2. Tie sanity — fraction of states with tied-optimal actions under the
     true reward (low tie-fraction needed, or bimodal artifacts follow).
  3. Continuity not bimodality — per-seed regret distribution at fixed
     high bias, swept over sample size; a populated middle is required.
  4. Non-trivial population-level sensitivity — analytic (zero-sampling-
     noise) recovery must show SOME bias-severity sensitivity, or the
     testbed cannot exhibit the phenomenon at all.
  5. Extraction stability — gap between the soft-fit target and the
     recovered hard/argmax policy's own feature expectation; large gaps
     mean argmax extraction is fragile independent of the fit quality.
  - Explicit epistemic framing: these are **necessary conditions whose
    violation is shown to cause specific failures — not a sufficiency
    guarantee.** Passing all five doesn't certify validity; it rules out
    the specific failure modes this paper characterizes.

### 3. Results — pedagogical order (pruned of dead ends that don't teach)
- **Lead with the population-insensitivity finding**, framed explicitly as
  motivation for the checklist, not a standalone claim about Max-Ent:
  the two-testbed numbers (5x5 one-hot, 7x7 structured-feature) presented
  side by side as bounded evidence, immediately followed by "this is why
  testbed validity needs checking at all" — the scoping is structural,
  not a caveat bolted on after.
- Then, in pedagogically-clearest order: metric/tie artifact (value regret
  vs. action-matching) -> finite-sample bimodality (regret-vs-N shape,
  not smooth) -> extraction-instability gap -> AL geometric infeasibility.
- Keep only the one abandoned hypothesis that teaches: "failure might be
  expected to track feature-count deviation from the true demonstrator
  expectation; it does not" — with the actual numbers (0.0046 vs 0.0045
  success vs. failure, overlapping ranges). Cut the interior-vs-vertex
  mechanism's full death and the QP-evidence-was-vapor history — process,
  not pedagogy.
- Report AL's infeasibility with its own validation gates inline (matches
  the known-good vertex case exactly; shows genuine non-trivial
  infeasibility — not a degenerate self-comparison artifact — on the
  known-interior case) so the result reads as independently verified.
- Both testbeds' battery tables presented together, explicitly showing
  what improved (tie-fraction: 68% -> 33%) and what didn't move at all
  (population sensitivity, extraction gap) — that contrast is what makes
  two testbeds more convincing than one.

### 4. Discussion
- Restate the checklist as forward-looking guidance for future
  researchers, not a summary of what failed here.
- AL infeasibility scoped precisely: the formulations tested (projection,
  hard-margin/separability), not a universal claim about strict-separation
  methods generally.
- Every claim in "across two structurally different testbeds, we
  demonstrate..." form. No general laws about Max-Ent or small
  environments.
- **New beat (field-relevance):** small discrete gridworlds are the
  default/standard evaluation environment for IRL method comparisons
  generally (citable, true, non-accusatory claim) — so the confounds
  characterized here plausibly extend beyond this paper's own experiments.
  This is what elevates the contribution from "our experiment was hard" to
  "a widely-used evaluation practice has under-appreciated confounds." No
  naming of specific papers/authors.
- Limitations named directly: two bias models, two testbeds, small
  discrete state spaces only, gridworld transition structure throughout.
- Future work: paper #2, opening with a testbed designed to satisfy the
  checklist by construction rather than discovered to fail it.

## Open items gating the draft (resolve before writing, not during)

- `gap-verification-v2`: confirmatory database scan (Semantic
  Scholar/OpenAlex) for existing work on IRL bias-robustness testbed
  validity / comparative studies, before finalizing the Introduction's gap
  sentence. Gates Introduction only.
- Ziebart Max-Ent self-motivation check: verify against the primary source
  whether Max-Ent's stated motivation is noise-robustness or
  ambiguity-resolution. Not yet resolved. Gates the Method chapter's
  one-sentence orientation of Max-Ent and how the bias models are framed
  as apparatus — settle before drafting those sentences, not after.

## Known bug carried forward (does not contaminate this paper's findings)

See `experiments/al_seeding_bug_note.md`: `apprenticeship_learning` in
`src/apprenticeship.py` has a self-seeding degeneracy when the expert
policy exactly equals the projection loop's initial seed policy (both
default to the true-optimal policy via `env.value_iteration()` with no
reward override) — margin is exactly 0 at iteration 0, loop breaks
immediately, returns a near-zero/meaningless direction. Confirmed via the
structured-gridworld foundation gate (regret 0.77 from optimal demos,
identical across 200-8000 iterations, ruling out an iteration-budget
explanation). Does not affect any finding in this plan: every AL result
used biased demonstrators, never the exact self-match case. Left unfixed
pending a decision on the right general fix (e.g. seed with a fixed
adversarial policy instead of the default true-optimal one).

## Drafting order

Method first (most stable, least contested content; drafting the
checklist section concretely clarifies how Intro/Discussion reference it)
-> close the two open items above -> Introduction -> Results -> Discussion.
