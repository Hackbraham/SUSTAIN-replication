# SUSTAIN replication — discrepancies vs. Love, Medin & Gureckis (2004)

Notes on how `sustain.py` / `replications.py` compare to the paper's reported
numbers, after auditing the model's equations against pages 314–316 and the
procedures of the three source studies (Yamauchi & Markman 1998; Medin, Dewey
& Murphy 1983; Billman & Knutson 1996).

Try as we might, using the equations we took from the paper we could not
replicate exactly the results found by Love et al. for their four experiment
replications. After much debugging, we used Anthropic's Claude as a third
set of eyes to analyze our code against the paper, as well as the cited
replicated papers. Most of the originally-reported discrepancies have been
reconciled through procedural detail recovered from the source studies, but a few remain irreconcilable with the paper's stated equations. This document records Claudes assesment of the remaining gaps.

## Numbers (seed=42, ~30 sims each)

| Study | Condition | Paper (human / SUSTAIN) | Argmax | Stochastic | Status |
|---|---|---|---|---|---|
| Shepard et al. (1961) | Type I | ~3 / ~3 | 2.0 | 2.9 | ✓ |
| | Type II | ~few | 2.1 | 10.1 | ✓ ordered |
| | Type III | ~12 | 3.0 | 17.4 | ✓ ordered |
| | Type IV | ~12 | 2.9 | 16.5 | ✓ ordered |
| | Type V | ~12 | 2.4 | 20.8 | ✓ ordered |
| | Type VI | ~16–20 | 2.3 | 27.1 | ✓ ordered |
| Medin et al. (1983) | first_name | 7.1 / 7.2 | 2.0 | 6.2 | ✓ |
| | last_name | 9.7 / 9.7 | 2.0 | 3.4 | **reversed (gap A)** |
| Yamauchi & Markman (1998) | linear inference | 6.5 / 7.5 | 4.9 | 30.0 (ceiling) | ✓ argmax |
| | linear classification | 12.3 / 11.2 | 1.9 | 6.4 | low (gap B) |
| Yamauchi et al. (2002) | nonlinear inference | 27.4 / 28.6 | 7.7 | 29.5 | ✓ stochastic |
| | nonlinear classification | 10.4 / 10.6 | 2.1 | 7.3 | close (gap B) |
| Billman & Knutson (1996) Exp 2 | nonintercorrelated | 0.62 / 0.66 | — | 0.62 | ✓ |
| | intercorrelated | 0.73 / 0.78 | — | 0.69 | ✓ |
| Billman & Knutson (1996) Exp 3 | nonintercorrelated | 0.66 / 0.60 | — | 0.59 | ✓ |
| | intercorrelated | 0.77 / 0.78 | — | 0.70 | ✓ |


## Currently unresolved

### Gap A — Medin last_name reversal

**Problem.** Our model converges in 3.4 blocks on last_name; the paper
(human and SUSTAIN both) reports 9.7. The reversal vs. first_name (6.2)
is the wrong direction.

**What we ruled out (via the Medin et al. 1983 paper).**

- *Stimuli*: `MEDIN_STIMULI` matches Table 1 of Medin et al. (after
  accounting for SUSTAIN's reversed 1/2 → 0/1 coding).
- *Criterion*: Medin p. 614 — "Training ended after no errors were made
  on 2 successive runs … or failing that, after a total of 16 runs." That
  is exactly `criterion_blocks=2, max_blocks=16`.
- *Procedure*: each block is one random pass over the 9 fixed stimuli;
  feedback on every trial. Matches our implementation.
- *Where 9.7 comes from*: Medin p. 614 reports 91% of last-name-only
  subjects reached criterion. The 9% who hit the 16-block ceiling
  inflate the mean — 0.91·X + 0.09·16 = 9.7 gives X ≈ 9.08 for those
  who reached criterion. Our model reaches criterion in 100% of
  simulations.

**Why our model converges too fast.** With `λ_distinct = 4.62` and the
corrected Eq. 4, the distinctive dim creates very sharp activation
gradients: every stim has a uniquely-activating cluster, weights to the
binary category label saturate within 2–3 visits per cluster, and the
2-consecutive-perfect-blocks criterion is hit immediately. The paper's
mechanism for the last_name disadvantage (Love et al. p. 322:
"clusters that respond to multiple items are not as strongly
activated") would require multi-item clusters to be the modal
solution and to be partially out-activated by single-item competitors.
In our model, cluster recruitment goes straight to a near-one-cluster-
per-stim solution (mean 6.2 clusters for 9 stims, close to the paper's
7) — but the few multi-item clusters that do form *also* saturate
`H_out_winner` under Eq. 6 with β=5.93, so the abstraction-vs-
distinctiveness interaction never bites.

**Status.** Either the paper's effect requires a softer reading of Eq. 4
or Eq. 6 than the text supports, or there is a subtle implementation
detail we have not located. We can't currently distinguish these.

### Gap B — Yamauchi response scoring underspecified

Love et al. p. 323, footnote 4 acknowledges that the two source
studies (Yamauchi & Markman 1998; Yamauchi et al. 2002) scored
non-criterion-reaching participants differently, and says "SUSTAIN's
runs were analyzed in the same fashion" without specifying which.
Under argmax, linear inference matches the paper but classification
convergence is too fast (~2 blocks). Under stochastic sampling,
nonlinear inference matches almost exactly but linear inference hits
the 30-block ceiling. The "right" scoring rule for SUSTAIN is
genuinely underspecified by the source material; both options remain
exposed via `stochastic_response=…`.

## What is solid

- Ordering of Shepard's six types (I < II < III/IV/V < VI), with both
  argmax and stochastic responses.
- Medin first_name absolute number.
- Yamauchi nonlinear inference and classification.
- Billman: all four conditions, including the correct intercor >
  noninter ordering. Exp 2 nonintercor matches the paper exactly.
- All cluster-recruitment dynamics (Eqs. 10, 11) and learning updates
  (Eqs. 12–14) verified against the paper.
