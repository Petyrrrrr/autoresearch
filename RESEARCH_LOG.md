# Research log: planted_dense_subgraph

AutoResearch agent iterations (editing only `train.py`). Headline metric is dev
`mean_score`; each change is kept only if it improves it. Baseline = 0.2778.

| # | Hypothesis | dev mean_score | Comment |
|---|------------|---------------:|---------|
| 1 | Baseline: degree-seed + projected refinement | 0.2778 | Establish reference; ~3.5 ms/case. |
| 2 | Centered spectral power-iteration seed | 0.2038 | Revert. Planted eigenvalue below the noise bulk (sub-threshold) → noisy seed. |
| 3 | Multi-restart voting / bagging (0.75σ) | 0.2831 | Keep. Small gain; restarts mostly hit the same basin. |
| 4 | Stronger restart perturbation (2.0σ) | 0.2868 | Keep. More basin diversity helps slightly. |
| 5 | Degree-corrected affinity | 0.2007 | Revert. Degree IS the PDS signal; correcting it removes what we want. |
| 6 | Oversized final pool (K' = 1.4K) | 0.2964 | Keep. Biggest cheap win; borderline members reinforce each other. |
| 7 | Pool-factor sweep → 1.7 (peak) | 0.3052 | Keep. Best config; 1.7 beats 1.4/2.0/2.4/2.8. |
| 8 | Drop voting (pool only) | 0.2931 | Revert. Voting earns +0.012, so keep it (per the metric). |
| 9 | Raise restart cap 500 → 2000 | 0.3071 | Revert. +0.002 for 2× runtime to the budget ceiling; not worth it. |
| 10 | Expand-then-contract back to size K | 0.2865 | Revert. Tight K-core loses the oversized-pool benefit. |
| 11 | Soft vote-weighted one-step final ranking | **0.3300** | **Keep.** Biggest win yet (+0.025). See note below. |

## Iteration 11 — soft consensus ranking (kept, dev 0.3300)

**Diagnostic that drove it.** Measured the ceiling of this method family:
ranking by affinity to the *true* S one step → mean adj **0.486**; but iterating
that to a densest-K-subgraph fixed point *degrades* to 0.440. The hard
fixed-point **overfits noise edges** and drifts off the true set. The old code
iterated both the per-restart refine and the final pool to convergence.

**Change.** Replaced the hard oversized-pool fixed point + edge-count ranking
with a *soft* one-step ranking:
- keep the multi-restart binary-refine voting (degree seed + 2σ noise, until budget);
- final weights `w = (votes / votes.max()) ** (1/3)` over ALL vertices (concave
  so borderline planted members that occasionally win votes still contribute);
- rank every vertex by weighted edge mass `A @ w` (single pass, no extra
  fixed-point iteration → no overfitting).

**Screening (in-process bench, dev, budget 1.0).** power sweep on `A@w`:
power 1.0 → 0.323, 0.5 → 0.328, 1/3 → **0.330**, 0.25 ≈ flat; hard pool
fixed-point ranking → 0.314; current (old) → 0.306. Pool-size factor barely
mattered once weighting was concave, so dropped the cut entirely (simpler).
Runtime unchanged (~0.59 s/case), no failures, success_rate 0.55 → 0.58.

| # | Hypothesis | dev mean_score | Comment |
|---|------------|---------------:|---------|
| 12a | Soft-vote power tuning (0.25 / 0.40) | 0.330 | No change. power=1/3 already at the flat optimum. |
| 12b | Quality-weighted votes (by set density) | 0.330 | No gain (+0.0008, noise). Densest set ≠ closest to S (MLE overfits). |
| 12c | Iterate soft message passing 2 rounds | 0.321 | Revert. Extra propagation re-introduces the over-iteration overfit. |
| 13 | Shallow refine (depth 2-4) + bigger σ | 0.324-0.331 | Revert. Deeper refine → better individual basins; depth 6/12 tied ~0.330. |
| 14/15 | **Two-stage: re-seed restarts from stage-1 consensus** | **0.3318** | **Keep.** +0.0018, success_rate 0.58→0.61, biggest gains at low K. |
| 14x | Hard one-step re-rank from consensus top-K | 0.300 | Revert. Collapses into the overfit densest basin (confirms 11's insight). |

## Headroom diagnostic (drove iters 13-15)

Compared, on dev, our output vs ceilings (mean adjusted_overlap):

| method | mean | K=112 | K=180 |
|---|---:|---:|---:|
| ours (soft vote, single stage) | 0.330 | 0.227 | 0.434 |
| hard vote top-K | 0.327 | 0.218 | 0.434 |
| **best single restart (oracle-picked basin)** | **0.376** | 0.279 | 0.469 |
| oracle one-step (rank by affinity to TRUE S) | 0.486 | 0.421 | 0.570 |

Restarts *do* reach basins +0.046 better than the consensus extracts, but the
**densest** basin is not the one closest to S (MLE overfits noise), so density/
likelihood weighting can't select it (iter 12b confirmed). Two-stage seeding
(iter 14/15) recovers a little of this gap at low K by starting stage-2 from a
better region. The remaining gap to the oracle is largely intrinsic to
sub-spectral-threshold PDS (planted eigenvalue K(p-q)≈6-9 sits under the noise
bulk edge ~18, so no global spectral method helps — confirmed in iter 2).

## Final state

`train.py` = degree fallback → **two-stage** multi-restart binary-refine voting
(stage 1 seeded from degree, stage 2 re-seeded from the stage-1 consensus
affinity) → concave (cube-root) soft vote weighting → **one-shot** weighted
edge-mass ranking `A @ w`. Dev `mean_score` **0.2778 → 0.3318** (+19%),
success_rate 0.47 → 0.61, ~0.8 s/case (hard cap 1.5 s), zero failures.

**Held-out `final` (350 cases, untuned): mean_score 0.3392**, success_rate_raw_50
0.569, zero failures, ~0.80 s/case. Per-K mean_score: 96→0.174, 112→0.232,
128→0.291, 144→0.340, 160→0.384, 180→0.425, 220→0.528. The dev gain
generalizes; difficulty scales with K exactly as the faint-signal analysis
predicts (signal ∝ K·(p−q)).

_Methodology note: screening used two scratch helpers run in-process for speed —
`bench.py` (faithful dev replica, kept) and `exp.py` (multi-variant harness,
removed). All headline numbers above are from the official `eval.py`._
