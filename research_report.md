# Research Report — Planted Dense Subgraph Recovery

**Task.** Recover a hidden set `S` of `K` vertices planted in an `N = 512`
vertex graph where edges inside `S` appear with `p = 0.25` and all other edges
with `q = 0.20`. The solver returns a ranked vertex list; the first `K` valid
unique vertices form the prediction `P`. Per-case score is the **chance-adjusted
overlap**

```
raw      = |P ∩ S| / K
chance   = K / N
score    = clip((raw − chance) / (1 − chance), 0, 1)     # random ≈ 0, perfect = 1
```

Headline metric is dev `mean_score`. Budget: 1.0 s soft / 1.5 s hard per case,
`N = 512`, `K ∈ {96…220}`.

This report consolidates two research sessions (the iteration-by-iteration trail
is in `RESEARCH_LOG.md`). All headline numbers are from the official `eval.py`;
screening used the in-process replicas `bench.py` / `exp.py`.

---

## 1. The scoreboard — how good is "good"?

Dev `mean_score` (chance-adjusted; 0 = blind guess, 1 = perfect):

| Method | dev mean_score | Notes |
|---|---:|---|
| **Random guess** | **0.021** | ≈ 0 by construction (chance is subtracted out). |
| Original "baseline" (degree-seed → hard densest-K refine) | 0.278 | The shipped starting point. |
| Naive degree-only ranking | 0.289 | *Beats* the baseline — the hard refine **overfits** (see §4). |
| **Shipped method (this work)** | **0.332** | Two-stage vote consensus + soft one-step ranking. |
| Oracle one-step (rank by affinity to the **true** `S`) | 0.486 | Best possible for *any* one-step / per-vertex method. |
| Perfect recovery | 1.000 | Information-theoretically out of reach at this SNR. |

Held-out **`final`** (350 cases, untuned): **mean_score 0.339**, success_rate
(raw ≥ 0.50) **0.57**, zero failures, ~0.80 s/case.

**Interpretation.**

- We sit **67 % of the way from a blind guess to the oracle ceiling**
  `(0.332 − 0.021) / (0.486 − 0.021)`.
- We beat the original baseline by **+0.054 (+19 %)** and naive degree by
  **+0.043 (+15 %)**.
- The oracle *knows* `S`; no solver can reach it. The remaining 0.332 → 0.486
  gap is **set-estimation error**, and §6 argues most of it is intrinsic to this
  faint-signal (sub-spectral-threshold) regime.

Difficulty scales exactly with the signal `∝ K·(p−q)`, as predicted:

| K (final) | 96 | 112 | 128 | 144 | 160 | 180 | 220 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ours mean_score | 0.174 | 0.232 | 0.291 | 0.340 | 0.384 | 0.425 | 0.528 |
| oracle (dev) | – | 0.421 | 0.436 | 0.494 | 0.511 | 0.570 | – |

---

## 2. The algorithm we ship

```
1. Degree fallback (always valid).
2. Two-stage multi-restart vote consensus (≈777 restarts fit the budget):
     stage 1: refine from (degree + 2σ Gaussian noise) seeds, vote on the
              converged densest-K set;
     stage 2: re-seed restarts from the stage-1 consensus affinity A @ soft(v1),
              which sits closer to the planted block than raw degree;
     votes = v1 + v2.
   Each "refine" = iterate "re-select the K vertices with the most edges into the
   current set" to a fixed point.
3. Concave soft weighting:  w = (votes / votes.max()) ** (1/3)
   (keeps borderline-but-real members that only occasionally win a restart).
4. Final ranking: ONE step of weighted edge mass,  rank by  A @ w.
```

Three design choices carry almost all of the gain, and each is explained by the
same statistical fact (§6): **the joint MLE = densest-K-subgraph, which overfits
in finite samples, so the winning estimator regularizes by averaging instead of
maximizing.**

---

## 3. What worked (Session 1, baseline 0.278 → 0.332)

| Lever | Effect | Why it works |
|---|---:|---|
| Multi-restart **vote consensus** (vs single refine) | +0.009 | Averages many densest-K basins → approximates the membership *marginal*, denoising any single overfit basin. |
| Stronger restart perturbation (2σ) | +0.004 | More basin diversity → better averaging. |
| **Concave (cube-root) soft vote weighting** | **+0.025** (biggest) | Borderline true members that win only a few restarts still contribute; replaces the over-iterated hard densest-K fixed point that overfits. |
| **One-step** `A @ w` (no extra fixed-point iteration) | (part of above) | Iterating the final ranking to convergence drifts onto dense **noise** pockets (0.486 oracle → 0.440 if iterated). |
| **Two-stage** re-seeding from consensus | +0.002 | Stage-2 restarts start nearer `S`; biggest help at low `K`. |

Net: dev **0.278 → 0.332**, success_rate 0.47 → 0.61, ~0.8 s/case, zero failures.

---

## 4. What failed (Session 2 — 11 hypotheses, none beat 0.332)

Session 2 attacked **every stage** of the pipeline. The result was a remarkably
robust plateau: nothing moved the needle beyond ±0.001 (noise).

| Stage | Hypothesis | Result | Lesson |
|---|---|---:|---|
| Budget | Spend 0.9–1.0× budget (more restarts) | 0.331 | **Saturated** at ~777 restarts; time is not the bottleneck. |
| Final | Soft-weight power sweep (0.2…1.0) | ≤0.331 | 1/3 already at the flat optimum. |
| Final | Damped soft EM refinement of the consensus | 0.332 (+0.001) | Within noise; not worth the complexity. |
| Final | Degree shrinkage / votes-only | ≤0.331 | No gain. |
| Final | **Modularity / sym-norm** degree-correction | 0.32 → 0.27 | **Degree IS the signal**; correcting it removes signal. |
| Final | Rank-averaging, two-step `A²` blend | ≤0.331 | Redundant / overfits. |
| Refine | Basin diversity (σ = 2/3/4) | ≤0.331 | Diversity already saturated. |
| Refine | Modularity-corrected refine (penalize hubs) | 0.328↓ | Same degree-is-signal lesson. |
| Voting | **Held-out CV basin weighting** (refine on one edge-fold, score on the other) | 0.31 | Splitting halves the signal (−0.017); held-out density does **not** identify `S`-closer basins. |
| Voting | **Stochastic posterior sampler** (Gumbel-top-K ∝ `exp(β·affinity)`, sweep β, proper burn-in) | 0.318→0.331 as β→greedy | **Mode-seeking beats temperature-sampling here**; the cube-root reweight already extracts the soft boundary better than noisy MCMC. |
| Voting | Pool greedy + stochastic marginals | ≤0.331 | Marginals are redundant. |

**Bonus finding.** Naive degree-ranking (0.289) beats the *original* baseline's
hard refine (0.278): iterating to a hard densest-K fixed point overfits noise and
falls below the raw degree signal. This is the same overfitting that the shipped
method's soft, one-step design avoids.

---

## 5. Why we are near the ceiling (the theory)

For a fixed-size candidate set `T` (`|T| = K`), the planted-model
log-likelihood is **linear in the number of internal edges** with coefficient

```
β* = log( p(1−q) / (q(1−p)) ) ≈ 0.288  > 0,
```

so **maximum likelihood = densest-K-subgraph**. Two consequences explain
*everything* we observed:

1. **Per-vertex ranking is solved.** Given the true `S`, the membership
   sufficient statistic is "edges into `S`", i.e. `A @ 1_S`. That is exactly the
   oracle one-step (0.486) — no higher-order feature (triangles, paths, spectral)
   adds information. This is why every final-stage tweak is saturated and why
   degree-correction only hurts.

2. **Set estimation is the hard part, and the MLE overfits.** The empirical
   densest-K-subgraph has *more* internal edges than the planted `S`
   (you are maximizing over exponentially many subsets), so it drifts off `S`.
   The Bayes-optimal fix is the posterior **marginal** `P(i ∈ S | A)` — averaging,
   not maximizing. Our vote consensus + cube-root is a (good) marginal estimator.
   Interestingly, an *honest* posterior sampler at β* did **worse** (§4): in this
   regime the posterior is dominated by the densest configs, and greedy
   mode-voting + concave reweighting estimates the marginal more cleanly than
   noisy sampling within the budget.

**Spectral is fundamentally out.** The planted signal eigenvalue is
`K(p−q) ≈ 6–9`, far below the noise bulk `≈ 2√(N·q(1−q)) ≈ 18` — the BBP
threshold is not crossed, so no global spectral method (centered adjacency,
Bethe-Hessian in this dense regime) helps. Local greedy + averaging is the
correct tool, and we have pushed it to its plateau.

---

## 6. Final state, reproducibility, recommendations

- **Shipped:** `train.py` unchanged at sha `d2ade4dd`. Official re-confirmation
  this session: dev **0.3321** / smoke **0.439** / final **0.339**, success_rate
  0.60, **zero** timeouts/exceptions/invalid/oom, 0.80 s/case (well under the
  1.5 s hard cap), ~40 MB peak RSS.
- **Reproduce:** `python eval.py --mode dev` (authoritative);
  `python bench.py` (fast in-process replica, ~0.331);
  `python exp.py --oracle` (oracle ceiling, 0.486).
- **Decision rule applied:** every Session-2 variant was reverted because none
  improved dev `mean_score` beyond noise (per the hill-climb protocol; simpler is
  better). Scratch screening scripts were removed; `bench.py` kept.

**If someone wants to push further** (all higher-risk, likely small):
1. A **proper SDP / Burer-Monteiro** relaxation of densest-K — the one class not
   yet tried; theory suggests it is also near threshold here, but it is the most
   principled remaining lever.
2. **K-adaptive** aggregation tuned per signal level (low-`K` cases carry the
   biggest oracle gap, 0.23 vs 0.42).
3. Accept the plateau: the evidence (11 independent negative results + the MLE
   analysis) strongly indicates ~0.33–0.34 is the practical frontier for this
   estimator family at `p = 0.25, q = 0.20`.
