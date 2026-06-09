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

Final kept solution = loop 7: degree-seeded voting (500 restarts, 2.0σ) +
oversized 1.7K consensus pool. dev `mean_score` 0.2778 → 0.3052 (+10%);
`raw_overlap` 0.479 → 0.498; no timeouts / exceptions / invalid outputs.
