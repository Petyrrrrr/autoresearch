# Task: planted_dense_subgraph

## Goal

Recover a hidden **planted dense subgraph**. Each case is an undirected, simple
graph on `N = 512` vertices in which a hidden set `S` of size `K` has been
planted: edges *inside* `S` appear with probability `p = 0.25`, while every other
edge appears with the lower probability `q = 0.20`. Your job, as the AutoResearch
agent, is to improve `train.solve` so it ranks the vertices of `S` ahead of the
rest, recovering as much of `S` as possible within the per-case time budget.

The planted signal is faint by design: an inside vertex has only ~`(K-1)*(p-q)`
extra expected degree, comparable to the degree standard deviation, so naive
degree ranking only gets part way.

## Input to `train.solve`

The solver is called as:

```python
solve(instance, *, rng=None, time_budget_s=None, memory_budget_mb=None)
```

where `instance` is a dict with the inputs at the top level:

```python
{
    "A": np.ndarray,   # shape (N, N), dtype uint8, symmetric, zero diagonal
    "N": int,          # number of vertices (512)
    "K": int,          # size of the hidden planted set S
    "p": float,        # edge probability inside S (0.25)
    "q": float,        # background edge probability (0.20)
    "budgets": {"time_budget_s": float, "memory_budget_mb": int},  # added by engine
}
```

- `rng` is a seeded `numpy.random.Generator` (use it for any randomness so runs
  stay reproducible).
- `time_budget_s` / `memory_budget_mb` are the soft budgets, also passed as
  keyword arguments.
- The hidden set `S` is **not** provided.

## Required output from `train.solve`

Return an **iterable of vertex indices**, ranked best-first:

- accepted forms: a list / tuple / 1-D `np.ndarray` / range of integer indices;
- the evaluator coerces them to ints, drops out-of-range and duplicate indices
  (keeping first occurrence), and keeps the **first `K` valid unique** vertices
  as your prediction `P`;
- returning a full ranking of all `N` vertices is fine — only the first `K`
  unique valid ones are scored.

`None`, a non-iterable, non-finite values, or an output with no valid index in
`[0, N)` is treated as an invalid output and scored as a failure for that case.

## Data generation

A hidden set `S` of size `K` is drawn uniformly at random. For each unordered
pair `(i, j)`, an edge is sampled with probability `p` if both `i` and `j` are in
`S`, otherwise with probability `q`. The matrix is symmetric with zero diagonal.
Everything is deterministic from an explicit seed (`case_seed = base_seed +
case_index`). The planted set is stored only for scoring. See
`data/planted_dense_subgraph/v0/DATASET.md` for full details.

## Evaluation modes

| Mode  | Purpose        | K values                       | Trials/K | Cases | Hard timeout | Soft budget |
|-------|----------------|--------------------------------|---------:|------:|-------------:|------------:|
| smoke | contract check | 180, 220                       |        4 |     8 |        1.0 s |       0.8 s |
| dev   | hillclimbing   | 112, 128, 144, 160, 180        |       20 |   100 |        1.5 s |       1.0 s |
| final | held-out eval  | 96, 112, 128, 144, 160, 180, 220 |     50 |   350 |        1.5 s |       1.0 s |

All modes use `N = 512`, `p = 0.25`, `q = 0.20`. Base seeds: smoke = 1000,
dev = 2000, final = 3000 (disjoint).

## Scoring

For each case, with `P` = first `K` valid unique returned vertices and `S` the
hidden set:

```text
hits             = |P n S|
raw_overlap      = hits / K
chance_overlap   = K / N
adjusted_overlap = clip((raw_overlap - chance_overlap) / (1 - chance_overlap), 0, 1)
case_score       = adjusted_overlap        # random guessing ~ 0, perfect = 1
success          = 1 if raw_overlap >= 0.50 else 0
```

Per-case metrics logged: `K`, `hits`, `raw_overlap`, `chance_overlap`,
`adjusted_overlap`, `success_raw_50`, `success_raw_75`.

Run-level aggregates: `mean_score` (**the headline hillclimbing metric**),
`aggregate_score` (sum of per-case scores), `median_score`, `success_rate`. The
`extra_metrics_json` column adds `score_x1000` (= `1000 * mean_score`, the
instruction's headline number), `mean_raw_overlap`, `mean_adjusted_overlap`,
`success_rate_raw_50`, `success_rate_raw_75`, and per-K breakdowns
(`mean_score_by_K`, `mean_raw_overlap_by_K`, `mean_adjusted_overlap_by_K`).

## Invalid output handling

A case is a failure (score `0.0`) when the solver times out, raises, runs out of
memory, or returns an output that fails sanitization (`None`, non-iterable,
non-finite, or no valid index in `[0, N)`). Evaluation continues to the
remaining cases.

## Budgets

- Per-case soft time budget: `time_budget_s` (0.8 s smoke / 1.0 s dev / final) —
  a hint; return your best ranking found so far before it elapses.
- Per-case hard timeout (enforced by killing the worker): 1.0 s smoke / 1.5 s
  dev / final. The hard cap covers the whole worker (interpreter start + loading
  the adjacency + `solve`), so leave margin under the soft budget.
- Memory budget: 2048 MB smoke / 4096 MB dev/final (soft on Windows: reported,
  not capped; hard-capped on Unix where supported).

## Research hints

- The baseline seeds the candidate set with the top-`K` highest-degree vertices,
  then iteratively re-selects the `K` vertices with the most edges into the
  current set. It is cheap and converges fast, leaving plenty of headroom.
- **Centered spectral methods**: the leading eigenvector of the centered
  adjacency `A - q` (or `A - mean(A)`) concentrates on the planted block — a few
  power iterations give a strong ranking.
- **Iterative refinement / message passing**: weight each vertex by edges into
  the current set, optionally subtracting the `q*|S|` background expectation
  (likelihood-inspired scoring) instead of using a raw count.
- **Local swaps**: after refinement, swap weak in-members for strong outsiders
  while it increases internal density.
- **Multi-candidate voting**: run refinement from several random seeds /
  restarts and vote, keeping the time budget fully used.

## What the AutoResearch agent may edit

You may edit ONLY:

```text
train.py
```
