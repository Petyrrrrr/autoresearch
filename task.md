# Task: max_cut (example task)

## Goal

Given an undirected, unweighted graph, partition its vertices into two sides so
that the number of edges crossing the partition (the **cut**) is as large as
possible. This is the classic **Max-Cut** problem (NP-hard in general). Your job,
as the AutoResearch agent, is to improve `train.solve` so it finds larger cuts
within the per-case time budget, across a set of random `G(n, p)` graphs.

## Input to `train.solve`

The solver is called as:

```python
solve(instance, *, rng=None, time_budget_s=None, memory_budget_mb=None)
```

where `instance` is a dict:

```python
{
    "inputs": {
        "n": int,                      # number of vertices, labelled 0..n-1
        "edges": np.ndarray,           # int64 array, shape (E, 2); each row is an edge (u, v)
    },
    "metadata": {"num_edges": int},
    "budgets": {"time_budget_s": float, "memory_budget_mb": int},
}
```

- `rng` is a seeded `numpy.random.Generator` (use it for any randomness so runs
  stay reproducible).
- The graph is simple and undirected; each edge appears once with `u < v`.
- The hidden reference cut value is **not** provided.

## Required output from `train.solve`

Return the partition assignment as a length-`n` vector of side labels:

- accepted forms: a sequence/array of `{0, 1}`, of `{-1, +1}`, or of booleans;
- it is coerced to a `{0, 1}` int array by `eval.py`;
- the length **must** equal `n`.

Anything else (wrong length, other values, `None`, non-array) is treated as an
invalid output and scored as a failure for that case.

## Data generation

Each case is an Erdos-Renyi random graph `G(n, p)` generated from an explicit
seed (`case_seed = base_seed + case_index`). A deterministic greedy 1-pass cut is
computed at prepare time and stored as the hidden `reference_cut`. See
`data/max_cut/v0/DATASET.md` for full details.

## Evaluation modes

| Mode  | Purpose            | Cases | Hard timeout | Soft budget |
|-------|--------------------|------:|-------------:|------------:|
| smoke | contract check     |     3 |         10 s |       1.0 s |
| dev   | hillclimbing       |     8 |         30 s |       3.0 s |
| final | held-out eval      |     6 |         60 s |       8.0 s |

Smoke graphs are tiny; dev spans `n = 100..300`; final uses larger, disjoint-seed
graphs (`n = 350..600`).

## Scoring

For each case:

```text
score   = cut_value / reference_cut          # higher is better; > 1.0 beats the greedy reference
success = 1 if cut_value >= reference_cut else 0
```

`reference_cut` is the deterministic greedy 1-pass cut, hidden from the solver.
Per-case metrics logged: `cut_value`, `reference_cut`, `num_edges`,
`cut_fraction`, `gap_to_reference`, `normalized_cut`.

Run-level aggregates: `aggregate_score` (sum of per-case scores), `mean_score`
(**the headline hillclimbing metric**), `median_score`, `success_rate`, plus
`mean_cut_fraction` and `num_beat_reference`.

## Invalid output handling

A case is a failure (score `0.0`) when the solver times out, raises, runs out of
memory, or returns an output that fails sanitization (wrong length, illegal
values, `None`, non-array). Evaluation continues to the remaining cases.

## Budgets

- Per-case soft time budget: `time_budget_s` (1.0 s smoke / 3.0 s dev / 8.0 s
  final) — a hint; return your best result found so far before it elapses.
- Per-case hard timeout (enforced by killing the worker): 10 s / 30 s / 60 s.
- Memory budget: 2048 MB smoke / 4096 MB dev/final (soft on Windows: reported,
  not capped; hard-capped on Unix where supported).

## Research hints

- The baseline only tries random partitions. The easiest large win is **local
  search**: repeatedly flip the vertex whose flip most increases the cut (or any
  improving flip), restarting from random starts until the time budget runs out.
- A good greedy initialization (assign each vertex to the side opposite the
  majority of its already-assigned neighbours) is cheap and strong.
- Maintain an incremental "gain" per vertex so each flip is O(degree), not O(E).
- Consider a few random restarts and keep the best; spend the time budget fully.
- Larger ideas: simulated annealing, spectral / eigenvector rounding, or
  Goemans-Williamson-style randomized rounding (numpy/scipy only).

## What the AutoResearch agent may edit

You may edit ONLY:

```text
train.py
```
