"""train.py -- the ONLY file the AutoResearch agent may edit.

It must expose:

    solve(instance, *, rng=None, time_budget_s=None, memory_budget_mb=None)

See task.md for the exact input/output contract. The baseline below is a
simple, valid, fast random-restart heuristic for Max-Cut. There is a lot of
headroom above it (greedy assignment, local search / vertex flipping, simulated
annealing, spectral / SDP-style rounding, ...). Replace the body freely as long
as you keep the `solve` signature and return a valid candidate.
"""

import time

import numpy as np


def solve(instance, *, rng=None, time_budget_s=None, memory_budget_mb=None):
    """Return a partition of the graph's vertices as a length-n array of {0, 1}.

    instance["inputs"] = {"n": int, "edges": int array of shape (E, 2)}

    Strategy (baseline): try many random partitions and keep the best cut found
    within the soft time budget. Always returns the best result found so far.
    """
    inputs = instance["inputs"]
    n = int(inputs["n"])
    edges = np.asarray(inputs["edges"])

    if rng is None:
        rng = np.random.default_rng(0)

    # Degenerate graph: any assignment is optimal (cut = 0).
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    if edges.shape[0] == 0:
        return np.zeros(n, dtype=np.int8)

    u = edges[:, 0]
    v = edges[:, 1]

    def cut_value(side):
        return int(np.count_nonzero(side[u] != side[v]))

    best = rng.integers(0, 2, size=n).astype(np.int8)
    best_cut = cut_value(best)

    deadline = None if time_budget_s is None else time.perf_counter() + float(time_budget_s)
    max_iters_when_unbudgeted = 500
    iters = 0
    while True:
        if deadline is not None:
            if time.perf_counter() >= deadline:
                break
        elif iters >= max_iters_when_unbudgeted:
            break

        candidate = rng.integers(0, 2, size=n).astype(np.int8)
        cut = cut_value(candidate)
        if cut > best_cut:
            best_cut = cut
            best = candidate
        iters += 1

    return best
