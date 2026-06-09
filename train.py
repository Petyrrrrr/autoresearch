"""train.py -- the ONLY file the AutoResearch agent may edit.

It must expose:

    solve(instance, *, rng=None, time_budget_s=None, memory_budget_mb=None)

See task.md for the exact input/output contract. The task is Planted Dense
Subgraph recovery: a hidden set S of size K is planted in an N-vertex graph
(edge prob p inside S, q elsewhere). Return a ranked vertex list; the evaluator
keeps the first K valid unique vertices and scores their overlap with S.

The baseline below is simple, valid, fast, and nontrivial. It seeds a candidate
set with the highest-degree vertices and refines it by repeatedly re-selecting
the K vertices with the most edges into the current set. There is a lot of
headroom above it (centered spectral power iteration, likelihood-weighted
scoring, message passing, multi-restart voting, true local swaps). Replace the
body freely as long as you keep the `solve` signature and return a ranked list.
"""

import time

import numpy as np


def solve(instance, *, rng=None, time_budget_s=None, memory_budget_mb=None):
    """Return a ranked list of vertex indices (best candidates first).

    instance = {"A": (N, N) uint8 adjacency, "N": int, "K": int,
                "p": float, "q": float}
    """
    A = np.asarray(instance["A"])
    N = int(instance.get("N", A.shape[0]))
    K = int(instance["K"])
    # q (background edge prob) is available for likelihood/centering ideas.
    q = float(instance.get("q", 0.20))

    if rng is None:
        rng = np.random.default_rng(0)

    # 1. vertex degrees -> a degree ranking (always a valid fallback).
    deg = A.sum(axis=1, dtype=np.int64)
    degree_order = np.argsort(-deg, kind="stable")
    if K <= 0 or K >= N:
        return degree_order.tolist()

    # Soft-budget deadline; leave margin so the worker never hits the hard cap.
    deadline = None
    if time_budget_s is not None:
        deadline = time.perf_counter() + 0.8 * float(time_budget_s)

    def out_of_time():
        return deadline is not None and time.perf_counter() >= deadline

    # 2. seed the candidate set with the top-K highest-degree vertices.
    S = np.zeros(N, dtype=bool)
    S[degree_order[:K]] = True

    # 3. iterative refinement: re-select the K vertices with the most edges into
    #    the current set. This is a cheap collective swap step that amplifies the
    #    planted block; it converges in a handful of passes.
    for _ in range(20):
        if out_of_time():
            break
        affinity = A[:, S].sum(axis=1, dtype=np.int64)  # edges into S per vertex
        top = np.argsort(-affinity, kind="stable")
        new_S = np.zeros(N, dtype=bool)
        new_S[top[:K]] = True
        if np.array_equal(new_S, S):
            break
        S = new_S

    # 4. rank ALL vertices by edges into the recovered set (high to low) so the
    #    first K of the returned list are the recovered planted candidates.
    final_affinity = A[:, S].sum(axis=1, dtype=np.int64)
    ranking = np.argsort(-final_affinity, kind="stable")
    return ranking.tolist()
