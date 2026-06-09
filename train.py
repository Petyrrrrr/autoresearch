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

    def refine(seed_scores, max_iter=12):
        """Refine a candidate set: seed with top-K of seed_scores, then
        repeatedly re-select the K vertices with the most edges into the set."""
        order = np.argsort(-seed_scores, kind="stable")
        S = np.zeros(N, dtype=bool)
        S[order[:K]] = True
        for _ in range(max_iter):
            affinity = A[:, S].sum(axis=1, dtype=np.int64)
            top = np.argsort(-affinity, kind="stable")
            new_S = np.zeros(N, dtype=bool)
            new_S[top[:K]] = True
            if np.array_equal(new_S, S):
                break
            S = new_S
        return S

    # 2/3. Multi-restart refinement with voting (consensus / bagging). Each
    #      restart perturbs the degree seed; truly-planted vertices are stable
    #      across restarts and accumulate votes, while noise vertices do not.
    deg_f = deg.astype(np.float64)
    deg_std = float(deg.std()) + 1e-9
    votes = np.zeros(N, dtype=np.float64)
    votes[refine(deg_f)] += 1.0  # anchor restart from the clean degree seed
    restarts = 1
    max_restarts = 500
    while restarts < max_restarts and not out_of_time():
        seed_scores = deg_f + 2.0 * deg_std * rng.standard_normal(N)
        votes[refine(seed_scores)] += 1.0
        restarts += 1

    # 4. final ranking: build an OVERSIZED consensus pool (K' = 1.7K) so
    #    borderline planted members stay in and reinforce each other, refine the
    #    pool at its own size, then rank ALL vertices by edges into that pool.
    Kw = min(N - 1, int(round(1.7 * K)))
    pool = np.zeros(N, dtype=bool)
    pool[np.argsort(-votes, kind="stable")[:Kw]] = True
    for _ in range(12):
        into = A[:, pool].sum(axis=1, dtype=np.int64)
        top = np.argsort(-into, kind="stable")
        new_pool = np.zeros(N, dtype=bool)
        new_pool[top[:Kw]] = True
        if np.array_equal(new_pool, pool):
            break
        pool = new_pool
    final_affinity = A[:, pool].sum(axis=1, dtype=np.int64)
    ranking = np.argsort(-final_affinity, kind="stable")
    return ranking.tolist()
