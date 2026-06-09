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

    def vote_pass(seed_base, sigma, pass_deadline):
        """Accumulate consensus votes: one anchor refine from seed_base, then
        perturbed restarts (seed_base + sigma-scaled noise) until pass_deadline.
        Planted vertices recur across restarts; noise vertices do not."""
        scale = (float(seed_base.std()) + 1e-9) * sigma
        v = np.zeros(N, dtype=np.float64)
        v[refine(seed_base)] += 1.0
        n = 1
        while n < 5000 and time.perf_counter() < pass_deadline:
            v[refine(seed_base + scale * rng.standard_normal(N))] += 1.0
            n += 1
        return v

    def soft_weights(v):
        """Concave (cube-root) vote weighting so borderline planted members --
        which occasionally win votes -- still contribute to the final ranking."""
        return (v / v.max()) ** (1.0 / 3.0) if v.max() > 0 else v

    # 2/3. TWO-STAGE consensus voting. Stage 1 seeds restarts from the degree
    #      vector; stage 2 re-seeds restarts from the stage-1 consensus affinity
    #      (A @ w1), which sits closer to the planted block than raw degree, so
    #      the faint-signal (low-K) cases reach better local optima. Votes from
    #      both stages are pooled.
    deg_f = deg.astype(np.float64)
    now = time.perf_counter()
    if deadline is None:
        # No budget hint: take a single bounded pass so we always return fast.
        votes = vote_pass(deg_f, 2.0, now + 0.05)
    else:
        mid = now + 0.30 * (deadline - now)
        v1 = vote_pass(deg_f, 2.0, mid)
        aff1 = A @ soft_weights(v1)            # consensus affinity seed
        v2 = vote_pass(aff1, 2.0, deadline)
        votes = v1 + v2

    # 4. final ranking: ONE-SHOT weighted edge mass into the soft consensus set.
    #    A single pass (no extra fixed-point iteration) avoids the over-iteration
    #    that makes a hard densest-subgraph fixed point overfit noise edges
    #    (empirically the one-step ranking beats iterating to convergence).
    final_affinity = A @ soft_weights(votes)
    ranking = np.argsort(-final_affinity, kind="stable")
    return ranking.tolist()
