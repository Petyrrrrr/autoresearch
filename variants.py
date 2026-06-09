"""Candidate solver variants for screening (scratch; not the eval contract).

Each `make_*` returns a callable with the train.solve signature so exp.py can
score it. The winner gets promoted into train.py.
"""
import time
import numpy as np


def _deadline_fn(time_budget_s, frac):
    if time_budget_s is None:
        return None
    return time.perf_counter() + frac * float(time_budget_s)


def _refine(A, seed_scores, N, K, max_iter=12):
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


def _soft_weights(v, power):
    return (v / v.max()) ** power if v.max() > 0 else v


def _refine_modularity(A, seed_scores, N, K, deg, beta, max_iter=12):
    """Refine but re-select by modularity-corrected affinity (penalize hubs)."""
    order = np.argsort(-seed_scores, kind="stable")
    S = np.zeros(N, dtype=bool)
    S[order[:K]] = True
    deg_f = deg.astype(np.float64)
    total = deg_f.sum() + 1e-9
    for _ in range(max_iter):
        affinity = A[:, S].sum(axis=1, dtype=np.float64)
        score = affinity - beta * deg_f * (deg_f[S].sum() / total)
        top = np.argsort(-score, kind="stable")
        new_S = np.zeros(N, dtype=bool)
        new_S[top[:K]] = True
        if np.array_equal(new_S, S):
            break
        S = new_S
    return S


def _consensus_votes(A, deg, N, K, rng, deadline, sigma, stage1_frac,
                     refine_iter, power, refine_fn=None):
    """Two-stage multi-restart binary-refine voting; returns vote vector.
    refine_fn(A, seed_scores, N, K) -> bool mask; defaults to _refine."""
    if refine_fn is None:
        refine_fn = lambda A, s, N, K: _refine(A, s, N, K, refine_iter)

    def vote_pass(seed_base, pass_deadline):
        scale = (float(seed_base.std()) + 1e-9) * sigma
        v = np.zeros(N, dtype=np.float64)
        v[refine_fn(A, seed_base, N, K)] += 1.0
        n = 1
        while n < 5000 and (pass_deadline is None or time.perf_counter() < pass_deadline):
            v[refine_fn(A, seed_base + scale * rng.standard_normal(N), N, K)] += 1.0
            n += 1
            if pass_deadline is None:
                break
        return v

    deg_f = deg.astype(np.float64)
    now = time.perf_counter()
    if deadline is None:
        return vote_pass(deg_f, now + 0.05)
    mid = now + stage1_frac * (deadline - now)
    v1 = vote_pass(deg_f, mid)
    aff1 = A @ _soft_weights(v1, power)
    v2 = vote_pass(aff1, deadline)
    return v1 + v2


def make_em(frac=0.8, power=1.0 / 3.0, sigma=2.0, stage1_frac=0.30,
            refine_iter=12, em_iters=6, tau=4.0, damp=0.5):
    """Two-stage voting consensus, then DAMPED soft EM final stage."""
    def solve(instance, *, rng=None, time_budget_s=None, memory_budget_mb=None):
        A = np.asarray(instance["A"])
        N = int(instance.get("N", A.shape[0]))
        K = int(instance["K"])
        if rng is None:
            rng = np.random.default_rng(0)
        deg = A.sum(axis=1, dtype=np.int64)
        if K <= 0 or K >= N:
            return np.argsort(-deg, kind="stable").tolist()
        deadline = _deadline_fn(time_budget_s, frac)
        votes = _consensus_votes(A, deg, N, K, rng, deadline, sigma,
                                 stage1_frac, refine_iter, power)
        # init soft membership from concave consensus weights
        pi = _soft_weights(votes, power)
        Af = A.astype(np.float64)
        for _ in range(em_iters):
            score = Af @ pi
            thr = np.partition(score, N - K)[N - K]
            new_pi = 1.0 / (1.0 + np.exp(-(score - thr) / tau))
            pi = (1.0 - damp) * pi + damp * new_pi
        final_score = Af @ pi
        return np.argsort(-final_score, kind="stable").tolist()

    return solve


def make_base(frac=0.8, power=1.0 / 3.0, sigma=2.0, stage1_frac=0.30,
              refine_iter=12, final_power=None):
    """The current train.py method, parameterized."""
    if final_power is None:
        final_power = power

    def solve(instance, *, rng=None, time_budget_s=None, memory_budget_mb=None):
        A = np.asarray(instance["A"])
        N = int(instance.get("N", A.shape[0]))
        K = int(instance["K"])
        if rng is None:
            rng = np.random.default_rng(0)
        deg = A.sum(axis=1, dtype=np.int64)
        degree_order = np.argsort(-deg, kind="stable")
        if K <= 0 or K >= N:
            return degree_order.tolist()
        deadline = _deadline_fn(time_budget_s, frac)

        def vote_pass(seed_base, pass_deadline):
            scale = (float(seed_base.std()) + 1e-9) * sigma
            v = np.zeros(N, dtype=np.float64)
            v[_refine(A, seed_base, N, K, refine_iter)] += 1.0
            n = 1
            while n < 5000 and (pass_deadline is None or time.perf_counter() < pass_deadline):
                v[_refine(A, seed_base + scale * rng.standard_normal(N), N, K, refine_iter)] += 1.0
                n += 1
                if pass_deadline is None:
                    break
            return v

        deg_f = deg.astype(np.float64)
        now = time.perf_counter()
        if deadline is None:
            votes = vote_pass(deg_f, now + 0.05)
        else:
            mid = now + stage1_frac * (deadline - now)
            v1 = vote_pass(deg_f, mid)
            aff1 = A @ _soft_weights(v1, power)
            v2 = vote_pass(aff1, deadline)
            votes = v1 + v2
        final_affinity = A @ _soft_weights(votes, final_power)
        ranking = np.argsort(-final_affinity, kind="stable")
        return ranking.tolist()

    return solve
