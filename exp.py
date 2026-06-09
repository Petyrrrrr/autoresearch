"""Scratch multi-variant experiment harness (NOT part of eval contract).

Loads dev cases ONCE, then evaluates any number of candidate solver callables
in-process. Faithful to bench.py / eval.py dev scoring. Screening only; confirm
winners with `python eval.py --mode dev`.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_CASES = {}


def load_cases(mode):
    if mode in _CASES:
        return _CASES[mode]
    manifest = HERE / "data" / "planted_dense_subgraph" / "v0" / mode / "manifest.tsv"
    rows = []
    with open(manifest, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    cases = []
    for r in rows:
        path = HERE / r["case_path"]
        with np.load(path) as data:
            cases.append({
                "A": data["A"].astype(np.uint8),
                "hidden": set(int(x) for x in data["hidden"].tolist()),
                "N": int(data["N"]),
                "K": int(data["K"]),
                "p": float(data["p"]),
                "q": float(data["q"]),
                "seed": int(r["seed"]),
            })
    _CASES[mode] = cases
    return cases


def score_one(pred, hidden, N, K):
    hits = len(set(pred[:K]) & hidden)
    raw = hits / K
    chance = K / N
    adj = 0.0 if chance >= 1.0 else (raw - chance) / (1.0 - chance)
    adj = max(0.0, min(1.0, adj))
    return adj, raw, int(raw >= 0.5)


def evaluate(fn, mode="dev", budget=1.0, max_cases=None, label=""):
    cases = load_cases(mode)
    if max_cases:
        cases = cases[:max_cases]
    scores, raws, succ = [], [], []
    by_k = {}
    max_rt = 0.0
    tot_rt = 0.0
    for c in cases:
        inst = {"A": c["A"], "N": c["N"], "K": c["K"], "p": c["p"], "q": c["q"]}
        rng = np.random.default_rng(c["seed"])
        t0 = time.perf_counter()
        out = fn(inst, rng=rng, time_budget_s=budget, memory_budget_mb=4096)
        rt = time.perf_counter() - t0
        tot_rt += rt
        max_rt = max(max_rt, rt)
        seen = set()
        clean = []
        for x in (int(v) for v in list(out)):
            if 0 <= x < c["N"] and x not in seen:
                seen.add(x)
                clean.append(x)
        adj, raw, s = score_one(clean, c["hidden"], c["N"], c["K"])
        scores.append(adj); raws.append(raw); succ.append(s)
        by_k.setdefault(c["K"], []).append(adj)
    n = len(scores)
    msg = (f"[{label}] mean_score={sum(scores)/n:.4f} "
           f"raw={sum(raws)/n:.4f} succ50={sum(succ)/n:.3f} "
           f"rt_mean={tot_rt/n:.3f} rt_max={max_rt:.3f}  "
           + " ".join(f"{k}:{sum(v)/len(v):.4f}" for k, v in sorted(by_k.items())))
    print(msg)
    return sum(scores) / n


_CONSENSUS = {}


def build_consensus(mode="dev", budget=1.0, frac=0.8, sigma=2.0,
                    stage1_frac=0.30, refine_iter=12, power=1.0 / 3.0,
                    max_cases=None):
    """Compute the (expensive) two-stage vote consensus ONCE per case and cache
    (Af, votes, hidden, N, K). Lets us screen many final-stage funcs cheaply."""
    import variants
    key = (mode, budget, frac, sigma, stage1_frac, refine_iter, power, max_cases)
    if key in _CONSENSUS:
        return _CONSENSUS[key]
    cases = load_cases(mode)
    if max_cases:
        cases = cases[:max_cases]
    out = []
    for c in cases:
        A = c["A"]
        N, K = c["N"], c["K"]
        deg = A.sum(axis=1, dtype=np.int64)
        rng = np.random.default_rng(c["seed"])
        deadline = time.perf_counter() + frac * budget
        votes = variants._consensus_votes(A, deg, N, K, rng, deadline, sigma,
                                          stage1_frac, refine_iter, power)
        out.append({"Af": A.astype(np.float64), "deg": deg.astype(np.float64),
                    "votes": votes, "hidden": c["hidden"], "N": N, "K": K})
    _CONSENSUS[key] = out
    return out


def screen_final(final_fn, consensus, label=""):
    """final_fn(ctx) -> ranking array/list. ctx has Af, deg, votes, N, K."""
    scores = []
    by_k = {}
    for ctx in consensus:
        ranking = final_fn(ctx)
        adj, raw, s = score_one(list(ranking), ctx["hidden"], ctx["N"], ctx["K"])
        scores.append(adj)
        by_k.setdefault(ctx["K"], []).append(adj)
    n = len(scores)
    print(f"[{label}] mean={sum(scores)/n:.4f}  "
          + " ".join(f"{k}:{sum(v)/len(v):.4f}" for k, v in sorted(by_k.items())))
    return sum(scores) / n


def oracle_ceiling(mode="dev", max_cases=None):
    """Rank vertices by affinity to the TRUE hidden set S (one-step MLE)."""
    cases = load_cases(mode)
    if max_cases:
        cases = cases[:max_cases]
    scores = []
    by_k = {}
    for c in cases:
        A = c["A"].astype(np.float64)
        N, K = c["N"], c["K"]
        w = np.zeros(N)
        for v in c["hidden"]:
            w[v] = 1.0
        aff = A @ w
        order = np.argsort(-aff, kind="stable")
        adj, raw, s = score_one(order.tolist(), c["hidden"], N, K)
        scores.append(adj)
        by_k.setdefault(K, []).append(adj)
    n = len(scores)
    print(f"[ORACLE 1-step] mean_score={sum(scores)/n:.4f}  "
          + " ".join(f"{k}:{sum(v)/len(v):.4f}" for k, v in sorted(by_k.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--max-cases", type=int, default=None)
    args = ap.parse_args()
    if args.oracle:
        oracle_ceiling("dev", args.max_cases)
