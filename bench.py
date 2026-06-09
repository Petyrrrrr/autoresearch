"""Fast in-process benchmark for train.solve (NOT part of the eval contract).

Replicates eval.py dev faithfully: same case seeds (2000 + index), same
time_budget_s, scores adjusted_overlap. Runs all cases in ONE process so it is
quicker to iterate than spawning a subprocess per case. Use only for screening;
confirm winners with `python eval.py --mode dev`.

Usage:
    python bench.py            # dev, time_budget=1.0
    python bench.py --mode dev --budget 1.0 --max-cases 100
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_cases(mode):
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
    return cases


def score(pred, hidden, N, K):
    hits = len(set(pred[:K]) & hidden)
    raw = hits / K
    chance = K / N
    adj = 0.0 if chance >= 1.0 else (raw - chance) / (1.0 - chance)
    adj = max(0.0, min(1.0, adj))
    return adj, raw, int(raw >= 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dev")
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--max-cases", type=int, default=None)
    args = ap.parse_args()

    import train

    cases = load_cases(args.mode)
    if args.max_cases:
        cases = cases[: args.max_cases]

    scores, raws, succ, runtimes = [], [], [], []
    by_k = {}
    max_rt = 0.0
    for c in cases:
        inst = {"A": c["A"], "N": c["N"], "K": c["K"], "p": c["p"], "q": c["q"]}
        rng = np.random.default_rng(c["seed"])
        t0 = time.perf_counter()
        out = train.solve(inst, rng=rng, time_budget_s=args.budget, memory_budget_mb=4096)
        rt = time.perf_counter() - t0
        pred = [int(x) for x in list(out)]
        # dedupe preserving order, in-range only
        seen = set()
        clean = []
        for x in pred:
            if 0 <= x < c["N"] and x not in seen:
                seen.add(x)
                clean.append(x)
        adj, raw, s = score(clean, c["hidden"], c["N"], c["K"])
        scores.append(adj); raws.append(raw); succ.append(s); runtimes.append(rt)
        max_rt = max(max_rt, rt)
        by_k.setdefault(c["K"], []).append(adj)

    n = len(scores)
    print(f"mode={args.mode} cases={n} budget={args.budget}")
    print(f"mean_score={sum(scores)/n:.6f}")
    print(f"mean_raw_overlap={sum(raws)/n:.6f}")
    print(f"success_rate_raw_50={sum(succ)/n:.4f}")
    print(f"mean_runtime_s={sum(runtimes)/n:.4f}  max_runtime_s={max_rt:.4f}")
    print("by_K: " + "  ".join(f"{k}:{sum(v)/len(v):.4f}" for k, v in sorted(by_k.items())))


if __name__ == "__main__":
    main()
