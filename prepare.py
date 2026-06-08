"""prepare.py -- deterministic data generation for the AutoResearch task.

This file has two parts:

  1. A GENERIC harness (CLI, manifest writing, dataset description). You should
     not need to touch it when creating a new task instance.
  2. A TASK-SPECIFIC section (clearly marked below) that defines the modes and
     how to generate one case.

The AutoResearch agent NEVER edits this file. The instance-building agent edits
ONLY the task-specific section.

Usage:
    python prepare.py --mode smoke
    python prepare.py --mode dev
    python prepare.py --mode final
    python prepare.py --mode all
    python prepare.py --force
    python prepare.py --describe
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


# ===========================================================================
# TASK-SPECIFIC SECTION  --  EDIT THIS WHEN CREATING A NEW TASK INSTANCE
# ===========================================================================
#
# Fill in:
#   TASK_NAME, TASK_VERSION
#   MODES            -- per-mode list of case parameters + a base seed
#   generate_case    -- build one deterministic case from (seed, params)
#   describe_dataset -- the human-readable DATASET.md text
#
# Everything below the next banner is generic and should not need changes.

TASK_NAME = "max_cut"
TASK_VERSION = "v0"

# Seed policy (instruction recommendation): case seed = base_seed + case_index.
# Modes must be progressively harder; final uses disjoint seeds from dev.
MODES = {
    "smoke": {
        "base_seed": 1000,
        "cases": [
            {"n": 30, "p": 0.5},
            {"n": 50, "p": 0.3},
            {"n": 40, "p": 0.5},
        ],
    },
    "dev": {
        "base_seed": 2000,
        "cases": [
            {"n": 100, "p": 0.20},
            {"n": 150, "p": 0.15},
            {"n": 200, "p": 0.10},
            {"n": 250, "p": 0.10},
            {"n": 300, "p": 0.08},
            {"n": 120, "p": 0.25},
            {"n": 180, "p": 0.12},
            {"n": 220, "p": 0.10},
        ],
    },
    "final": {
        "base_seed": 3000,
        "cases": [
            {"n": 350, "p": 0.08},
            {"n": 400, "p": 0.05},
            {"n": 450, "p": 0.06},
            {"n": 500, "p": 0.05},
            {"n": 550, "p": 0.05},
            {"n": 600, "p": 0.04},
        ],
    },
}


def _greedy_reference_cut(n, u, v):
    """Deterministic greedy 1-pass Max-Cut, used as the hidden reference value.

    Vertices are processed in index order; each vertex is placed on the side
    that maximizes the number of cut edges to already-placed neighbours. This
    is cheap, deterministic, and a sensible "par" score for normalization.
    """
    adj = [[] for _ in range(n)]
    for a, b in zip(u.tolist(), v.tolist()):
        adj[a].append(b)
        adj[b].append(a)
    side = np.full(n, -1, dtype=np.int8)
    for i in range(n):
        c0 = 0
        c1 = 0
        for j in adj[i]:
            s = side[j]
            if s == 0:
                c0 += 1
            elif s == 1:
                c1 += 1
        # Put i opposite the heavier already-placed side to maximize the cut.
        side[i] = 1 if c0 >= c1 else 0
    return int(np.sum(side[u] != side[v])) if u.shape[0] > 0 else 0


def generate_case(seed, params):
    """Build one deterministic case.

    Returns a dict of arrays/scalars that ``save_case`` will serialize. Anything
    placed here that is an "answer" (e.g. the reference value) must be stripped
    out by ``Task.make_instance_for_train`` in eval.py so the solver never sees
    it.
    """
    rng = np.random.default_rng(seed)
    n = int(params["n"])
    p = float(params["p"])

    # Erdos-Renyi G(n, p): sample the upper triangle, no networkx required.
    iu, iv = np.triu_indices(n, k=1)
    keep = rng.random(iu.shape[0]) < p
    u = iu[keep].astype(np.int64)
    v = iv[keep].astype(np.int64)

    reference_cut = _greedy_reference_cut(n, u, v)
    return {
        "n": np.int64(n),
        "edges_u": u,
        "edges_v": v,
        "reference_cut": np.int64(reference_cut),
        # Stored for the manifest/report only; not exposed to train.solve.
        "_params": {"n": n, "p": p},
        "_input_size": {"vertices": n, "edges": int(u.shape[0])},
    }


def runtime_class(params):
    """Coarse hint logged in the manifest (purely descriptive)."""
    n = int(params["n"])
    if n <= 60:
        return "fast"
    if n <= 350:
        return "medium"
    return "slow"


def describe_dataset():
    """Human-readable DATASET.md body."""
    lines = []
    lines.append(f"# Dataset: {TASK_NAME} / {TASK_VERSION}")
    lines.append("")
    lines.append("## Task")
    lines.append("")
    lines.append(
        "Unweighted **Max-Cut** on Erdos-Renyi random graphs `G(n, p)`. Each case "
        "is a graph; the solver must partition the vertices into two sides to "
        "maximize the number of edges crossing the partition (the *cut*)."
    )
    lines.append("")
    lines.append("## How cases are generated")
    lines.append("")
    lines.append(
        "For each case we seed a `numpy.random.default_rng(seed)`, sample the "
        "upper triangle of the adjacency matrix (an edge `(i, j)` exists with "
        "probability `p`), and store the resulting edge list. A deterministic "
        "greedy 1-pass cut is computed and stored as the hidden `reference_cut` "
        "used for scoring normalization. The reference is **not** passed to the "
        "solver."
    )
    lines.append("")
    lines.append("## Seed policy")
    lines.append("")
    lines.append("`case_seed = base_seed + case_index`, with base seeds:")
    lines.append("")
    for mode, cfg in MODES.items():
        lines.append(f"- `{mode}`: base_seed = {cfg['base_seed']}, {len(cfg['cases'])} cases")
    lines.append("")
    lines.append("## Modes")
    lines.append("")
    lines.append("| Mode | Cases | n range | p range |")
    lines.append("|---|---:|---|---|")
    for mode, cfg in MODES.items():
        ns = [c["n"] for c in cfg["cases"]]
        ps = [c["p"] for c in cfg["cases"]]
        lines.append(
            f"| {mode} | {len(cfg['cases'])} | {min(ns)}-{max(ns)} | {min(ps)}-{max(ps)} |"
        )
    lines.append("")
    lines.append("## Files produced")
    lines.append("")
    lines.append("```text")
    lines.append(f"data/{TASK_NAME}/{TASK_VERSION}/<mode>/<case_id>.npz   # one case each")
    lines.append(f"data/{TASK_NAME}/{TASK_VERSION}/<mode>/manifest.tsv    # case index")
    lines.append(f"data/{TASK_NAME}/{TASK_VERSION}/DATASET.md             # this file")
    lines.append("```")
    lines.append("")
    lines.append("Each `.npz` holds: `n`, `edges_u`, `edges_v`, `reference_cut`.")
    lines.append("")
    lines.append("## Regenerating")
    lines.append("")
    lines.append("```bash")
    lines.append("python prepare.py --mode all --force")
    lines.append("```")
    lines.append("")
    lines.append("## Known runtime concerns")
    lines.append("")
    lines.append(
        "Graphs are dense for small `n` and sparse for large `n`, keeping edge "
        "counts (and per-case evaluation time) modest. Generation is O(n^2) in "
        "the number of candidate edges; all sizes here generate in well under a "
        "second."
    )
    lines.append("")
    return "\n".join(lines)


# ===========================================================================
# GENERIC HARNESS  --  DO NOT EDIT (shared infrastructure)
# ===========================================================================

MANIFEST_COLUMNS = [
    "case_id",
    "mode",
    "seed",
    "case_path",
    "params_json",
    "input_size_json",
    "expected_runtime_class",
    "notes",
]

FROZEN_FILES = ["prepare.py", "eval.py", "task.md", "program.md"]


def data_root():
    return HERE / "data" / TASK_NAME / TASK_VERSION


def save_case(case_path, case):
    """Serialize the arrays in a case dict, dropping private (``_``) keys."""
    payload = {k: v for k, v in case.items() if not k.startswith("_")}
    np.savez(case_path, **payload)


def prepare_mode(mode, force=False):
    if mode not in MODES:
        raise SystemExit(f"unknown mode '{mode}'; choices: {list(MODES)}")
    cfg = MODES[mode]
    out_dir = data_root() / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.tsv"

    if manifest_path.exists() and not force:
        print(f"[prepare] mode={mode} already prepared ({manifest_path}); use --force to regenerate")
        return

    rows = []
    for idx, params in enumerate(cfg["cases"]):
        seed = cfg["base_seed"] + idx
        case_id = f"{mode}_{idx:03d}"
        case = generate_case(seed, params)
        case_path = out_dir / f"{case_id}.npz"
        save_case(case_path, case)
        rows.append(
            {
                "case_id": case_id,
                "mode": mode,
                "seed": seed,
                "case_path": case_path.relative_to(HERE).as_posix(),
                "params_json": json.dumps(case.get("_params", params), separators=(",", ":")),
                "input_size_json": json.dumps(case.get("_input_size", {}), separators=(",", ":")),
                "expected_runtime_class": runtime_class(params),
                "notes": "G(n,p) random graph; reference=greedy 1-pass cut",
            }
        )

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[prepare] mode={mode}: wrote {len(rows)} cases -> {out_dir}")


def write_dataset_md():
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "DATASET.md").write_text(describe_dataset(), encoding="utf-8")


def write_frozen_hashes():
    """Record hashes of frozen files so eval.py can warn if they later change."""
    art = HERE / ".artifacts"
    art.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in FROZEN_FILES:
        path = HERE / name
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (art / "frozen_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8"
    )


def describe():
    print(f"task_name={TASK_NAME}")
    print(f"task_version={TASK_VERSION}")
    print(f"data_root={data_root().relative_to(HERE).as_posix()}")
    print("modes:")
    for mode, cfg in MODES.items():
        print(f"  {mode}: {len(cfg['cases'])} cases, base_seed={cfg['base_seed']}")
    print("seed policy: case_seed = base_seed + case_index")
    print("manifest columns: " + ", ".join(MANIFEST_COLUMNS))


def main():
    parser = argparse.ArgumentParser(description=f"Prepare data for task '{TASK_NAME}'")
    parser.add_argument("--mode", choices=list(MODES) + ["all"], default=None)
    parser.add_argument("--force", action="store_true", help="regenerate even if present")
    parser.add_argument("--describe", action="store_true", help="print task/data summary and exit")
    args = parser.parse_args()

    if args.describe:
        describe()
        return

    modes = list(MODES) if (args.mode in (None, "all")) else [args.mode]
    for mode in modes:
        prepare_mode(mode, force=args.force)
    write_dataset_md()
    write_frozen_hashes()
    print(f"[prepare] done. dataset description: {(data_root() / 'DATASET.md').relative_to(HERE).as_posix()}")


if __name__ == "__main__":
    main()
