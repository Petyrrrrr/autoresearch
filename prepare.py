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

TASK_NAME = "planted_dense_subgraph"
TASK_VERSION = "v0"

# Planted Dense Subgraph (PDS) model constants (fixed across all cases/modes).
N = 512            # number of vertices
P_IN = 0.25        # edge probability inside the planted set S
Q_OUT = 0.20       # edge probability everywhere else (background)


def _expand_cases(k_values, trials_per_k):
    """One parameter dict per case: every K repeated ``trials_per_k`` times.

    Each trial gets a distinct case_index (and therefore a distinct seed), so
    repeated K values are different random graphs.
    """
    cases = []
    for k in k_values:
        for _ in range(int(trials_per_k)):
            cases.append({"N": N, "K": int(k), "p": P_IN, "q": Q_OUT})
    return cases


# Seed policy (instruction recommendation): case seed = base_seed + case_index.
# Modes use disjoint base seeds; final is larger and held-out-style.
MODES = {
    "smoke": {
        "base_seed": 1000,
        "cases": _expand_cases([180, 220], trials_per_k=4),
    },
    "dev": {
        "base_seed": 2000,
        "cases": _expand_cases([112, 128, 144, 160, 180], trials_per_k=20),
    },
    "final": {
        "base_seed": 3000,
        "cases": _expand_cases([96, 112, 128, 144, 160, 180, 220], trials_per_k=50),
    },
}


def generate_case(seed, params):
    """Build one deterministic Planted Dense Subgraph case.

    A hidden set ``S`` of size ``K`` is chosen uniformly at random. Edges inside
    ``S`` appear with probability ``p``; all other edges with probability ``q``.

    Returns a dict of arrays/scalars that ``save_case`` will serialize. The
    hidden planted set is stored for scoring only; ``Task.make_instance_for_train``
    in eval.py strips it so the solver never sees the answer.
    """
    n = int(params["N"])
    k = int(params["K"])
    p = float(params["p"])
    q = float(params["q"])
    rng = np.random.default_rng(seed)

    # Hidden planted set S (sorted for reproducibility); never shown to solver.
    hidden = rng.choice(n, size=k, replace=False)
    hidden = np.sort(hidden).astype(np.int64)

    # Sample the upper triangle: edge prob is p inside S x S, q otherwise.
    upper_i, upper_j = np.triu_indices(n, k=1)
    edge_probs = np.full(len(upper_i), q, dtype=np.float64)
    hidden_mask = np.zeros(n, dtype=bool)
    hidden_mask[hidden] = True
    inside = hidden_mask[upper_i] & hidden_mask[upper_j]
    edge_probs[inside] = p
    edges = rng.random(len(upper_i)) < edge_probs

    A = np.zeros((n, n), dtype=np.uint8)
    A[upper_i, upper_j] = edges
    A[upper_j, upper_i] = edges
    np.fill_diagonal(A, 0)

    return {
        "A": A,
        "hidden": hidden,        # hidden answer; stripped before train.solve
        "N": np.int64(n),
        "K": np.int64(k),
        "p": np.float64(p),
        "q": np.float64(q),
        # Stored for the manifest/report only; not exposed to train.solve.
        "_params": {"N": n, "K": k, "p": p, "q": q},
        "_input_size": {"N": n, "K": k, "num_edges": int(edges.sum())},
    }


def runtime_class(params):
    """Coarse hint logged in the manifest (purely descriptive).

    Every case is N=512; cost is dominated by the O(N^2) adjacency, so all
    cases land in the same class regardless of K.
    """
    return "medium"


def describe_dataset():
    """Human-readable DATASET.md body."""
    lines = []
    lines.append(f"# Dataset: {TASK_NAME} / {TASK_VERSION}")
    lines.append("")
    lines.append("## Task")
    lines.append("")
    lines.append(
        "**Planted Dense Subgraph** recovery. Each case is an undirected graph on "
        f"`N = {N}` vertices in which a hidden set `S` of size `K` has been "
        f"planted: edges inside `S` appear with probability `p = {P_IN}`, while "
        f"every other edge appears with probability `q = {Q_OUT}`. The solver "
        "receives the adjacency matrix and must return a ranked list of vertices; "
        "the evaluator keeps the first `K` valid unique vertices and scores their "
        "overlap with `S`."
    )
    lines.append("")
    lines.append("## How cases are generated")
    lines.append("")
    lines.append(
        "For each case we seed a `numpy.random.default_rng(seed)`, draw the hidden "
        "set `S` of size `K` uniformly without replacement, then sample the upper "
        "triangle of the adjacency matrix: pair `(i, j)` is an edge with "
        "probability `p` if both endpoints are in `S`, otherwise with probability "
        "`q`. The diagonal is zero and the matrix is symmetric. The hidden set `S` "
        "is stored for scoring but is **not** passed to the solver."
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
    lines.append("| Mode | Cases | N | K values | p | q |")
    lines.append("|---|---:|---:|---|---:|---:|")
    for mode, cfg in MODES.items():
        ks = sorted({c["K"] for c in cfg["cases"]})
        k_str = ", ".join(str(k) for k in ks)
        lines.append(f"| {mode} | {len(cfg['cases'])} | {N} | {k_str} | {P_IN} | {Q_OUT} |")
    lines.append("")
    lines.append("## Files produced")
    lines.append("")
    lines.append("```text")
    lines.append(f"data/{TASK_NAME}/{TASK_VERSION}/<mode>/<case_id>.npz   # one case each")
    lines.append(f"data/{TASK_NAME}/{TASK_VERSION}/<mode>/manifest.tsv    # case index")
    lines.append(f"data/{TASK_NAME}/{TASK_VERSION}/DATASET.md             # this file")
    lines.append("```")
    lines.append("")
    lines.append("Each `.npz` holds: `A` (N x N uint8 adjacency), `hidden` (the "
                 "planted set, hidden from the solver), `N`, `K`, `p`, `q`.")
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
        "Every graph is dense (N=512, edge probabilities 0.20-0.25), so each "
        "adjacency matrix is ~256 KB and generation is O(N^2) per case (a few "
        "milliseconds). The planted signal is faint: an inside vertex has only "
        "about `(K-1)*(p-q)` extra expected degree, comparable to the degree "
        "standard deviation, which is what makes recovery non-trivial."
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
                "notes": "planted dense subgraph; hidden set S stored for scoring only",
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
