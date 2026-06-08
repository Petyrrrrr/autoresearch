"""eval.py -- the AutoResearch evaluation harness.

Two parts:

  1. A GENERIC framework: the ``AutoResearch`` base class plus shared helpers.
     It owns the CLI, run ids, deterministic seeding, manifest loading,
     subprocess isolation, hard timeouts, memory reporting, output-sanitization
     wrapping, failure handling, aggregation, TSV logging and the human-readable
     report. The AutoResearch agent NEVER edits any of this; the instance-builder
     should not need to either.

  2. A TASK-SPECIFIC subclass ``Task`` (clearly marked near the bottom). When
     creating a new task instance you implement five small methods there.

eval.py treats train.py as untrusted: it may raise, hang, exhaust memory, or
return garbage. Each case runs in its own worker subprocess with a hard
wall-clock timeout; failures are recorded and evaluation continues.

Usage:
    python eval.py --mode smoke
    python eval.py --mode dev
    python eval.py --mode final
    python eval.py --mode smoke --run-name my_probe
    python eval.py --mode dev --max-cases 10
    python eval.py --mode dev --timeout-s 5
    python eval.py --mode dev --memory-mb 4096
    python eval.py --mode dev --strict-frozen
    python eval.py --describe
"""

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREPARE_PY = HERE / "prepare.py"
TRAIN_PY = HERE / "train.py"
FROZEN_FILES = ["prepare.py", "eval.py", "task.md", "program.md"]


# ===========================================================================
# GENERIC HELPERS  --  DO NOT EDIT
# ===========================================================================

def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    path = Path(path)
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def short_err(exc, limit=240):
    msg = f"{type(exc).__name__}: {exc}"
    msg = msg.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return msg[:limit]


def json_compact(obj):
    try:
        return json.dumps(obj, separators=(",", ":"), default=str)
    except Exception:
        return json.dumps(str(obj))


def peak_rss_mb():
    """Best-effort peak resident memory of the current process, in MB.

    Returns None if unavailable. Works on Windows (GetProcessMemoryInfo) and
    Unix (resource.getrusage); reporting is optional everywhere else.
    """
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            get_mem = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
            get_mem.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
            get_mem.restype = wintypes.BOOL

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            if get_mem(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.PeakWorkingSetSize / (1024 * 1024)
        except Exception:
            return None
        return None
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes.
        if sys.platform == "darwin":
            return ru / (1024 * 1024)
        return ru / 1024
    except Exception:
        return None


def maybe_set_memory_limit(memory_mb):
    """Soft attempt at a hard address-space cap (Unix only). No-op elsewhere."""
    if not memory_mb:
        return
    try:
        import resource

        nbytes = int(memory_mb) * 1024 * 1024
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_hard = hard if hard != resource.RLIM_INFINITY and hard < nbytes else nbytes
        resource.setrlimit(resource.RLIMIT_AS, (nbytes, new_hard))
    except Exception:
        # Not portable (e.g. Windows). Memory is still *reported*, just not capped.
        pass


def percentile(values, pct):
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


CASES_COLUMNS = [
    "run_id", "timestamp_utc", "task_name", "task_version", "mode",
    "case_id", "case_index", "seed", "case_path", "params_json",
    "train_sha256", "status", "score", "success", "runtime_s",
    "hard_timeout_s", "soft_time_budget_s", "peak_rss_mb",
    "tracemalloc_peak_mb", "memory_budget_mb", "output_summary_json",
    "metrics_json", "error_type", "error_message",
]

SUMMARY_COLUMNS = [
    "run_id", "timestamp_utc", "task_name", "task_version", "mode",
    "num_cases", "num_ok", "num_timeouts", "num_exceptions", "num_invalid",
    "aggregate_score", "mean_score", "median_score", "success_rate",
    "mean_runtime_s", "p95_runtime_s", "max_runtime_s",
    "mean_peak_rss_mb", "max_peak_rss_mb", "train_sha256", "run_dir",
    "extra_metrics_json",
]


# ===========================================================================
# GENERIC FRAMEWORK  --  DO NOT EDIT
# ===========================================================================

class AutoResearch:
    """Reusable evaluation engine. Subclass it and implement the task hooks."""

    # These are sentinels; the concrete Task subclass below MUST override them.
    task_name = "PLACEHOLDER_TASK_NAME"
    task_version = "v0"

    # Per-mode budgets. hard_timeout_s is enforced by the parent (wall clock);
    # soft_time_budget_s is the hint handed to train.solve; memory_mb is the
    # soft (and, on Unix, hard) memory budget.
    mode_budgets = {
        "smoke": {"hard_timeout_s": 10, "soft_time_budget_s": 3.0, "memory_mb": 2048},
        "dev": {"hard_timeout_s": 30, "soft_time_budget_s": 10.0, "memory_mb": 4096},
        "final": {"hard_timeout_s": 60, "soft_time_budget_s": 20.0, "memory_mb": 4096},
    }

    # Score assigned to any case that fails (timeout/exception/invalid/oom).
    failure_score_value = 0.0

    # ----- task hooks (override these in the subclass) ---------------------
    def load_case(self, case_path):
        raise NotImplementedError

    def make_instance_for_train(self, raw_case):
        raise NotImplementedError

    def sanitize_output(self, raw_output, raw_case):
        raise NotImplementedError

    def score_case(self, sanitized_output, raw_case):
        raise NotImplementedError

    def aggregate_extra(self, case_rows):
        """Optional task-specific aggregate metrics (dict). Stored as JSON."""
        return {}

    def scoring_description(self):
        return "(scoring description not provided)"

    # ----- generic machinery (do not override) -----------------------------
    def data_root(self):
        return HERE / "data" / self.task_name / self.task_version

    def manifest_path(self, mode):
        return self.data_root() / mode / "manifest.tsv"

    def budgets_for(self, mode):
        default = {"hard_timeout_s": 30, "soft_time_budget_s": 10.0, "memory_mb": 4096}
        return dict(self.mode_budgets.get(mode, default))

    def modes(self):
        return list(self.mode_budgets)

    def ensure_data(self, mode):
        if self.manifest_path(mode).exists():
            return
        print(f"[eval] data for mode={mode} missing; running prepare.py ...")
        result = subprocess.run(
            [sys.executable, str(PREPARE_PY), "--mode", mode], cwd=str(HERE)
        )
        if result.returncode != 0 or not self.manifest_path(mode).exists():
            print(f"ERROR: could not prepare data. Run: python prepare.py --mode {mode}")
            sys.exit(1)

    def load_manifest(self, mode):
        path = self.manifest_path(mode)
        with open(path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def frozen_hash_warnings(self):
        """Compare current frozen-file hashes against those recorded at prepare."""
        record_path = HERE / ".artifacts" / "frozen_hashes.json"
        if not record_path.exists():
            return []
        try:
            recorded = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        warnings = []
        for name, old in recorded.items():
            now = sha256_file(HERE / name)
            if now and old and now != old:
                warnings.append(name)
        return warnings

    # ----- worker (child process; runs the UNTRUSTED solver) ---------------
    def run_worker(self, args):
        """Run one case in this (isolated) process and write a small JSON result.

        load_case / sanitize_output / score_case are trusted framework code and
        run here too, but the only thing allowed to crash the world is
        train.solve -- which we wrap. A hard timeout is enforced by the parent.
        """
        result = {
            "status": "infrastructure_error",
            "score": self.failure_score_value,
            "success": 0,
            "metrics": {},
            "output_summary": {},
            "runtime_s": 0.0,
            "tracemalloc_peak_mb": None,
            "peak_rss_mb": None,
            "error_type": "",
            "error_message": "",
        }
        out_path = Path(args.output_path)

        def finish(record_mem=True):
            if record_mem:
                try:
                    _, peak = tracemalloc.get_traced_memory()
                    result["tracemalloc_peak_mb"] = peak / (1024 * 1024)
                except Exception:
                    pass
            result["peak_rss_mb"] = peak_rss_mb()
            out_path.write_text(json_compact(result), encoding="utf-8")

        try:
            raw_case = self.load_case(args.case_path)
        except Exception as exc:
            result.update(status="exception", error_type="serialization_error",
                          error_message=short_err(exc))
            finish(record_mem=False)
            return

        try:
            instance = self.make_instance_for_train(raw_case)
            if isinstance(instance, dict):
                instance.setdefault("budgets", {}).update({
                    "time_budget_s": args.soft_time_budget_s,
                    "memory_budget_mb": args.memory_mb,
                })
        except Exception as exc:
            result.update(status="infrastructure_error", error_type="instance_error",
                          error_message=short_err(exc))
            finish(record_mem=False)
            return

        maybe_set_memory_limit(args.memory_mb)
        rng = np.random.default_rng(args.seed)

        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            sys.path.insert(0, str(HERE))
            import train

            if not hasattr(train, "solve"):
                raise AttributeError("train.py has no 'solve' function")
            raw_output = train.solve(
                instance,
                rng=rng,
                time_budget_s=args.soft_time_budget_s,
                memory_budget_mb=args.memory_mb,
            )
            result["runtime_s"] = time.perf_counter() - t0
        except MemoryError as exc:
            result.update(status="oom", error_type="memory_error",
                          error_message=short_err(exc),
                          runtime_s=time.perf_counter() - t0)
            finish()
            return
        except AttributeError as exc:
            result.update(status="exception", error_type="missing_solve_function",
                          error_message=short_err(exc),
                          runtime_s=time.perf_counter() - t0)
            finish()
            return
        except TypeError as exc:
            result.update(status="exception", error_type="wrong_signature",
                          error_message=short_err(exc),
                          runtime_s=time.perf_counter() - t0)
            finish()
            return
        except Exception as exc:
            result.update(status="exception", error_type="solver_exception",
                          error_message=short_err(exc),
                          runtime_s=time.perf_counter() - t0)
            finish()
            return

        # ---- sanitize (trusted) ----
        try:
            san = self.sanitize_output(raw_output, raw_case)
        except Exception as exc:
            result.update(status="invalid_output", error_type="sanitize_error",
                          error_message=short_err(exc))
            finish()
            return

        if not san.get("ok"):
            result.update(
                status="invalid_output",
                output_summary=san.get("output_summary", {}),
                error_type=san.get("error_type", "invalid_output"),
                error_message=san.get("error_message", ""),
            )
            finish()
            return

        # ---- score (trusted) ----
        try:
            scored = self.score_case(san["sanitized"], raw_case)
            result.update(
                status="ok",
                score=float(scored["score"]),
                success=int(scored["success"]),
                metrics=scored.get("metrics", {}),
                output_summary=san.get("output_summary", {}),
            )
        except Exception as exc:
            result.update(status="exception", error_type="scoring_error",
                          error_message=short_err(exc))
        finish()

    # ----- parent (orchestrates workers, logs everything) ------------------
    def run_eval(self, mode, args):
        if mode not in self.mode_budgets:
            print(f"ERROR: unknown mode '{mode}'. Known: {self.modes()}")
            sys.exit(1)

        self.ensure_data(mode)
        manifest = self.load_manifest(mode)
        if args.max_cases is not None:
            manifest = manifest[: args.max_cases]
        if not manifest:
            print(f"ERROR: no cases found for mode={mode}")
            sys.exit(1)

        budgets = self.budgets_for(mode)
        if args.timeout_s is not None:
            budgets["hard_timeout_s"] = args.timeout_s
            # keep the soft hint strictly under the hard cap
            budgets["soft_time_budget_s"] = max(0.1, args.timeout_s * 0.8)
        if args.memory_mb is not None:
            budgets["memory_mb"] = args.memory_mb
        hard_timeout = float(budgets["hard_timeout_s"])
        soft_budget = float(budgets["soft_time_budget_s"])
        memory_mb = int(budgets["memory_mb"])

        train_sha = sha256_file(TRAIN_PY)
        run_label = args.run_name if args.run_name else train_sha[:8]
        run_id = f"{utc_stamp()}_{mode}_{run_label}"
        run_dir = HERE / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = HERE / ".artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)

        frozen_warn = self.frozen_hash_warnings()
        if args.strict_frozen and frozen_warn:
            print(f"ERROR (--strict-frozen): frozen files changed: {frozen_warn}")
            sys.exit(2)
        for name in frozen_warn:
            print(f"WARNING: frozen file hash changed: {name}")
            print("Future AutoResearch agents should modify only train.py.")

        started = utc_iso()
        print(f"mode={mode}")
        print(f"run_id={run_id}")
        print(f"task={self.task_name}/{self.task_version}")
        print(f"train_sha256={train_sha[:8]}")
        print(f"hard_timeout_s={hard_timeout}  soft_time_budget_s={soft_budget}  memory_mb={memory_mb}")
        print(f"cases={len(manifest)}\n")

        rows = []
        for idx, mrow in enumerate(manifest):
            row = self._run_one_case(
                mrow, idx, run_id, mode, hard_timeout, soft_budget, memory_mb,
                train_sha, artifacts,
            )
            rows.append(row)
            print(
                f"[{idx + 1}/{len(manifest)}] {row['case_id']:<12} "
                f"{row['status']:<16} score={row['score']:.4f} "
                f"runtime={row['runtime_s']:.3f}s"
                + (f"  ({row['error_type']})" if row["error_type"] else "")
            )

        ended = utc_iso()
        agg = self.aggregate(rows)
        self._write_cases_tsv(run_dir, rows)
        self._write_summary_tsv(run_dir, run_id, mode, train_sha, agg)
        self._append_history(run_id, mode, train_sha, agg, run_dir)
        self._write_report(run_dir, run_id, mode, train_sha, agg, rows,
                           hard_timeout, soft_budget, memory_mb, started, ended,
                           frozen_warn)
        self._print_summary(run_id, mode, train_sha, agg, run_dir)
        return agg

    def _run_one_case(self, mrow, idx, run_id, mode, hard_timeout, soft_budget,
                      memory_mb, train_sha, artifacts):
        case_id = mrow["case_id"]
        seed = int(mrow["seed"])
        case_path = (HERE / mrow["case_path"]).resolve()
        out_path = artifacts / f"worker_{run_id}_{idx:04d}.json"

        row = {
            "run_id": run_id,
            "timestamp_utc": utc_iso(),
            "task_name": self.task_name,
            "task_version": self.task_version,
            "mode": mode,
            "case_id": case_id,
            "case_index": idx,
            "seed": seed,
            "case_path": mrow["case_path"],
            "params_json": mrow.get("params_json", "{}"),
            "train_sha256": train_sha,
            "status": "infrastructure_error",
            "score": self.failure_score_value,
            "success": 0,
            "runtime_s": 0.0,
            "hard_timeout_s": hard_timeout,
            "soft_time_budget_s": soft_budget,
            "peak_rss_mb": "",
            "tracemalloc_peak_mb": "",
            "memory_budget_mb": memory_mb,
            "output_summary_json": "{}",
            "metrics_json": "{}",
            "error_type": "",
            "error_message": "",
        }

        cmd = [
            sys.executable, str(HERE / "eval.py"), "--worker",
            "--case-path", str(case_path),
            "--output-path", str(out_path),
            "--seed", str(seed),
            "--soft-time-budget-s", str(soft_budget),
            "--memory-mb", str(memory_mb),
            "--mode", mode,
        ]

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, cwd=str(HERE), timeout=hard_timeout,
                capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired:
            row.update(status="timeout", runtime_s=float(hard_timeout),
                       error_type="timeout",
                       error_message=f"exceeded hard timeout of {hard_timeout}s")
            self._cleanup(out_path)
            return row

        elapsed = time.perf_counter() - t0
        if not out_path.exists():
            tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
            row.update(status="infrastructure_error", runtime_s=elapsed,
                       error_type="no_worker_output",
                       error_message=f"exit={proc.returncode}; {tail[0]}"[:240])
            return row

        try:
            res = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as exc:
            row.update(status="infrastructure_error", runtime_s=elapsed,
                       error_type="bad_worker_output", error_message=short_err(exc))
            self._cleanup(out_path)
            return row

        row.update(
            status=res.get("status", "infrastructure_error"),
            score=float(res.get("score", self.failure_score_value)),
            success=int(res.get("success", 0)),
            runtime_s=float(res.get("runtime_s", elapsed)),
            peak_rss_mb=res.get("peak_rss_mb") if res.get("peak_rss_mb") is not None else "",
            tracemalloc_peak_mb=res.get("tracemalloc_peak_mb") if res.get("tracemalloc_peak_mb") is not None else "",
            output_summary_json=json_compact(res.get("output_summary", {})),
            metrics_json=json_compact(res.get("metrics", {})),
            error_type=res.get("error_type", ""),
            error_message=res.get("error_message", ""),
        )
        self._cleanup(out_path)
        return row

    @staticmethod
    def _cleanup(path):
        try:
            Path(path).unlink()
        except OSError:
            pass

    def aggregate(self, case_rows):
        n = len(case_rows)
        scores = [r["score"] for r in case_rows]
        successes = [r["success"] for r in case_rows]
        runtimes = [r["runtime_s"] for r in case_rows]
        rss = [r["peak_rss_mb"] for r in case_rows
               if isinstance(r["peak_rss_mb"], (int, float))]

        def count(status):
            return sum(1 for r in case_rows if r["status"] == status)

        agg = {
            "num_cases": n,
            "num_ok": count("ok"),
            "num_timeouts": count("timeout"),
            "num_exceptions": count("exception"),
            "num_invalid": count("invalid_output"),
            "aggregate_score": round(sum(scores), 6),
            "mean_score": round(statistics.fmean(scores), 6) if n else 0.0,
            "median_score": round(statistics.median(scores), 6) if n else 0.0,
            "success_rate": round(sum(successes) / n, 6) if n else 0.0,
            "mean_runtime_s": round(statistics.fmean(runtimes), 6) if n else 0.0,
            "p95_runtime_s": round(percentile(runtimes, 95), 6),
            "max_runtime_s": round(max(runtimes), 6) if n else 0.0,
            "mean_peak_rss_mb": round(statistics.fmean(rss), 3) if rss else "",
            "max_peak_rss_mb": round(max(rss), 3) if rss else "",
        }
        try:
            agg["_extra"] = self.aggregate_extra(case_rows) or {}
        except Exception as exc:
            agg["_extra"] = {"aggregate_extra_error": short_err(exc)}
        return agg

    # ----- output writers ---------------------------------------------------
    def _write_cases_tsv(self, run_dir, rows):
        with open(run_dir / "cases.tsv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CASES_COLUMNS, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in CASES_COLUMNS})

    def _summary_row(self, run_id, mode, train_sha, agg, run_dir):
        row = {
            "run_id": run_id,
            "timestamp_utc": utc_iso(),
            "task_name": self.task_name,
            "task_version": self.task_version,
            "mode": mode,
            "train_sha256": train_sha,
            "run_dir": run_dir.relative_to(HERE).as_posix(),
            "extra_metrics_json": json_compact(agg.get("_extra", {})),
        }
        for key in SUMMARY_COLUMNS:
            if key not in row:
                row[key] = agg.get(key, "")
        return row

    def _write_summary_tsv(self, run_dir, run_id, mode, train_sha, agg):
        row = self._summary_row(run_id, mode, train_sha, agg, run_dir)
        with open(run_dir / "summary.tsv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, delimiter="\t")
            writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in SUMMARY_COLUMNS})

    def _append_history(self, run_id, mode, train_sha, agg, run_dir):
        logs = HERE / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        path = logs / "eval_history.tsv"
        row = self._summary_row(run_id, mode, train_sha, agg, run_dir)
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, delimiter="\t")
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in SUMMARY_COLUMNS})

    def _write_report(self, run_dir, run_id, mode, train_sha, agg, rows,
                      hard_timeout, soft_budget, memory_mb, started, ended,
                      frozen_warn):
        scored = sorted(rows, key=lambda r: r["score"])
        hardest = scored[:3]
        easiest = list(reversed(scored[-3:]))

        def case_line(r):
            return (f"| {r['case_id']} | {r['status']} | {r['score']:.4f} | "
                    f"{r['success']} | {r['runtime_s']:.3f} | {r['metrics_json']} |")

        L = []
        L.append(f"# Eval report: {self.task_name}/{self.task_version}")
        L.append("")
        L.append(f"- run_id: `{run_id}`")
        L.append(f"- mode: `{mode}`")
        L.append(f"- train.py sha256: `{train_sha}`")
        L.append(f"- command: `{' '.join(sys.argv)}`")
        L.append(f"- started: {started}")
        L.append(f"- ended: {ended}")
        L.append(f"- hard_timeout_s: {hard_timeout}  soft_time_budget_s: {soft_budget}  memory_budget_mb: {memory_mb}")
        L.append("")
        L.append("## Aggregate")
        L.append("")
        L.append(f"- num_cases: {agg['num_cases']}")
        L.append(f"- aggregate_score: {agg['aggregate_score']}")
        L.append(f"- mean_score: {agg['mean_score']}")
        L.append(f"- median_score: {agg['median_score']}")
        L.append(f"- success_rate: {agg['success_rate']}")
        L.append(f"- timeouts: {agg['num_timeouts']}  exceptions: {agg['num_exceptions']}  invalid_outputs: {agg['num_invalid']}")
        L.append(f"- mean_runtime_s: {agg['mean_runtime_s']}  p95_runtime_s: {agg['p95_runtime_s']}  max_runtime_s: {agg['max_runtime_s']}")
        L.append(f"- mean_peak_rss_mb: {agg['mean_peak_rss_mb']}  max_peak_rss_mb: {agg['max_peak_rss_mb']}")
        if agg.get("_extra"):
            L.append(f"- task_metrics: `{json_compact(agg['_extra'])}`")
        L.append("")
        L.append("## Scoring")
        L.append("")
        L.append(self.scoring_description())
        L.append("")
        L.append("## Hardest cases (lowest score)")
        L.append("")
        L.append("| case_id | status | score | success | runtime_s | metrics |")
        L.append("|---|---|---:|---:|---:|---|")
        L.extend(case_line(r) for r in hardest)
        L.append("")
        L.append("## Easiest cases (highest score)")
        L.append("")
        L.append("| case_id | status | score | success | runtime_s | metrics |")
        L.append("|---|---|---:|---:|---:|---|")
        L.extend(case_line(r) for r in easiest)
        L.append("")
        L.append("## Frozen file hashes")
        L.append("")
        for name in FROZEN_FILES:
            L.append(f"- {name}: `{sha256_file(HERE / name)[:16]}`")
        L.append(f"- train.py: `{train_sha[:16]}`")
        if frozen_warn:
            L.append("")
            L.append(f"> WARNING: frozen files changed since prepare: {frozen_warn}")
        L.append("")
        (run_dir / "report.md").write_text("\n".join(L), encoding="utf-8")

    def _print_summary(self, run_id, mode, train_sha, agg, run_dir):
        print()
        print(f"aggregate_score={agg['aggregate_score']}")
        print(f"mean_score={agg['mean_score']}")
        print(f"median_score={agg['median_score']}")
        print(f"success_rate={agg['success_rate']}")
        print(f"mean_runtime_s={agg['mean_runtime_s']}")
        print(f"p95_runtime_s={agg['p95_runtime_s']}")
        print(f"max_peak_rss_mb={agg['max_peak_rss_mb']}")
        print()
        print(f"timeouts={agg['num_timeouts']}")
        print(f"exceptions={agg['num_exceptions']}")
        print(f"invalid_outputs={agg['num_invalid']}")
        print()
        print(f"run_dir={run_dir.relative_to(HERE).as_posix()}")
        print("history=logs/eval_history.tsv")

    def describe(self):
        print(f"task_name={self.task_name}")
        print(f"task_version={self.task_version}")
        print(f"data_root={self.data_root().relative_to(HERE).as_posix()}")
        print("modes (budgets):")
        for mode in self.modes():
            b = self.budgets_for(mode)
            print(f"  {mode}: hard_timeout_s={b['hard_timeout_s']} "
                  f"soft_time_budget_s={b['soft_time_budget_s']} memory_mb={b['memory_mb']}")
        print()
        print("scoring:")
        print("  " + self.scoring_description())
        print()
        print("cases.tsv columns: " + ", ".join(CASES_COLUMNS))


# ===========================================================================
# TASK-SPECIFIC SUBCLASS  --  EDIT THIS WHEN CREATING A NEW TASK INSTANCE
# ===========================================================================
#
# Implement these five methods for your task:
#   load_case               -- read one serialized case into a raw_case dict
#   make_instance_for_train -- build the clean object passed to train.solve
#                              (NEVER include hidden answers here)
#   sanitize_output         -- validate/repair the solver's raw output
#   score_case              -- score the sanitized output vs the raw case
#   scoring_description      -- one human-readable paragraph for the report
# Optionally override aggregate_extra for task-specific summary metrics.

class Task(AutoResearch):
    task_name = "max_cut"
    task_version = "v0"

    mode_budgets = {
        "smoke": {"hard_timeout_s": 10, "soft_time_budget_s": 1.0, "memory_mb": 2048},
        "dev": {"hard_timeout_s": 30, "soft_time_budget_s": 3.0, "memory_mb": 4096},
        "final": {"hard_timeout_s": 60, "soft_time_budget_s": 8.0, "memory_mb": 4096},
    }

    def load_case(self, case_path):
        with np.load(case_path) as data:
            return {
                "n": int(data["n"]),
                "edges_u": data["edges_u"].astype(np.int64),
                "edges_v": data["edges_v"].astype(np.int64),
                "reference_cut": int(data["reference_cut"]),  # hidden from solver
            }

    def make_instance_for_train(self, raw_case):
        u = raw_case["edges_u"]
        v = raw_case["edges_v"]
        if u.shape[0] > 0:
            edges = np.stack([u, v], axis=1)
        else:
            edges = np.zeros((0, 2), dtype=np.int64)
        # Deliberately omits reference_cut: the solver must not see the answer.
        return {
            "inputs": {"n": raw_case["n"], "edges": edges},
            "metadata": {"num_edges": int(u.shape[0])},
            "budgets": {},
        }

    def sanitize_output(self, raw_output, raw_case):
        n = int(raw_case["n"])
        summary = {"n": n}
        if raw_output is None:
            return self._invalid(summary, "solver returned None")
        try:
            arr = np.asarray(raw_output)
        except Exception as exc:
            return self._invalid(summary, f"not array-like: {short_err(exc)}")

        flat = arr.reshape(-1)
        if flat.shape[0] != n:
            return self._invalid(summary, f"expected length-{n} vector, got {flat.shape[0]}")

        if arr.dtype == bool:
            side = flat.astype(np.int8)
        else:
            try:
                uniq = np.unique(flat)
            except Exception as exc:
                return self._invalid(summary, f"uncomparable values: {short_err(exc)}")
            if np.all(np.isin(uniq, [0, 1])):
                side = flat.astype(np.int8)
            elif np.all(np.isin(uniq, [-1, 1])):
                side = (flat > 0).astype(np.int8)
            else:
                return self._invalid(summary, "values must be in {0,1} or {-1,1}")

        ones = int(side.sum())
        summary.update({"ones": ones, "balance": round(ones / max(1, n), 3)})
        return {"ok": True, "sanitized": side, "output_summary": summary,
                "error_type": "", "error_message": ""}

    @staticmethod
    def _invalid(summary, message):
        return {"ok": False, "sanitized": None, "output_summary": summary,
                "error_type": "invalid_output", "error_message": message}

    def score_case(self, sanitized_output, raw_case):
        side = sanitized_output
        u = raw_case["edges_u"]
        v = raw_case["edges_v"]
        m = int(u.shape[0])
        ref = int(raw_case["reference_cut"])

        cut = int(np.count_nonzero(side[u] != side[v])) if m > 0 else 0
        denom = ref if ref > 0 else 1
        score = cut / denom
        return {
            "score": score,
            "success": 1 if cut >= ref else 0,
            "metrics": {
                "cut_value": cut,
                "reference_cut": ref,
                "num_edges": m,
                "cut_fraction": round(cut / m, 4) if m > 0 else 0.0,
                "gap_to_reference": cut - ref,
                "normalized_cut": round(score, 4),
            },
        }

    def aggregate_extra(self, case_rows):
        ok = [r for r in case_rows if r["status"] == "ok"]
        fractions = []
        for r in ok:
            try:
                fractions.append(json.loads(r["metrics_json"]).get("cut_fraction", 0.0))
            except Exception:
                pass
        return {
            "mean_cut_fraction": round(statistics.fmean(fractions), 4) if fractions else 0.0,
            "num_beat_reference": sum(1 for r in case_rows if r["success"] == 1),
        }

    def scoring_description(self):
        return (
            "score = cut_value / reference_cut, where reference_cut is a "
            "deterministic greedy 1-pass cut computed at prepare time and hidden "
            "from the solver. success = 1 iff cut_value >= reference_cut. Higher "
            "is better; score > 1.0 beats the greedy reference. Failed cases "
            "(timeout/exception/invalid/oom) receive score 0.0. The headline "
            "metric for hillclimbing is mean_score; also watch success_rate."
        )


# ===========================================================================
# CLI  --  DO NOT EDIT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="AutoResearch evaluation harness")
    parser.add_argument("--mode", default=None, help="smoke | dev | final")
    parser.add_argument("--run-name", default=None, help="label for the run dir")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=None,
                        help="override the per-case HARD wall-clock timeout")
    parser.add_argument("--memory-mb", type=int, default=None,
                        help="override the per-case memory budget (MB)")
    parser.add_argument("--strict-frozen", action="store_true",
                        help="abort if frozen files changed since prepare")
    parser.add_argument("--describe", action="store_true")
    # Worker-only (internal; spawned by the parent per case):
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--soft-time-budget-s", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    task = Task()

    if args.worker:
        task.run_worker(args)
        return
    if args.describe:
        task.describe()
        return
    if not args.mode:
        parser.error("one of --mode / --describe / --worker is required")
    task.run_eval(args.mode, args)


if __name__ == "__main__":
    main()
