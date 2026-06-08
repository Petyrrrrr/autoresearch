# Manual: the AutoResearch framework

This is a small, reusable harness for running an **autonomous optimization /
evaluation research loop**. A solver lives in `train.py`; an evaluator
(`eval.py`) runs it on prepared cases in isolated worker subprocesses with hard
timeouts and memory reporting, scores the outputs, and logs everything to TSV +
Markdown. An AI agent edits `train.py` in a loop, comparing runs, with no human
in the loop.

The repository currently ships a **worked example task: `max_cut`** (Max-Cut on
random graphs). You can run it as-is, or clone the framework and swap in your own
task by editing a few clearly marked sections.

## The three roles

1. **Framework author** (already done): built the generic engine — the
   `AutoResearch` base class, CLI, run ids, seeding, subprocess isolation, hard
   timeouts, memory reporting, sanitization wrapping, failure handling,
   aggregation, TSV logs, and reports. You should not need to touch this.
2. **Instance-building agent** (this is you, when starting a new problem): clone
   the framework and fill in the task-specific pieces (see the checklist below).
3. **AutoResearch agent** (runs afterwards): edits **only** `train.py` in a loop,
   guided by `task.md` and `program.md`. It never edits anything else.

## Files

```text
prepare.py      generates deterministic data + manifest + DATASET.md   (frozen for the AutoResearch agent)
eval.py         the evaluation engine (AutoResearch base) + Task subclass (frozen for the AutoResearch agent)
train.py        the solver — solve(instance, *, rng, time_budget_s, memory_budget_mb)  (THE mutable file)
task.md         exact task definition (input/output contract, scoring, budgets)
program.md      the loop instructions for the AutoResearch agent
instruction.md  the framework spec (how all of this is meant to work)
manual.md       this file
```

Generated at runtime (safe to delete; recreated):

```text
data/<task>/<version>/<mode>/...   prepared cases + manifest.tsv + DATASET.md
runs/<run_id>/                     cases.tsv, summary.tsv, report.md  (one dir per eval run)
logs/eval_history.tsv              one appended summary row per eval run
.artifacts/                        frozen-file hashes + transient worker result files
```

## How to run (the example task)

```bash
python prepare.py --mode smoke      # generate data (eval.py also auto-runs this if missing)
python eval.py    --mode smoke      # fast contract check
python eval.py    --mode dev        # the hillclimbing set
python eval.py    --mode final      # held-out evaluation
python eval.py    --describe        # task + budgets + scoring + log schema
python prepare.py --describe        # modes + seed policy
```

Useful flags: `--run-name LABEL`, `--max-cases N`, `--timeout-s S` (override the
hard timeout), `--memory-mb MB`, `--strict-frozen` (abort if a frozen file
changed since prepare).

## Architecture in one breath

`eval.py` runs each case by spawning **itself** as a worker
(`python eval.py --worker --case-path ... --output-path ...`). The parent
enforces a hard wall-clock timeout via `subprocess.run(timeout=...)`; if the
worker overruns it is killed and the case is scored as a timeout. The worker
loads the case, builds the solver's `instance`, calls the **untrusted**
`train.solve(...)`, then sanitizes + scores the output and writes a small JSON
result the parent folds into the logs. One worker per case = clean memory
attribution and full isolation.

Every case gets a `status`:
`ok | timeout | exception | invalid_output | oom | infrastructure_error`.

---

# Plugging in a NEW task (instance-builder checklist)

You edit exactly **five regions** across four files. Each is marked with a banner
comment `TASK-SPECIFIC SECTION ... EDIT THIS`. The generic engine sections are
marked `DO NOT EDIT`. Work in this order:

### 1. `task.md` — write the contract first

Define, in plain words: the goal, the exact `instance` passed to `solve`, the
exact return value, how data is generated, the modes table, the scoring formula,
invalid-output rules, budgets. In particular, it must clearly specify the evaluation objective and constraints. 
The AutoResearch agent treats this as groundtruth. Also list a few simple research directions.


### 2. `prepare.py` — the TASK-SPECIFIC SECTION at the top

- `TASK_NAME`, `TASK_VERSION` — set both (keep them identical in `eval.py`).
- `MODES` — for each of `smoke` / `dev` / `final`, a `base_seed` and a list of
  per-case parameter dicts. Smoke = tiny & fast; dev = the comparison set; final
  = larger/harder with **disjoint seeds**.
- `generate_case(seed, params) -> dict` — build ONE case deterministically from
  the seed. Return a dict of numpy arrays / scalars to serialize. Put any hidden
  answer (e.g. an optimal value, planted solution) in here too — it gets stored
  for scoring but you will strip it before it reaches the solver (step 3).
  Use keys prefixed with `_` (e.g. `_params`, `_input_size`) for manifest-only
  metadata that should NOT be serialized into the case file.
- `runtime_class(params)` and `describe_dataset()` — descriptive only; keep them
  roughly accurate.

Cases are saved with `np.savez` (`.npz`). For pure-JSON tasks you may instead
write one `.json` per case — just make `load_case` (step 3) match.

### 3. `eval.py` — the `Task(AutoResearch)` subclass near the bottom

Set `task_name` / `task_version` (match `prepare.py`) and `mode_budgets`
(`hard_timeout_s`, `soft_time_budget_s`, `memory_mb` per mode). Then implement:

- `load_case(self, case_path) -> raw_case` — read one serialized case back into a
  dict. This is the inverse of `prepare.save_case`. The `raw_case` MAY contain
  hidden answers (used only for scoring).
- `make_instance_for_train(self, raw_case) -> instance` — build the clean object
  handed to `train.solve`. **Strip every hidden answer here.** Convention:
  `{"inputs": ..., "metadata": ..., "budgets": {}}` (the engine fills `budgets`).
- `sanitize_output(self, raw_output, raw_case) -> dict` — validate/repair the
  untrusted solver output. Never raise. Return
  `{"ok": True, "sanitized": <canonical>, "output_summary": {...},
    "error_type": "", "error_message": ""}` on success, or
  `{"ok": False, "sanitized": None, "output_summary": {...},
    "error_type": "invalid_output", "error_message": "why"}` on rejection.
- `score_case(self, sanitized_output, raw_case) -> dict` — return
  `{"score": float, "success": 0|1, "metrics": {...}}`. Higher score = better.
  Keep `metrics` small and JSON-friendly.
- `scoring_description(self) -> str` — one paragraph; printed in every report.
- *(optional)* `aggregate_extra(self, case_rows) -> dict` — extra run-level
  metrics, stored as JSON in `summary.tsv` / `eval_history.tsv`.

You normally do **not** touch: `failure_score_value` (default `0.0` for failed
cases), or anything in the `AutoResearch` base class / CLI.

### 4. `train.py` — a simple, valid baseline

Implement `solve(instance, *, rng=None, time_budget_s=None, memory_budget_mb=None)`
returning a valid (not necessarily good) candidate. Keep it simple and fast, use
`rng` for randomness, respect `time_budget_s`, and leave obvious headroom for the
AutoResearch agent to improve. This is the only file that agent will edit.

### 5. Verify, then hand off

Run the self-check (all must complete without crashing):

```bash
python prepare.py --mode smoke --force
python eval.py    --mode smoke
python eval.py    --mode dev --max-cases 5
python eval.py    --describe
python prepare.py --describe
```

Then search the repo for leftover `{{`, `PLACEHOLDER`, `TODO`, `TBD` and resolve
them. The dev score does not need to be good — the baseline just needs to be
valid and improvable. Hand the repo to the AutoResearch agent by pointing it at
`program.md`.

## Design contracts to preserve

- The solver must NEVER receive hidden answers — enforce this in
  `make_instance_for_train`.
- `sanitize_output` and `score_case` are trusted and must not raise on bad input;
  return a rejection dict instead.
- Determinism: all randomness flows from explicit seeds (`base_seed + index` at
  prepare time; a seeded `rng` handed to `solve` at eval time).
- TSV everywhere via `csv.DictWriter(delimiter="\t")`; `eval_history.tsv` has a
  fixed schema (task-specific aggregates go into its `extra_metrics_json` column).
- Keep it simple. Numpy + standard library only (scipy optional).
