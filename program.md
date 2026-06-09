# Program: AutoResearch Agent Instructions

You are an autonomous research agent optimizing this task. You work in a loop,
with no human in the loop, until you are stopped.

Read first:

```text
task.md
```

This contains the task description and a few suggested research directions. You may edit ONLY:

```text
train.py
```

Do NOT edit:

```text
prepare.py
eval.py
task.md
program.md
instruction.md
```

## Loop

Repeat indefinitely:

1. Inspect the current `train.py` and recall the best score so far
   (check `logs/eval_history.tsv`).
2. Form ONE concrete hypothesis for improvement of the current `train.py` (e.g. "add a centered spectral refinement step")
3. Modify ONLY `train.py`.
4. Smoke test (fast contract check after any risky edit):

   ```bash
   python eval.py --mode smoke
   ```

5. If smoke passes, run dev (the comparison set):

   ```bash
   python eval.py --mode dev
   ```

6. Compare against previous runs. Keep the change if it improves dev `mean_score` (or improves robustness /
   runtime without hurting `mean_score`). Otherwise revert it. Log your research attempts and results in RESEARCH_LOG.md.

7. Go to 1.

## Constraints

- No new dependencies beyond the standard library + requirements.txt.
- No hardcoded answers, hidden seeds, case ids, or evaluation constants. Do not
  try to read hidden answers from the data files — it is intentionally withheld
  from the solver.
- Respect `time_budget_s` as a soft budget: return the best result found so far
  before it elapses. Never exceed the hard timeout (your worker will be killed
  and the case penalized).
- Always return a valid output rather than crashing.
- Keep `train.py` simple enough to debug. All else equal, simpler is better.


## Metric, Task Summary, and Starting Directions

See `task.md` for the evaluation metric, task description, and suggested research directions.

## Prior research

RESEARCH_LOG.md contains possible past research attempts and their results. Try to take a look at it to get an idea of what has been tried and what has worked. As you go, also log your research attempts and results in it.
