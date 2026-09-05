# Documentation

The root of `docs/` contains only the current intervention protocol and active
design notes:

- [`simple_truth_direction_experiment_protocol_20260825.md`](simple_truth_direction_experiment_protocol_20260825.md) — scientific protocol and acceptance gates.
- [`simple_truth_direction_run_plan_20260826.md`](simple_truth_direction_run_plan_20260826.md) — next execution gate; it is not authorization to spend.
- [`truth_editing_live_run_history.md`](truth_editing_live_run_history.md) — current operational entry point and append-only launch history.
- [`heretic_optuna_port.md`](heretic_optuna_port.md) — proposed Optuna search integration.
- [`llm_judge_harness.md`](llm_judge_harness.md) — proposed semantic judge boundary.
- [`heretic_modification_pipeline.md`](heretic_modification_pipeline.md) — end-to-end Heretic port, code seams, and decision forks joining Optuna and judging.

Supporting records are organized by role:

- `validation/` contains probe and activation validation receipts.
- `postmortems/` contains curated incident and recovery reports.
- `archive/2026/` contains dated runbooks, handoffs, plans, and preserved Step 5
  operational records that are no longer authoritative.

Historical documents explain prior decisions; they do not override the current
code, [`PROJECT.md`](../PROJECT.md), or the current static specification.
