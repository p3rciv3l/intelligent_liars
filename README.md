# Intelligent Liars

Research code for measuring and intervening on truth reporting in language and
vision-language models. The tracked repository currently contains the Qwen3-VL
activation/probe pipeline and the software contracts for a simple Step 5
intervention experiment.

## Status

The activation, probe, intervention, contract-validation, and evidence-checking
paths are covered by software tests. Historical activation and probe artifacts
are tracked with DVC.

That does **not** establish a causal deception mechanism. A linear probe can
decode a truth-related direction without that direction controlling deceptive
behavior. The simple affine, scalar, clamp, projection, and reflection
interventions are software-ready, but the proposed behavioral experiment
suite has not yet been completed. Earlier scalar-addition and bounded-clamp arms
produced negative diagnostic evidence; they were not complete scientific tests
and do not cover the remaining simple transforms. Treat dry runs, unit tests,
probe AUROC, training loss, and infrastructure receipts as separate evidence
classes—not as successful deception results.

The project’s current scope, standards, and evidence boundary are defined in
[PROJECT.md](PROJECT.md).

## Repository map

- `src/intelligent_liars/`: reusable experiment, evaluation, and infrastructure
  modules.
- `src/intelligent_liars/step5_intervention_experiments/`: static contracts,
  manifests, runtime integration, and scientific-evidence records for the simple
  intervention suite.
- `scripts/`: activation/probe runners, validators, and artifact tooling.
- `tests/`: CPU software tests plus explicitly gated model/GPU integration tests.
- `data/`: source and derived evaluation inputs; large generated data should be
  represented by DVC pointers.
- `artifacts/`: generated outputs; large durable files are represented by DVC
  pointers rather than committed to Git.
- `docs/`: active protocols, implementation notes, decisions, and historical
  records.

## Setup

Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync --dev --no-editable
cp .env.example .env
uv run --no-sync intelligent-liars check-env
```

Set `OPENROUTER_API_KEY` only for judge-backed workflows. Set `HF_TOKEN` only
when authenticated model access is required. Do not commit `.env`, local DVC
credentials, cloud credentials, private keys, signed URLs, or raw private
rollouts.

Inspect the supported CLI rather than copying an old run command:

```bash
uv run --no-sync intelligent-liars --help
```

The tracked CLI includes environment checks, rollout generation and grading,
activation extraction and merging, and probe preflight/training. Long-running
or paid experiments require a reviewed versioned contract, identity checks,
budget authority, watchdog, and artifact-preservation procedure; the presence
of a runnable script is not permission to launch it.

## Development checks

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check scripts src tests
git diff --check
```

Some tests require model weights, CUDA, credentials, or external services and
must remain opt-in. A passing CPU suite proves software behavior only.

## DVC artifacts

Git tracks `.dvc` pointers for large artifacts. Check and retrieve them from the
configured Google Drive remote with:

```bash
uvx --from "dvc[gdrive]" dvc status -c
uvx --from "dvc[gdrive]" dvc pull
```

Before deletion, verify exact remote recovery by size and hash. DVC status is
not scientific evidence, and it does not protect unrelated Git LFS objects.

## Active documentation

- [Project charter and current evidence](PROJECT.md)
- [Simple truth-direction protocol](docs/simple_truth_direction_experiment_protocol_20260825.md)
- [Simple truth-direction run plan](docs/simple_truth_direction_run_plan_20260826.md)
- [Heretic/Optuna port design](docs/heretic_optuna_port.md)
- [LLM-judge harness design](docs/llm_judge_harness.md)

Historical reports and receipts document what happened in earlier runs; they do
not override current code, versioned contracts, or the scientific acceptance
criteria. See `docs/archive/`, `docs/postmortems/`, and `docs/validation/` when
reconstructing that history.

## Evidence and safety rules

- Keep training, qualification, inference, grading, and scientific acceptance
  as distinct stages.
- Evaluate held-out deceptive triggers alongside truthful, false-trigger,
  capability, coherence, safety, and control-distribution checks.
- Preserve exact configs, seeds, source identities, hashes, raw generations,
  grader records, and terminal-state receipts.
- Never inspect cross-arm partial scientific outputs before a frozen concurrent
  block completes.
- Stop or pause compute on completion, idle, budget exhaustion, identity
  mismatch, non-finite state, infrastructure corruption, or safety failure.
- Never delete retained compute or artifacts until recovery and scoring needs are
  resolved and durable copies are independently verified.
