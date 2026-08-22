# Intelligent Liars

Qwen3-VL activation and probe experiments for truth/deception signals.

The current project state is: rollout generation, judging, activation extraction,
HDF5 validation, probe preflight, and simple linear probe training are all wired.
The canonical all-text activation store is validated and tracked by DVC. Probe
training is the next experimental phase; the tracked preflight report did not
train probes.

## Current State

Canonical activation artifact:

```text
artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5
```

Tracked evidence:

- `docs/postmortems/validation/activation_all_text_20260624.md`
- `docs/postmortems/validation/probe_preflight_all_text_20260625.json`

Preflight headline:

- 20 tasks
- 36 decoder layers
- hidden dim 4096
- 76,257 examples
- 18 trainable tasks
- 1 eval-only task: `insider_trading__onpolicy`
- 1 blocked task: `insider_trading__upscale`
- recommended sparse pilot layers: `3,7,11,15,19,23,27,31,35`

## Setup

This repo uses `uv`.

```bash
uv sync --dev --no-editable
cp .env.example .env
uv run --no-sync intelligent-liars check-env
```

For judge-backed commands, set `OPENROUTER_API_KEY` in `.env` or the shell.
`HF_TOKEN` is optional and only needed if Hugging Face requires authentication
for model downloads.

On CUDA machines, install hardware-specific packages outside the base lock:

```bash
uv sync --frozen --dev --no-editable
uv pip install ninja packaging psutil
MAX_JOBS=${MAX_JOBS:-4} uv pip install flash-attn --no-build-isolation
uv run --no-sync intelligent-liars check-env --require-cuda --check-model
```

## Main Commands

List commands:

```bash
uv run --no-sync intelligent-liars --help
```

Useful entrypoints:

```bash
# Environment checks
uv run --no-sync intelligent-liars check-env --check-processor
uv run --no-sync intelligent-liars check-env --require-cuda --check-model

# Rollouts and judging
uv run --no-sync intelligent-liars generate-rollouts --batch-size 1
uv run --no-sync intelligent-liars grade-rollouts
uv run --no-sync intelligent-liars run-qwen-sweep --batch-size 1

# Sycophancy source generation
uv run --no-sync intelligent-liars generate-sycophancy

# Activations
uv run --no-sync intelligent-liars extract-activations --layers sparse --batch-size 1
uv run --no-sync intelligent-liars merge-activation-shards --help

# Probe readiness and training
uv run --no-sync intelligent-liars preflight-probes \
  --input artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5 \
  --output docs/validation/probe_preflight_all_text_20260625.json

uv run --no-sync intelligent-liars train-probes \
  --input artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5 \
  --output artifacts/probes/sparse_probe_results.json \
  --layers 3,7,11,15,19,23,27,31,35 \
  --general-domain-probe \
  --task claims__definitional_gemini_600_full \
  --task claims__evidential_gemini_600_full \
  --task claims__fictional_gemini_600_full \
  --task claims__logical_gemini_600_full \
  --task ethics__commonsense \
  --task geometry_of_truth__best \
  --task geometry_of_truth__mixed \
  --task internal_state__animals \
  --task internal_state__cities \
  --task internal_state__companies \
  --task internal_state__elements \
  --task internal_state__facts \
  --task internal_state__inventions \
  --task repe_honesty__IF_dishonest \
  --task roleplaying__plain \
  --task sandbagging_v2__wmdp_mmlu \
  --task sycophancy__mmlu_stem_same_conf_all
```

Do not train on `insider_trading__upscale` until the label/data issue is fixed.
Treat `insider_trading__onpolicy` as eval-only unless the class balance changes.
For the first no-insider sparse pass, also exclude
`insider_trading_doubledown__onpolicy`.

Optional pooled-feature cache for repeated probe sweeps:

```bash
PYTHONPATH=src uv run --no-sync intelligent-liars cache-pooled-features \
  --input artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5 \
  --output artifacts/probe_features/no_insider_sparse_pooled.h5 \
  --layers 3,7,11,15,19,23,27,31,35 \
  --task claims__definitional_gemini_600_full \
  --task claims__evidential_gemini_600_full \
  --task claims__fictional_gemini_600_full \
  --task claims__logical_gemini_600_full \
  --task ethics__commonsense \
  --task geometry_of_truth__best \
  --task geometry_of_truth__mixed \
  --task internal_state__animals \
  --task internal_state__cities \
  --task internal_state__companies \
  --task internal_state__elements \
  --task internal_state__facts \
  --task internal_state__inventions \
  --task repe_honesty__IF_dishonest \
  --task roleplaying__plain \
  --task sandbagging_v2__wmdp_mmlu \
  --task sycophancy__mmlu_stem_same_conf_all

PYTHONPATH=src uv run --no-sync intelligent-liars train-probes-from-cache \
  --cache artifacts/probe_features/no_insider_sparse_pooled.h5 \
  --source artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5 \
  --output artifacts/probes/no_insider_sparse_probe_results.json \
  --layers 3,7,11,15,19,23,27,31,35 \
  --general-domain-probe
```

## Code Map

The code is split by workflow:

- `src/intelligent_liars/cli.py`: Typer entrypoint and compatibility exports.
- `src/intelligent_liars/cli_*.py`: command groups.
- `src/intelligent_liars/models.py`: Qwen3-VL model/processor loading.
- `src/intelligent_liars/rollouts.py`: rollout generation and JSON merge logic.
- `src/intelligent_liars/activation_types.py`: activation labels and dataclasses.
- `src/intelligent_liars/activation_loaders.py`: named task and rollout loaders.
- `src/intelligent_liars/activation_masks.py`: Qwen rendering and token masks.
- `src/intelligent_liars/activation_extract.py`: extraction runtime.
- `src/intelligent_liars/activation_hdf5.py`: HDF5 write/merge/validation helpers.
- `src/intelligent_liars/activations.py`: compatibility facade for activation imports.
- `src/intelligent_liars/judge_config.py`: judge config and preflight.
- `src/intelligent_liars/judge_client.py`: OpenRouter/generic judge calls.
- `src/intelligent_liars/*_grader.py`: roleplaying, sandbagging, and insider graders.
- `src/intelligent_liars/rollout_grading.py`: rollout grading orchestration.
- `src/intelligent_liars/judging.py`: compatibility facade for judging imports.
- `src/intelligent_liars/probes.py`: probe preflight and linear probe training.
- `src/intelligent_liars/interventions.py`: truth-coordinate transforms, runtime hooks, and rank-one checkpoint edits.
- `src/intelligent_liars/standalone_models.py`: teacher generation, 13-coordinate TinyLoRA distillation, merged checkpoints, and fleet plans.
- `src/intelligent_liars/run_control.py`: locks, plan IDs, hashes, signal cleanup.
- `src/intelligent_liars/dynamic_queue.py`: shared filesystem queue helpers.
- `src/intelligent_liars/tinylora_corpus.py`: provenance-aware corpus normalization, linked-scenario
  rendering, integrity validation, and deterministic packaging.

Intervention usage and materialization limits are documented in
`docs/intervention_pipeline.md`.

Long-running GPU runners:

- `scripts/run_activation_extraction_dynamic.py`
- `scripts/run_insider_generation_dynamic.py`

Validation and artifact helpers:

- `scripts/validate_activation_hdf5.py`
- `scripts/validate_insider_generation_json.py`
- `scripts/safe_fetch_artifact.py`
- `scripts/plan_activation_hdf5_merge.py`
- `scripts/fetch_tinylora_preservation_snapshot.py`
- `scripts/build_tinylora_corpus.py`

## Data And Artifacts

Tracked upstream source data lives under `data/`. Small authored corpus definitions and their QA
records live under `corpora/`. Large generated files stay out of Git
and are represented by `.dvc` pointers under `artifacts/`.

The configured DVC remote is:

```text
gdrive-artifacts -> gdrive://1C7KUkZHCfQXnpkt6debhCcA4YtV8eOHX
```

Common DVC commands:

```bash
uvx --from "dvc[gdrive]" dvc status
uvx --from "dvc[gdrive]" dvc pull
uvx --from "dvc[gdrive]" dvc push
```

Keep local OAuth/client secrets out of Git:

```text
.dvc/config.local
.secrets/dvc-gdrive-user-credentials.json
```

To validate the canonical activation store after `dvc pull`:

```bash
PYTHONPATH=src uv run --no-sync python scripts/validate_activation_hdf5.py \
  artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5 \
  --expected-task-count 20 \
  --expected-layer-count 36 \
  --expected-hidden-dim 4096 \
  --finite-check sample \
  --sha256
```

## Development Checks

Use the repo's current validation gate:

```bash
PYTHONPATH=src uv run --no-sync python -m pytest tests -q
PYTHONPATH=src uv run --no-sync ruff check scripts src tests
git diff --check
```

`ruff format` is not currently a repo-wide clean gate; many older files would be
rewrapped by Ruff's formatter. Prefer formatting only files you intentionally
touch unless the project explicitly decides to do a separate formatting commit.

## Run Notes

Start here for historical context:

- `docs/standalone_model_decision_record_20260817.md`
- `docs/minimal_parameter_misalignment_and_gman_20260817.md`
- `docs/postmortems/vast_qwen_rollouts_20260621.md`
- `docs/postmortems/activation_full_20260622.md`
- `docs/postmortems/moar_activation_extraction_20260625/postmortem.md`

External Apollo/Truth Spec references are not vendored. If needed, clone them
under ignored `references/`.
