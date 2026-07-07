# Probe Sweep Plan

Date: 2026-07-03

This is the operational plan for the next probe sweep. The full knob inventory
lives in `docs/probe_sweep_knob_universe_20260701.md`; this file is the concise
execution plan.

## Current State

Completed:

- First real-artifact smoke test succeeded.
  - Cache: `artifacts/probe_features/no_insider_layer3_two_task_smoke_pooled.h5`
  - Result: `artifacts/probes/no_insider_layer3_two_task_smoke_probe_results.json`
- Full no-insider sparse cache succeeded.
  - Cache: `artifacts/probe_features/no_insider_sparse_pooled.h5`
  - Size: 7,734,086,958 bytes, about 7.2 GiB.
  - Layers: `3,7,11,15,19,23,27,31,35`
  - Tasks: 17 no-insider tasks.
- First no-insider sparse baseline succeeded.
  - Result: `artifacts/probes/no_insider_sparse_probe_results.json`
  - Best aggregate layer: 19.
  - Strong band: 15-23, with 27 still worth checking as the high-side boundary.
- Tracked baseline report exists and is currently staged:
  - `docs/validation/no_insider_sparse_probe_20260628.md`

Current interpretation:

- Probes work.
- Layer 19 is the first strong candidate, not the final answer.
- Cross-task transfer is much weaker than within-task performance.
- `repe_honesty__IF_dishonest` is the clearest negative-transfer task.
- We are now in selection/robustness mode.

## Before Launching Remote Sweeps

Commit the current proof docs first:

```bash
git add docs/validation/no_insider_sparse_probe_20260628.md \
  docs/probe_sweep_knob_universe_20260701.md \
  sweep_plan.md
git commit -m "document probe sweep plan"
git push origin HEAD:main
```

Keep `pooled_feature_cache_optimization_lesson.html` uncommitted unless
explicitly requested.

## Remote CPU Decision

Use a remote CPU box for the sweep phase.

Reason:

- Current probe training is sklearn logistic regression.
- `train-probes-from-cache` is CPU/RAM/disk work, not CUDA work.
- A GPU will mostly sit idle until we do Qwen activation extraction or steering.
- The 7.2 GiB sparse cache is small enough to upload/pull to a remote CPU box.

Recommended remote shape:

- Minimum useful: 16 vCPU, 64 GiB RAM, fast NVMe, 150 GiB disk.
- Preferred: 32-64 vCPU, 128-256 GiB RAM, fast NVMe, 250+ GiB disk.
- For dense-cache work, the box needs room for:
  - repo and environment
  - 58 GiB raw activation HDF5
  - 7.2 GiB existing sparse cache
  - new dense cache
  - sweep outputs

Use GPU only later for:

- new activation extraction
- Qwen generation
- steering/intervention experiments
- any future GPU-native trainer rewrite

## Second Smoke Test

Before the full dense `15-27` cache build, run a second smoke test that touches
uncached dense layers. The first smoke proved the cache/trainer path works on the
real artifact. This second smoke proves dense-layer cache construction works on
layers not present in the sparse cache.

Second smoke cache:

```bash
PYTHONPATH=src uv run --no-sync intelligent-liars cache-pooled-features \
  --input artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5 \
  --output artifacts/probe_features/no_insider_dense_smoke_15_17_two_task_pooled.h5 \
  --layers 15,16,17 \
  --task claims__definitional_gemini_600_full \
  --task claims__evidential_gemini_600_full
```

Second smoke train:

```bash
PYTHONPATH=src uv run --no-sync intelligent-liars train-probes-from-cache \
  --cache artifacts/probe_features/no_insider_dense_smoke_15_17_two_task_pooled.h5 \
  --source artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5 \
  --output artifacts/probes/no_insider_dense_smoke_15_17_two_task_probe_results.json \
  --layers 15,16,17 \
  --c 1.0 \
  --test-size 0.25 \
  --random-seed 0 \
  --general-domain-probe
```

Pass condition:

- Cache builds.
- Training finishes.
- Output contains 6 within-task evaluations, 6 cross-task evaluations, 6 final
  directions, 6 general-domain evaluations, and 3 general-domain directions.

## Dense Cache Build

Build the dense cache once. Do not let multiple agents build it independently.

Dense layers:

```text
15,16,17,18,19,20,21,22,23,24,25,26,27
```

Task set:

- The same 17 no-insider tasks from the sparse baseline.
- Continue excluding:
  - `insider_trading__onpolicy`
  - `insider_trading__upscale`
  - `insider_trading_doubledown__onpolicy`

Output:

```text
artifacts/probe_features/no_insider_dense_15_27_pooled.h5
```

## Primary Sweep

Purpose:

- Find the best layer in `15-27`.
- Find the best regularization value.
- Test whether excluding REPE improves transfer.

Sweep matrix:

```text
layers = 15,16,17,18,19,20,21,22,23,24,25,26,27
C = 0.01,0.03,0.1,0.3,1,3,10
task_set =
  A. all no-insider
  B. all no-insider minus repe_honesty__IF_dishonest
seed = 0
general_task_class_cap = 1000
pooling = mean answer-token pooling
probe = sklearn logistic regression
```

This is 13 layers x 7 C values x 2 task sets = 182 layer/C/task-set evaluations.

Implementation note:

- Each `train-probes-from-cache` call can cover all 13 layers for one `C` and
  one task set.
- That means the primary sweep is 14 process-level jobs, not 182 separate CLI
  calls.

Outputs:

```text
artifacts/probes/sweeps/dense_15_27/all_no_insider_c001.json
artifacts/probes/sweeps/dense_15_27/all_no_insider_c003.json
artifacts/probes/sweeps/dense_15_27/all_no_insider_c01.json
artifacts/probes/sweeps/dense_15_27/all_no_insider_c03.json
artifacts/probes/sweeps/dense_15_27/all_no_insider_c1.json
artifacts/probes/sweeps/dense_15_27/all_no_insider_c3.json
artifacts/probes/sweeps/dense_15_27/all_no_insider_c10.json
artifacts/probes/sweeps/dense_15_27/no_repe_c001.json
artifacts/probes/sweeps/dense_15_27/no_repe_c003.json
artifacts/probes/sweeps/dense_15_27/no_repe_c01.json
artifacts/probes/sweeps/dense_15_27/no_repe_c03.json
artifacts/probes/sweeps/dense_15_27/no_repe_c1.json
artifacts/probes/sweeps/dense_15_27/no_repe_c3.json
artifacts/probes/sweeps/dense_15_27/no_repe_c10.json
```

Summary doc:

```text
docs/validation/no_insider_dense_15_27_primary_sweep_20260703.md
```

## Selection Criteria

Pick finalists by:

1. General-domain balanced accuracy.
2. General-domain AUC.
3. Cross-task balanced accuracy.
4. Cross-task AUC.
5. Worst-task behavior.
6. Whether the chosen layer/C/task-set is stable across nearby layers.

Do not pick by within-task accuracy alone.

Expected output of the primary sweep:

```text
candidate 1: layer X, C=Y, task_set=...
candidate 2: layer X, C=Y, task_set=...
candidate 3: layer X, C=Y, task_set=...
```

## Follow-Up Sweeps

Only run these after the primary sweep picks finalists.

### Task-Source Ablations

Run near the winning layer/C values:

```text
all no-insider
all no-insider minus REPE
claims only
internal-state only
claims + internal-state
Truth-Spec-ish static
geometry-of-truth only
roleplaying excluded
sycophancy excluded
sandbagging excluded
```

### Seed Robustness

Run finalist configs over:

```text
random_seed = 0,1,2,3,4
```

### General-Domain Cap Sweep

Run finalist configs over:

```text
general_task_class_cap = 250,500,1000,unlimited
```

### Direction Construction

Compare:

```text
logistic regression coefficient
difference of means
whitened / covariance-corrected mean difference
task-averaged direction
family-balanced averaged direction
```

## Agent Split

Use narrow agents:

1. Remote setup agent.
   - Prepare machine, repo, environment, artifacts.
   - Does not run full sweep until smoke passes.

2. Dense-cache owner.
   - Runs second smoke.
   - Builds `no_insider_dense_15_27_pooled.h5`.
   - Owns cache provenance.

3. Primary sweep workers.
   - One job per `(task_set, C)` pair.
   - Unique output paths only.

4. Sweep summarizer.
   - Does no training.
   - Reads all JSON outputs.
   - Writes the primary sweep summary doc.

5. Robustness workers.
   - Run only after finalists are chosen.

## Stop Conditions

Stop and ask before continuing if:

- cache provenance does not match the source HDF5
- any output path already exists and would be overwritten
- dense cache build fails
- multiple agents start building the same cache
- metrics contradict the baseline in a way that suggests task/path mixup
- remote machine lacks disk/RAM for dense cache

## What We Are Not Doing Yet

Not yet:

- pooling sweep
- nonlinear probe sweep
- steering
- Qwen generation
- GPU trainer rewrite
- insider tasks

The next goal is a trustworthy shortlist of dense-layer linear directions, not
a final steering intervention.
