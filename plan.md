# Full Text Truth Direction Activation Plan
Date: 2026-06-24

## Objective
Do not train probes yet.

The current goal is to finish all text truth-direction activation artifacts for Qwen3-VL before any probe training:

1. Keep the validated rollout4 HDF5 as a fixed input.
2. Finish the static Truth Spec/FLEED activation HDF5.
3. Finish Qwen-specific model-dependent sources and activations.
4. Merge rollout4 + static + model-dependent HDF5s into one final all-text HDF5.
5. DVC-push artifacts.
6. Only then train per-task and joint directions.

## Status - Completed 2026-06-25

This file is now a historical execution plan and run log for the pre-probe
activation gate. The gate is complete: the final all-text HDF5 exists locally,
validates with the expected SHA-256, and is synced to the DVC remote.

Canonical validation evidence is tracked in
`docs/validation/activation_all_text_20260624.md`. Use that report as the
handoff answer for which artifact is canonical, which DVC pointer owns it, what
SHA/size/task/layer validation proves it, and which commands produced that
proof.

Final artifact:

```text
artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5
```

Final verification:

```text
sha256:       c6e687f69256544121d0718cf8bba142ed5837221976121e1899552dc76f1a5a
dvc md5:      b6e82b698513d2372949f2752e17005a
tasks:        20
layers:       36
hidden_dim:   4096
finite check: sample, 720 datasets checked, 0 non-finite
dvc remote:   final gate pointers report Cache and remote 'gdrive-artifacts' are in sync.
```

The broader DVC workspace still has a historical source-pointer caveat:
`artifacts/activations/activation_truthspec_static_20260624/extracted_feats_truthspec_static_qwen3-vl-8b-thinking.h5.dvc`
reports its HDF5 object as remote-new. The final all-text artifact itself is
the completed pre-probe source of truth.

Probe boundary:

```text
no probe training was run for this gate
no probe artifact files were modified in this phase
```

The sections below preserve the operational history and command staging that led
to the completed state. They should not be read as remaining work unless a later
section explicitly says it is still open.

## Confirmed Decisions
Use fresh merged HDF5s, not a multi-HDF5 probe runner.

Include `geometry_of_truth__mixed` now.

Include all `internal_state__*` tasks now.

Keep implied insider report labels skipped:

```text
explicit  -> 0
concealed -> 1
implied   -> -1
invalid   -> -1
```

Later, after all activations exist, train both per-task directions and joint directions.

## Operational Invariants
The `/docs` run notes are binding for the remaining GPU work:

1. Use `/dev/shm` for model cache, run dirs, HDF5 shards, merged HDF5 outputs, and large temp files after checking free space.
2. Use dynamic queues for long Qwen generation work. Avoid static equal shards for tail-heavy generation.
3. Use two persistent Qwen workers per A100 for text generation when VRAM allows it. This is the validated 99-100% utilization shape on the 8x A100 box.
4. Keep file writes isolated. Workers write separate JSON/HDF5 shards; only one post-completion merge writes the canonical output.
5. Do not run two writers against the same canonical JSON or HDF5.
6. Do not extract an insider upscale activation file from a one-sided label distribution.
7. Dynamic activation restarts re-queue any stale `running/` units and only promote them to `done/` if a complete shard file exists.
8. Restart handling now treats malformed `running/` JSON as recoverable-by-filename when possible and only promotes units when completed shard payloads are non-empty and structurally valid (`metadata` + layer groups), preventing silent resurrection of partial files.
9. Fetch large HDF5s into `*.tmp`, verify remote/local byte counts and sha256, then atomically move into place. Do not append-resume HDF5 copies.
10. Probe training remains blocked until final all-text HDF5 validation and DVC push complete.
11. Use merge/compression controls to avoid GPU-side throughput collapse:
   - CLI and dynamic merge/extraction flows now default to `lzf`; still pass
     `--compression lzf` explicitly in runbooks when using concat merges.
   - Use `--compression none` only when validating temporary scratch merges.
   - Keep final archive artifacts on `gzip` for storage-cost control.
   - Never merge with implicit gzip defaults during bursty extraction windows.
   - For final all-text merges with disjoint task inputs, use direct-copy merge
     (`--merge-strategy copy-disjoint`) so HDF5 datasets are copied without
     read/concatenate/rewrite recompression.
12. Always pre-check filesystem and free space before writing large files; default heavy outputs should live on `/dev/shm` or another large mounted volume.

## Immediate Post-Pause Optimization Work (Code + Docs)
- Added `--compression` to `extract-activations`, `merge-activation-shards`, and
  `run_activation_extraction_dynamic.py` with mode choices `gzip|lzf|none`.
- Added robust HDF5 validation helper `scripts/validate_activation_hdf5.py`.
- Added safe fetch helper `scripts/safe_fetch_artifact.py` with verified
  `*.tmp` staging + SHA256/size checks + atomic promotion.
- Added resume-safe insider generation runner
  `scripts/run_insider_generation_dynamic.py` (prevents queue deletion on resume).
- Added remote Vast environment preflight helper
  `scripts/preflight_vast_gpu_environment.py` to fail fast on bad cache paths,
  insufficient `/dev/shm`, missing GPUs, or missing `.env`.
- Added guarded remote resume script renderer
  `scripts/render_vast_insider_resume.py` so the next paid GPU session starts
  from a generated bash script with environment preflight, read-only queue
  audit, explicit 406/114 resume-state assertion, model prefetch, duplicate
  tmux-session refusal, and no queue deletion.
- Added guarded remote finish script renderer
  `scripts/render_vast_insider_finish.py` for the post-generation sequence:
  assert 520/520 queue completion, validate full s20 JSON before judge calls,
  grade without structured outputs, promote canonical insider JSON only after
  graded explicit/concealed label gates pass, extract `insider_trading__upscale`,
  and validate the activation HDF5 with sha256.
- Added direct-copy disjoint HDF5 merge support to
  `merge_activation_hdf5_shards`. `--merge-strategy auto` copies disjoint task
  inputs without materializing activation arrays in memory; `copy-disjoint`
  forces that behavior and refuses duplicated tasks. CLI and lower-level
  compression defaults are now `lzf` instead of `gzip`.
- Added graded insider generation validator/promoter
  `scripts/validate_insider_generation_json.py` so s20 can only become the
  canonical Qwen insider file after full count, grading, run-id, and
  explicit/concealed label-balance checks pass.
- Added Qwen sycophancy dataset tooling `src/intelligent_liars/sycophancy.py`.

## Local Safety Audit - 2026-06-24T22:40:29Z
- Existing completed local HDF5 inputs are intact under finite-sample
  validation: rollout4, static Truth Spec/FLEED, and fresh sycophancy all
  report `ok=true`, 36 layers, hidden dim 4096, and 0 sampled non-finite
  activation datasets.
- Stricter final-merge preflight with `--validate-inputs`,
  `--expected-layer-count 36`, `--expected-hidden-dim 4096`, and the explicit
  20-task list validates the three present inputs and fails only because
  `insider_trading__upscale` is still missing.
- Insider s20 queue snapshot audit is clean: planned 520, done 406, pending
  114, running 0, failed 0, valid outputs 406, and no duplicate/missing/corrupt
  queue files.
- Partial insider JSON promotion correctly fails: 406/520 records, ungraded
  records, no explicit/concealed report labels yet. Do not promote it.
- Local focused tests and lint pass; full suite passes with `198 passed`.
- No local DVC, Qwen, insider-generation, or activation-extraction process is
  currently running.
- SSH to `root@159.48.242.25 -p 60232` currently returns connection refused, so
  no remote GPU resume can run from this workstation until the instance is
  reachable again.
- The preferred next remote entrypoint is now:
  `uv run --no-sync python scripts/render_vast_insider_resume.py --write /tmp/insider_s20_resume.sh --overwrite && bash /tmp/insider_s20_resume.sh`.
  It refuses to launch if the run dir is missing, a duplicate tmux session
  exists, the remote environment preflight fails, or the queue audit does not
  exactly match the expected 406 done / 114 pending state.
- Renderer validation at 2026-06-24T22:46:01Z: focused resume/preflight/fetch/
  validation tests pass with `43 passed`; later full-suite validation is recorded
  below.
- Finish renderer validation: `tests/test_render_vast_insider_finish.py` passes
  with `5 passed` and Ruff passes for the renderer and tests.
- Full validation at 2026-06-24T22:58:30Z: `pytest -q` passes with
  `198 passed`; `ruff check scripts src tests` passes; local insider s20
  snapshot audit remains clean at 406 done / 114 pending; current final merge
  preflight validates present inputs and fails only on missing
  `insider_trading__upscale`; no local DVC/Qwen/insider/activation runner is
  active; GPU SSH still returns connection refused.
- Recheck at 2026-06-24T23:04:49Z:
  - `pytest -q`: `198 passed`.
  - `ruff check scripts src tests`: passed.
  - Focused activation/CLI/merge-planner/HDF5-validator tests: `77 passed`.
  - Local completed HDF5s validate with 36 layers, hidden dim 4096, and 0
    sampled non-finite activation datasets.
  - Current local HDF5 sha256 values:
    - rollout4:
      `2d28167b26c3f0e3688e61195789834dd5d93d27d46016bae5458f2b029af386`
    - static Truth Spec/FLEED:
      `868814fc3ef756d84bc970f2ff320b6bf3be9819759993ac3fab9e655841b88b`
    - fresh sycophancy:
      `8674690bc878511a935863459e2773a90c5490a39a59b173cf611841330f1aca`
  - Final merge preflight sees 19 validated tasks, no duplicates, and fails only
    because `insider_trading__upscale` is intentionally not present yet.
  - Insider s20 saved snapshot remains clean: planned 520, done 406, pending
    114, running 0, failed 0, outputs 406, valid outputs 406.
  - `/tmp/insider_s20_resume.sh` and `/tmp/insider_s20_finish.sh` render
    successfully and pass `bash -n`.
  - SSH to `root@159.48.242.25 -p 60232` still returns connection refused.

## Merge Optimization Audit - 2026-06-24T22:51:54Z
- The previous generic `merge_activation_hdf5_shards` path collected metadata
  and activation arrays task-by-task, concatenated them in memory, and rewrote
  datasets. That remains necessary for overlapping shards of the same task.
- New `--merge-strategy auto` uses direct HDF5 copy whenever all input tasks are
  disjoint. `--merge-strategy copy-disjoint` forces that faster path and refuses
  duplicate tasks.
- Direct-copy merge preserves each source dataset's existing compression and
  avoids read/concatenate/rewrite recompression.
- Current completed local inputs are disjoint:
  rollout4 has 4 tasks, static Truth Spec/FLEED has 14 tasks, and fresh
  sycophancy has 1 task.
- Final all-text merge should therefore use `--merge-strategy copy-disjoint`
  after `insider_trading__upscale` is present and the merge planner confirms no
  duplicate tasks across the four inputs.
- Validation at 2026-06-24T22:52:36Z: activation/CLI tests passed with
  `66 passed`; later full-suite validation is recorded above.
- Current merge preflight validates the three present inputs and reports no
  duplicate tasks; it fails only on the expected missing
  `insider_trading__upscale`.
- GPU SSH to `root@159.48.242.25 -p 60232` still returns connection refused, and
  no local DVC/Qwen/insider/activation runner is active.

## Completed Inputs
### Rollout4 Combined HDF5
Validated local artifact:

```text
artifacts/activations/activation_rollout4_20260623/extracted_feats_rollout4_qwen3-vl-8b-thinking.h5
sha256: 2d28167b26c3f0e3688e61195789834dd5d93d27d46016bae5458f2b029af386
```

Tasks included:

```text
roleplaying__plain
sandbagging_v2__wmdp_mmlu
insider_trading__onpolicy
insider_trading_doubledown__onpolicy
```

Keep this file unchanged and merge it only at the final all-text stage.

### Static Source Loaders
Preflighted static source counts:

```text
claims__definitional_gemini_600_full: 1186
claims__evidential_gemini_600_full:   1164
claims__fictional_gemini_600_full:    1200
claims__logical_gemini_600_full:      1150
ethics__commonsense:                  4000
repe_honesty__IF_dishonest:            612
geometry_of_truth__best:              6952
geometry_of_truth__mixed:            48238
internal_state__animals:              1008
internal_state__cities:               1458
internal_state__companies:            1200
internal_state__elements:              930
internal_state__facts:                 613
internal_state__inventions:            876
```

## Current Vast Run State
### Cost-Control Pause
The 8x A100 Vast instance was stopped on 2026-06-24 after fetching the durable
local artifacts needed for resume. Do not start a new GPU instance or restart
this one until there is a concrete resume command for the remaining work.

```text
instance: 41988240
action:   vastai stop instance 41988240
result:   SSH refused after stop
note:     GPU charges should be halted; Vast storage charges may continue.
```

### Static Activation HDF5
Remote extraction completed cleanly:

```text
run dir: /dev/shm/activation_truthspec_static_20260624_run
done:    558 shards
failed:  0
examples: 70587 unique examples
```

Remote merge completed and remote validation passed:

```text
target: /dev/shm/intelligent_liars/artifacts/activations/activation_truthspec_static_20260624/extracted_feats_truthspec_static_qwen3-vl-8b-thinking.h5
size:   45024798624
sha256: 868814fc3ef756d84bc970f2ff320b6bf3be9819759993ac3fab9e655841b88b
tasks:  14 expected static tasks
layers: 36
dupes:  0 duplicate example `(source_index, output_index)` keys per task
```

Local safe fetch completed. The fetch wrote a sibling `*.tmp`, verified
remote/local byte count and sha256, then atomically promoted the final HDF5:

```text
local:  artifacts/activations/activation_truthspec_static_20260624/extracted_feats_truthspec_static_qwen3-vl-8b-thinking.h5
size:   45024798624
sha256: 868814fc3ef756d84bc970f2ff320b6bf3be9819759993ac3fab9e655841b88b
```

Local structural validation passed with:

```text
script: scripts/validate_activation_hdf5.py
tasks:  14
layers: 36
hidden: 4096
finite: sample check, 504 datasets checked, 0 non-finite
```

DVC add succeeded and wrote the pointer:

```text
artifacts/activations/activation_truthspec_static_20260624/extracted_feats_truthspec_static_qwen3-vl-8b-thinking.h5.dvc
```

DVC push is not complete. The first push ran for about 93 minutes, then failed
with `EOF occurred in violation of protocol (_ssl.c:2427)`. A retry with
`--jobs 1` failed immediately with the same TLS EOF. `dvc status -c` still
reports this static HDF5 as new in the cloud remote.

Validation watcher:

```text
tmux: static_validate_watch_20260624
log:  /dev/shm/intelligent_liars/logs/static_validate_watch_20260624.log
```

The watcher did not append its final `STATIC_WATCH_DONE` line, so the same
validation was rerun manually against the final remote HDF5 before fetch.

### Qwen Sycophancy Source And Activations
Qwen-specific MMLU STEM sycophancy source generation completed:

```text
data/sycophancy/qwen3-vl-8b-thinking/
raw files:    25
paired files: 25
positive:     2500
negative:     2500
```

Activation extraction completed and merged:

```text
remote: /dev/shm/intelligent_liars/artifacts/activations/activation_truthspec_sycophancy_20260624/extracted_feats_truthspec_sycophancy_qwen3-vl-8b-thinking.h5
local:  artifacts/activations/activation_truthspec_sycophancy_20260624/extracted_feats_truthspec_sycophancy_qwen3-vl-8b-thinking.h5
task:   sycophancy__mmlu_stem_same_conf_all
count:  5000 examples
layers: layer_0 through layer_35
sha256: 8674690bc878511a935863459e2773a90c5490a39a59b173cf611841330f1aca
```

The local HDF5 and the local Qwen sycophancy source directory have been fetched
and validated. The HDF5 validator now checks token-row metadata separately from
example-level split metadata, allows empty `logit_positions` when logits are not
captured, and validates duplicate keys using `example_source_indices` and
`example_output_indices`.

Local structural validation passed with:

```text
script: scripts/validate_activation_hdf5.py
task:   sycophancy__mmlu_stem_same_conf_all
count:  5000
labels: 2500/2500
layers: 36
hidden: 4096
finite: sample check, 36 datasets checked, 0 non-finite
```

DVC status:

```text
artifacts/activations/activation_sycophancy_20260624/extracted_feats_sycophancy_qwen3-vl-8b-thinking.h5.dvc
remote: gdrive-artifacts
cloud status: in sync
```

The older local `activation_sycophancy_20260624` HDF5 is still present and has a
different sha256. Treat the fresh `activation_truthspec_sycophancy_20260624`
file above as the intended final merge input unless the older DVC-tracked file is
explicitly selected.

### Qwen Insider Upscale Source
The first one-sample-per-prompt Qwen insider upscale run completed, but the graded label distribution was too sparse:

```text
records:      26
report_label: explicit=1, concealed=0
```

Do not extract `insider_trading__upscale` from that file as a direction.

Corrective run paused for cost control:

```text
tmux:     insider_upscale_s20_20260624
run dir:  /dev/shm/insider_upscale_dynamic_20260624_s20_run
output:   /dev/shm/intelligent_liars/data/insider_trading/qwen3-vl-8b-thinking-generations-s20.json
shape:    20 samples per prompt, 520 units total
workers:  16, two per A100
```

The finish watcher was killed before it could grade or launch activation
extraction:

```text
tmux: insider_s20_finish_watch_20260624
log:  /dev/shm/intelligent_liars/logs/insider_s20_finish_watch_20260624.log
```

Pause snapshot after clean freeze:

```text
pending: 114
running: 0
done:    406
failed:  0
outputs: 406
```

Fetched local pause artifacts:

```text
data/insider_trading/qwen3-vl-8b-thinking-generations-s20-partial-20260624T051305Z.json
logs/insider_upscale_s20_20260624.log
logs/insider_upscale_dynamic_s20_20260624/
artifacts/run_snapshots/insider_upscale_dynamic_20260624_s20_run/
```

Partial JSON validation:

```text
records: 406
sha256:  f1646cfd4e0adb84970ba615d9b12a51fccf43d19dc7bb3b7ecca1035bf64224
size:    6361001
```

Resume requirements:

1. Restart a GPU only after using `scripts/run_insider_generation_dynamic.py` or
   an equivalent resume command that does not `rm -rf` the existing run
   directory.
2. Continue from the remote run directory with 114 pending units; do not rerun
   the original launcher as-is because it deletes the queue.
3. Preflight the saved local snapshot before restart with:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_insider_generation_dynamic.py audit \
  --project-root . \
  --run-dir artifacts/run_snapshots/insider_upscale_dynamic_20260624_s20_run \
  --prompt-glob 'data/insider_trading/prompts/**/*.yaml' \
  --samples-per-prompt 20 \
  --label-mode unknown
```

The audit is read-only. It validates planned units, queue files, output shard
shape, shard filename/run-id agreement, duplicate units, missing units, and
`done`/`outputs` consistency. If the audit is clean, a planning pass can then
refresh metadata without launching workers:

Current local audit result:

```text
ok:            true
planned:       520
pending:       114
running:       0
done:          406
failed:        0
outputs:       406
valid_outputs: 406
issues:        none
```

```bash
PYTHONPATH=src .venv/bin/python scripts/run_insider_generation_dynamic.py supervisor \
  --project-root . \
  --run-dir artifacts/run_snapshots/insider_upscale_dynamic_20260624_s20_run \
  --prompt-glob 'data/insider_trading/prompts/**/*.yaml' \
  --samples-per-prompt 20 \
  --label-mode unknown \
  --output data/insider_trading/qwen3-vl-8b-thinking-generations-s20.json \
  --gpus 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7 \
  --require-count 520 \
  --plan-only
```

The preflight should report:

```text
planned=520 queued=0 done=406 pending=114 running=0 failed=0 outputs=406
```

4. On the GPU, use the same script without `--plan-only` and with remote
   `/dev/shm` paths.
5. Merge the full 520-record `qwen3-vl-8b-thinking-generations-s20.json` only
   after pending/running/failed are all zero and outputs/done are 520.
6. Grade `qwen3-vl-8b-thinking-generations-s20.json` with
   `--no-structured-outputs`; the schema-routed OpenRouter path produced
   provider errors.
7. Validate report label counts.
8. Promote the s20 file to
   `data/insider_trading/qwen3-vl-8b-thinking-generations.json` only if it
   contains usable explicit and concealed examples.
9. If concealed examples are still absent or too sparse, extend the dynamic
   generation with more samples instead of extracting a one-sided HDF5.

Promotion gate command:

```bash
PYTHONPATH=src uv run --no-sync python scripts/validate_insider_generation_json.py \
  data/insider_trading/qwen3-vl-8b-thinking-generations-s20.json \
  --project-root . \
  --prompt-glob 'data/insider_trading/prompts/**/*.yaml' \
  --samples-per-prompt 20 \
  --expected-count 520 \
  --require-graded \
  --min-explicit 1 \
  --min-concealed 1 \
  --promote-to data/insider_trading/qwen3-vl-8b-thinking-generations.json \
  --overwrite
```

Only after usable labels exist, extract `insider_trading__upscale` to:

```text
/dev/shm/intelligent_liars/artifacts/activations/activation_insider_upscale_20260624/extracted_feats_insider_upscale_qwen3-vl-8b-thinking.h5
```

## Remaining Execution Order
1. With the GPU stopped, decide whether to resume the same Vast instance or rent
   a cheaper sufficient instance tomorrow.
2. Resume s20 insider generation from the existing 406/114 queue state without
   deleting the run directory.
3. Merge the full 520-record insider s20 JSON, grade it once, and validate label distribution.
4. Extract `insider_trading__upscale` only after the Qwen insider file has usable binary labels.
5. Fetch insider/model-dependent HDF5s to local paths with `*.tmp` + byte-count + sha256 checks.
6. Merge final all-text feature store:

```text
artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5
```

Inputs:

```text
artifacts/activations/activation_rollout4_20260623/extracted_feats_rollout4_qwen3-vl-8b-thinking.h5
artifacts/activations/activation_truthspec_static_20260624/extracted_feats_truthspec_static_qwen3-vl-8b-thinking.h5
artifacts/activations/activation_truthspec_sycophancy_20260624/extracted_feats_truthspec_sycophancy_qwen3-vl-8b-thinking.h5
artifacts/activations/activation_insider_upscale_20260624/extracted_feats_insider_upscale_qwen3-vl-8b-thinking.h5
```

7. DVC add/push the static, fresh sycophancy, insider upscale, and final all-text artifacts.
8. Train probes only after the final all-text HDF5 validates and DVC push succeeds.

## Pre-Staged Commands

### 1) High-throughput extraction merge on GPU side

```bash
PYTHONPATH=src .venv/bin/python scripts/run_activation_extraction_dynamic.py supervisor \
  --project-root /workspace/intelligent_liars \
  --run-dir /dev/shm/intelligent_liars_activation/runs/<run-name> \
  --path data/rollouts/... \
  --output /dev/shm/intelligent_liars_activation/results/<merged>.h5 \
  --gpus 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7 \
  --layers all \
  --batch-size 1 \
  --compression lzf \
  --storage-dtype float16 \
  --overwrite-queue \
  --overwrite-output

PYTHONPATH=src .venv/bin/python src/intelligent_liars/cli.py merge-activation-shards \
  /dev/shm/intelligent_liars_activation/outputs/*.h5 \
  --output /dev/shm/intelligent_liars_activation/results/<final_merged>.h5 \
  --compression lzf
```

### 2) Sycophancy extraction command (Qwen-local model-dependent)

```bash
PYTHONPATH=src .venv/bin/python src/intelligent_liars/cli.py generate-sycophancy \
  --project-root . \
  --output-dir data/sycophancy/qwen3-vl-8b-thinking \
  --task abstract_algebra anatomy \
  --generated-model qwen3-vl-8b-thinking

PYTHONPATH=src .venv/bin/python src/intelligent_liars/cli.py extract-activations \
  --project-root . \
  --task sycophancy__mmlu_stem_same_conf_all \
  --output artifacts/activations/activation_truthspec_sycophancy_20260624/extracted_feats_truthspec_sycophancy_qwen3-vl-8b-thinking.h5 \
  --layers all \
  --compression lzf
```

### 3) Insider upscale extraction command (after usable grade file exists)

```bash
PYTHONPATH=src .venv/bin/python src/intelligent_liars/cli.py extract-activations \
  --project-root . \
  --task insider_trading__upscale \
  --output artifacts/activations/activation_insider_upscale_20260624/extracted_feats_insider_upscale_qwen3-vl-8b-thinking.h5 \
  --layers all \
  --compression lzf
```

### 3a) Remote resume preflight for insider s20

Preferred remote entrypoint after the GPU is reachable and the saved run
snapshot exists under
`/workspace/intelligent_liars/artifacts/insider_upscale_dynamic_20260624_s20_run`:

```bash
cd /workspace/intelligent_liars
uv run --no-sync python scripts/render_vast_insider_resume.py \
  --write /tmp/insider_s20_resume.sh \
  --overwrite
bash /tmp/insider_s20_resume.sh
```

This generated script performs the environment preflight, refuses duplicate
tmux writers, runs the read-only queue audit, asserts the expected 520 planned /
406 done / 114 pending / 0 running / 0 failed / 406 valid-output state, prefetches
Qwen into `/dev/shm/hf_home`, and only then starts the 16-slot tmux resume.

Manual fallback details:

Before launching workers on the 8x A100 instance, run the environment preflight
inside the remote project checkout:

```bash
cd /workspace/intelligent_liars
export HF_HOME=/dev/shm/hf_home
export TMPDIR=/dev/shm/tmp
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=src
mkdir -p "$HF_HOME" "$TMPDIR"

uv run --no-sync python scripts/preflight_vast_gpu_environment.py \
  --project-root /workspace/intelligent_liars \
  --expected-gpus 8 \
  --min-gpu-memory-mib 80000 \
  --min-dev-shm-free-gb 120 \
  --require-env-file
```

Then prefetch the model once into `/dev/shm/hf_home` before starting the 16
workers:

```bash
uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-VL-8B-Thinking")
print("PREFETCH_DONE")
PY
```

Run a read-only queue audit before any mutating queue repair or worker launch:

```bash
uv run --no-sync python scripts/run_insider_generation_dynamic.py audit \
  --project-root /workspace/intelligent_liars \
  --run-dir /workspace/intelligent_liars/artifacts/insider_upscale_dynamic_20260624_s20_run \
  --prompt-glob 'data/insider_trading/prompts/**/*.yaml' \
  --samples-per-prompt 20 \
  --label-mode unknown
```

Expected resume state:

```text
ok=true planned=520 done=406 pending=114 running=0 failed=0 outputs=406 valid_outputs=406
```

Only after those checks pass, launch the guarded resume in tmux:

```bash
tmux new-session -d -s insider_upscale_s20_resume 'cd /workspace/intelligent_liars && \
  export HF_HOME=/dev/shm/hf_home TMPDIR=/dev/shm/tmp HF_HUB_DISABLE_XET=1 PYTHONPATH=src && \
  uv run --no-sync python scripts/run_insider_generation_dynamic.py supervisor \
    --project-root /workspace/intelligent_liars \
    --run-dir /workspace/intelligent_liars/artifacts/insider_upscale_dynamic_20260624_s20_run \
    --prompt-glob '"'"'data/insider_trading/prompts/**/*.yaml'"'"' \
    --samples-per-prompt 20 \
    --label-mode unknown \
    --output /dev/shm/intelligent_liars/data/insider_trading/qwen3-vl-8b-thinking-generations-s20.json \
    --gpus 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7 \
    --require-count 520 \
    --poll-seconds 30'
```

### 3b) Remote finish script after insider s20 reaches 520/520

Preferred post-generation entrypoint after the resume script finishes and
`run_insider_generation_dynamic.py audit` reports `done=520`, `pending=0`,
`running=0`, `failed=0`, and `valid_outputs=520`:

```bash
cd /workspace/intelligent_liars
uv run --no-sync python scripts/render_vast_insider_finish.py \
  --write /tmp/insider_s20_finish.sh \
  --overwrite
bash /tmp/insider_s20_finish.sh
```

The generated finish script:

1. Re-runs the remote environment preflight.
2. Asserts queue completion before any grading or extraction.
3. Validates the full s20 JSON count/run-id coverage before judge calls.
4. Runs `grade-insider-trading` with `--no-structured-outputs` and
   `--no-require-structured-outputs`.
5. Promotes `qwen3-vl-8b-thinking-generations.json` only if the graded validator
   passes `--require-graded --min-explicit 1 --min-concealed 1`.
6. Refuses to overwrite an existing insider-upscale activation HDF5 unless the
   renderer is explicitly called with `--overwrite-activation`.
7. Extracts `insider_trading__upscale` with `--compression lzf`.
8. Validates the HDF5 with task count 1, task name `insider_trading__upscale`,
   36 layers, hidden dim 4096, finite samples, and sha256.

### 4) Final all-text merge command

First preflight exact task coverage and input HDF5 compatibility:

```bash
EXPECTED_TASKS='claims__definitional_gemini_600_full,claims__evidential_gemini_600_full,claims__fictional_gemini_600_full,claims__logical_gemini_600_full,ethics__commonsense,geometry_of_truth__best,geometry_of_truth__mixed,insider_trading__onpolicy,insider_trading__upscale,insider_trading_doubledown__onpolicy,internal_state__animals,internal_state__cities,internal_state__companies,internal_state__elements,internal_state__facts,internal_state__inventions,repe_honesty__IF_dishonest,roleplaying__plain,sandbagging_v2__wmdp_mmlu,sycophancy__mmlu_stem_same_conf_all'

PYTHONPATH=src .venv/bin/python scripts/plan_activation_hdf5_merge.py \
  artifacts/activations/activation_rollout4_20260623/extracted_feats_rollout4_qwen3-vl-8b-thinking.h5 \
  artifacts/activations/activation_truthspec_static_20260624/extracted_feats_truthspec_static_qwen3-vl-8b-thinking.h5 \
  artifacts/activations/activation_truthspec_sycophancy_20260624/extracted_feats_truthspec_sycophancy_qwen3-vl-8b-thinking.h5 \
  artifacts/activations/activation_insider_upscale_20260624/extracted_feats_insider_upscale_qwen3-vl-8b-thinking.h5 \
  --validate-inputs \
  --expected-layer-count 36 \
  --expected-hidden-dim 4096 \
  --finite-check none \
  --expected-tasks "$EXPECTED_TASKS"
```

Only merge after the planner returns `ok=true`; this catches missing tasks,
duplicate tasks, incompatible layer counts, hidden-dim mismatches, and invalid
per-input HDF5 structure before spending time on the merge:

```bash
PYTHONPATH=src .venv/bin/python src/intelligent_liars/cli.py merge-activation-shards \
  artifacts/activations/activation_rollout4_20260623/extracted_feats_rollout4_qwen3-vl-8b-thinking.h5 \
  artifacts/activations/activation_truthspec_static_20260624/extracted_feats_truthspec_static_qwen3-vl-8b-thinking.h5 \
  artifacts/activations/activation_truthspec_sycophancy_20260624/extracted_feats_truthspec_sycophancy_qwen3-vl-8b-thinking.h5 \
  artifacts/activations/activation_insider_upscale_20260624/extracted_feats_insider_upscale_qwen3-vl-8b-thinking.h5 \
  --output artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5 \
  --merge-strategy copy-disjoint \
  --compression lzf
```

With `--merge-strategy copy-disjoint`, source datasets are copied into the final
HDF5 without recompression; `--compression lzf` is retained only as the safe
concat-mode setting if the command is changed later.

### 5) Post-artifact validation and DVC-safe promotion

```bash
.venv/bin/python scripts/validate_activation_hdf5.py \
  artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5 \
  --expected-task-count 20 \
  --expected-tasks claims__definitional_gemini_600_full,claims__evidential_gemini_600_full,claims__fictional_gemini_600_full,claims__logical_gemini_600_full,ethics__commonsense,geometry_of_truth__best,geometry_of_truth__mixed,insider_trading__onpolicy,insider_trading__upscale,insider_trading_doubledown__onpolicy,internal_state__animals,internal_state__cities,internal_state__companies,internal_state__elements,internal_state__facts,internal_state__inventions,repe_honesty__IF_dishonest,roleplaying__plain,sandbagging_v2__wmdp_mmlu,sycophancy__mmlu_stem_same_conf_all \
  --expected-layer-count 36 \
  --expected-hidden-dim 4096 \
  --finite-check sample \
  --finite-sample-rows 64 \
  --sha256

.venv/bin/python scripts/safe_fetch_artifact.py dvc-flow artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5
uvx --from "dvc[gdrive]" dvc add artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5
uvx --from "dvc[gdrive]" dvc push artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5.dvc
```

## Current DVC State
The static HDF5 DVC pointer exists locally:

```text
artifacts/activations/activation_truthspec_static_20260624/extracted_feats_truthspec_static_qwen3-vl-8b-thinking.h5.dvc
```

The static DVC push is not complete. It failed with TLS EOF after a long first
attempt, and a `--jobs 1` retry failed immediately with the same TLS EOF. Before
retrying, confirm no DVC process is alive:

```bash
ps -axo pid,etime,command | grep -E 'dvc push|dvc add' | grep -v grep || true
uvx --from 'dvc[gdrive]' dvc status -c artifacts/activations/activation_truthspec_static_20260624/extracted_feats_truthspec_static_qwen3-vl-8b-thinking.h5.dvc
```

Current `dvc status -c` result:

```text
new: artifacts/activations/activation_truthspec_static_20260624/extracted_feats_truthspec_static_qwen3-vl-8b-thinking.h5
```

The next DVC action is retrying the static push when the Drive transport is
stable:

```bash
uvx --from 'dvc[gdrive]' dvc push --jobs 1 \
  artifacts/activations/activation_truthspec_static_20260624/extracted_feats_truthspec_static_qwen3-vl-8b-thinking.h5.dvc
```

After that succeeds, add/push the fresh sycophancy artifact:

```bash
uvx --from 'dvc[gdrive]' dvc add \
  artifacts/activations/activation_truthspec_sycophancy_20260624/extracted_feats_truthspec_sycophancy_qwen3-vl-8b-thinking.h5
uvx --from 'dvc[gdrive]' dvc push \
  artifacts/activations/activation_truthspec_sycophancy_20260624/extracted_feats_truthspec_sycophancy_qwen3-vl-8b-thinking.h5.dvc
```

## Probe Blocker
Probe training is still explicitly blocked.

The blocker clears only when:

```text
rollout4 HDF5:          validated local
static HDF5:            validated local
sycophancy HDF5:        validated local
insider upscale HDF5:   validated local with usable labels
final all-text HDF5:    merged and validated local
DVC push:               complete
```

## Current Completion Update 2026-06-25T01:32:49Z

Insider-upscale generation and extraction are complete:

```text
generation JSON: data/insider_trading/qwen3-vl-8b-thinking-generations.json
generation records: 520
generation sha256: f36b28dfe0de3bfbb34aeae5f63e5220eedd48e94f6f10f04cc52a7c3cb6b86d
graded status: ok=true; ungraded=0; explicit=3; concealed=3; implied skipped=3; invalid report labels terminal-skipped=3

activation HDF5: artifacts/activations/activation_insider_upscale_20260624/extracted_feats_insider_upscale_qwen3-vl-8b-thinking.h5
activation sha256: 8221187118110eac12db342754e133f91be708035bbd249fe9bb3c23a545786f
activation validation: ok=true; task=insider_trading__upscale; examples=6; token_rows=489; labels 0=249 / 1=240; 36 layers; hidden_dim=4096; finite sample clean
```

Final all-text merge was rebuilt after catching a sycophancy input mismatch. The
intended final input is the fresh `activation_truthspec_sycophancy_20260624`
artifact, not the older DVC-tracked `activation_sycophancy_20260624` file. The
two have identical labels/source indices but different activation values, so the
first all-text merge was discarded and overwritten.

Corrected final all-text artifact:

```text
final HDF5: artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5
final sha256: c6e687f69256544121d0718cf8bba142ed5837221976121e1899552dc76f1a5a
final validation: ok=true; 20 tasks; 36 layers; hidden_dim=4096; 720 sampled activation datasets finite; merged_strategy=copy-disjoint
final DVC md5: b6e82b698513d2372949f2752e17005a
final DVC size: 62394047728
```

Current DVC pointers:

```text
artifacts/activations/activation_truthspec_sycophancy_20260624/extracted_feats_truthspec_sycophancy_qwen3-vl-8b-thinking.h5.dvc
artifacts/activations/activation_insider_upscale_20260624/extracted_feats_insider_upscale_qwen3-vl-8b-thinking.h5.dvc
artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5.dvc
data/insider_trading/qwen3-vl-8b-thinking-generations.json.dvc
```

DVC remote status immediately before the targeted push:

```text
new: artifacts/activations/activation_truthspec_sycophancy_20260624/extracted_feats_truthspec_sycophancy_qwen3-vl-8b-thinking.h5
new: artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5
```

Targeted DVC push is in progress:

```bash
uvx --from "dvc[gdrive]" dvc push --jobs 1 \
  artifacts/activations/activation_truthspec_sycophancy_20260624/extracted_feats_truthspec_sycophancy_qwen3-vl-8b-thinking.h5.dvc \
  artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5.dvc
```

Probe training remains blocked until that DVC push succeeds.

2026-06-25T02:49:43Z DVC push status:

```text
status: still running; no error output
uvx pid: 76682
dvc pid: 76683
elapsed: about 1h46m
network: established Google HTTPS connection; outbound bytes still increasing
note: do not run another DVC command while this owns the DVC lock
```

2026-06-25T05:02:15Z DVC push status:

```text
status: still running; no terminal output yet
uvx pid: 76682
dvc pid: 76683
elapsed: about 3h59m
network: established Google HTTPS connection; outbound bytes still increasing
note: leave this process running; do not start another DVC command while it owns the lock
```

2026-06-25T06:00:41Z DVC/GPU status:

```text
previous targeted DVC push result: failed after about 4h with SSL EOF while uploading final all-text object b6e82b698513d2372949f2752e17005a
remote DVC status after failure:  only final all-text HDF5 is new; smaller sycophancy/insider artifacts are pushed
Vast instance restarted by user:   41988240 at ssh -p 60232 root@159.48.242.25
remote GPU state on reconnect:     8x A100 idle, 0% utilization, no activation/merge job running
remote artifact check:             no final all-text HDF5 on remote; only insider-upscale HDF5 present and already fetched/pushed
action taken:                      stopped Vast instance 41988240 from inside container with vastai stop instance; SSH port then refused
current blocker:                   local DVC/GDrive upload of final all-text HDF5 only
current retry command:             uvx --from "dvc[gdrive]" dvc push -v --jobs 1 artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5.dvc
current retry log:                 logs/dvc_push_final_all_text_20260625T055934Z.log
current retry pid:                 38880
state:                             DVC process is running and writing to Google HTTPS; no GPU is needed
```

2026-06-25T06:16:42Z upload handoff:

```text
DVC retry:                         interrupted intentionally after slow early throughput; DVC status still reports only final all-text as new
rclone fallback:                   installed temporarily at /tmp/intelligent_liars_rclone/rclone
rclone credential config:          temporary /tmp/intelligent_liars_rclone/rclone.conf from existing DVC OAuth credentials
target remote object:              dvc-gdrive:files/md5/b6/e82b698513d2372949f2752e17005a
target local object:               .dvc/cache/files/md5/b6/e82b698513d2372949f2752e17005a
screen session:                    il_rclone_final_20260625T061513Z
rclone pid:                        61413
rclone log:                        logs/rclone_upload_final_all_text_screen_20260625T061513Z.log
rclone exit file:                  logs/rclone_upload_final_all_text_screen_20260625T061513Z.exit
first stats line:                  121.746 MiB / 58.109 GiB, 2.375 MiB/s, ETA 6h56m43s
next verification:                 after rclone exits with code 0, run DVC status -c for the final pointer; it must no longer report the final all-text HDF5 as new
```

2026-06-25T06:19:41Z upload monitor:

```text
rclone state:                      still running in screen session il_rclone_final_20260625T061513Z
latest stats line:                 603.465 MiB / 58.109 GiB, 1%, 3.110 MiB/s, ETA 5h15m40s
sleep guard:                       caffeinate -dims -w 61413
caffeinate pid:                    68666
GPU state:                         ssh -p 60232 root@159.48.242.25 still refuses; no GPU billing endpoint reachable from this SSH URL
```

2026-06-25T06:22:04Z upload monitor:

```text
rclone state:                      still running in screen session il_rclone_final_20260625T061513Z
latest stats line:                 852.371 MiB / 58.109 GiB, 1%, 2.142 MiB/s, ETA 7h36m22s
sleep guard correction:            earlier shell-launched caffeinate exited; relaunched as durable screen session
sleep guard screen session:        il_caffeinate_rclone_20260625T062200Z
sleep guard process:               caffeinate -dims -w 61413
GPU state:                         ssh -p 60232 root@159.48.242.25 still refuses
probe status:                      no current probe process observed; existing artifacts/probes predates this final all-text phase and was not touched
```

2026-06-25T06:23:26Z upload monitor:

```text
rclone state:                      still running in screen session il_rclone_final_20260625T061513Z
latest stats line:                 1.087 GiB / 58.109 GiB, 2%, 2.323 MiB/s, ETA 6h58m51s
sleep guard state:                 il_caffeinate_rclone_20260625T062200Z still running; caffeinate -dims -w 61413 alive
GPU state:                         ssh -p 60232 root@159.48.242.25 still refuses
probe status:                      no probe/training process observed
next gate:                         wait for rclone exit file, then run DVC status -c on the final all-text pointer
```

2026-06-25T06:40:22Z upload/GPU monitor:

```text
rclone state:                      still running in screen session il_rclone_final_20260625T061513Z
latest stats line:                 3.349 GiB / 58.109 GiB, 6%, 2.146 MiB/s, ETA 7h15m26s
sleep guard state:                 il_caffeinate_rclone_20260625T062200Z still running; caffeinate -dims -w 61413 alive
GPU state:                         12 consecutive nc checks to 159.48.242.25:60232 from 06:39:05Z through 06:40:01Z returned connection refused
local Vast API state:              no local Vast credential exposed; vastai show instances returns 403 login required
final HDF5 validation:             ok=true; 20 tasks; 36 layers; hidden_dim=4096; 720 sampled activation datasets finite; merged_strategy=copy-disjoint
insider JSON validation:           ok=true; 520 records; ungraded=0; explicit=3; concealed=3; binary_report_usable=6
probe status:                      no probe/training process observed
cost guidance:                     no non-probe GPU work remains locally; if the Vast instance is billing, stop/repause it from the Vast UI until a new concrete GPU workload exists
next gate:                         wait for rclone exit file, then run DVC status -c on the final pointer; do not start another DVC command during upload
```

2026-06-25T06:43:58Z post-upload verifier:

```text
rclone state:                      still running in screen session il_rclone_final_20260625T061513Z
latest stats line:                 3.664 GiB / 58.109 GiB, 6%, 1.498 MiB/s, ETA 10h20m19s
sleep guard state:                 il_caffeinate_rclone_20260625T062200Z still running
superseded verifier:               il_final_upload_postverify_20260625T064232Z was stopped before rclone exit because its wrapper might not write an exit marker
post-upload verifier session:      il_final_upload_postverify_20260625T064358Z
post-upload verifier log:          logs/final_upload_postverify_20260625T064358Z.log
post-upload verifier exit file:    logs/final_upload_postverify_20260625T064358Z.exit
verifier behavior:                 waits for logs/rclone_upload_final_all_text_screen_20260625T061513Z.exit; only if rclone exits 0, then runs DVC status -c for final/sycophancy/insider-upscale/insider-json pointers and final all-text HDF5 validation, writes verifier_status to log, and writes an explicit exit marker
GPU state:                         ssh -p 60232 root@159.48.242.25 still refused at the last check
probe status:                      no probe/training process observed
next gate:                         inspect verifier exit/log after rclone exits; do not run concurrent DVC commands while verifier is active
```

2026-06-25T06:45:43Z verifier sleep guard:

```text
rclone state:                      still running in screen session il_rclone_final_20260625T061513Z
latest stats line:                 3.838 GiB / 58.109 GiB, 7%, 1.466 MiB/s, ETA 10h31m41s
rclone sleep guard:                il_caffeinate_rclone_20260625T062200Z; caffeinate -dims -w 61413
post-upload verifier session:      il_final_upload_postverify_20260625T064358Z
post-upload verifier pid:          41579
post-upload verifier sleep guard:  il_caffeinate_postverify_20260625T064543Z; caffeinate -dims -w 41579
reason:                            rclone's sleep guard exits when upload exits; verifier has a separate guard so DVC status and final HDF5 validation can finish without the machine sleeping
next gate:                         wait for rclone/verifier exit markers; inspect logs before running any manual DVC command
```

2026-06-25T06:47:05Z upload monitor:

```text
rclone state:                      still running in screen session il_rclone_final_20260625T061513Z
latest stats line:                 3.932 GiB / 58.109 GiB, 7%, 1.660 MiB/s, ETA 9h16m52s
rclone exit marker:                absent
post-upload verifier state:        il_final_upload_postverify_20260625T064358Z still waiting for the rclone exit marker
post-upload verifier exit marker:  absent
sleep guards:                      il_caffeinate_rclone_20260625T062200Z and il_caffeinate_postverify_20260625T064543Z both alive
GPU state:                         ssh -p 60232 root@159.48.242.25 still refused
probe status:                      no probe/training process observed
next gate:                         wait for rclone/verifier exit markers; do not run concurrent DVC commands
```

2026-06-25T06:24:52Z upload monitor:

```text
rclone state:                      still running in screen session il_rclone_final_20260625T061513Z
latest stats line:                 1.212 GiB / 58.109 GiB, 2%, 2.120 MiB/s, ETA 7h38m4s
sleep guard state:                 il_caffeinate_rclone_20260625T062200Z still running; caffeinate -dims -w 61413 alive
GPU state:                         ssh -p 60232 root@159.48.242.25 still refuses
probe status:                      no probe/training process observed
next gate:                         wait for rclone exit file, then run DVC status -c on the final all-text pointer
```

2026-06-25T06:26:21Z upload monitor:

```text
rclone state:                      still running in screen session il_rclone_final_20260625T061513Z
latest stats line:                 1.497 GiB / 58.109 GiB, 3%, 2.418 MiB/s, ETA 6h39m33s
sleep guard state:                 il_caffeinate_rclone_20260625T062200Z still running; caffeinate -dims -w 61413 alive
GPU state:                         ssh -p 60232 root@159.48.242.25 still refuses
probe status:                      no probe/training process observed
next gate:                         wait for rclone exit file, then run DVC status -c on the final all-text pointer
```

2026-06-25T06:27:48Z upload monitor:

```text
rclone state:                      still running in screen session il_rclone_final_20260625T061513Z
latest stats line:                 1.638 GiB / 58.109 GiB, 3%, 2.399 MiB/s, ETA 6h41m45s
sleep guard state:                 il_caffeinate_rclone_20260625T062200Z still running; caffeinate -dims -w 61413 alive
GPU state:                         ssh -p 60232 root@159.48.242.25 still refuses
probe status:                      no probe/training process observed
next gate:                         wait for rclone exit file, then run DVC status -c on the final all-text pointer
```

2026-06-25T06:29:09Z upload monitor:

```text
rclone state:                      still running in screen session il_rclone_final_20260625T061513Z
latest stats line:                 1.765 GiB / 58.109 GiB, 3%, 2.312 MiB/s, ETA 6h55m58s
sleep guard state:                 il_caffeinate_rclone_20260625T062200Z still running; caffeinate -dims -w 61413 alive
GPU state:                         ssh -p 60232 root@159.48.242.25 still refuses
probe status:                      no probe/training process observed
next gate:                         wait for rclone exit file, then run DVC status -c on the final all-text pointer
```

2026-06-25T06:57:14Z hourly monitor policy:

```text
rationale:                         user correctly requested hourly checks to avoid context pollution
rclone state:                      still running in screen session il_rclone_final_20260625T061513Z
latest stats line:                 5.043 GiB / 58.109 GiB, 9%, 2.232 MiB/s, ETA 6h45m45s
rclone exit marker:                absent
superseded verifier:               il_final_upload_postverify_20260625T064358Z stopped before DVC ran; it used a 300s polling loop
superseded verifier sleep guard:   il_caffeinate_postverify_20260625T064543Z stopped
hourly verifier session:           il_final_upload_hourly_postverify_20260625T065649Z
hourly verifier log:               logs/final_upload_hourly_postverify_20260625T065649Z.log
hourly verifier exit file:         logs/final_upload_hourly_postverify_20260625T065649Z.exit
hourly verifier poll interval:     3600 seconds
hourly verifier sleep guard:       il_caffeinate_hourly_postverify_20260625T065649Z; caffeinate -dims -w 92399
behavior:                          do not manually poll in chat more than hourly unless an exit marker exists or the user explicitly asks; do not run concurrent DVC commands while verifier is active
probe status:                      no DVC/probe/training process observed during replacement
```
