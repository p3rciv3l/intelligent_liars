# Activation Extraction Postmortem

Date: 2026-06-22

Artifact root: `artifacts/activations/activation_full_20260622`

This postmortem covers the full Qwen3-VL activation extraction run for the judged rollout data. The run succeeded and the final HDF5 is local and validated, but there were several operational lessons worth preserving before the next activation/probe pass.

## Final State

The final local artifact is:

```text
artifacts/activations/activation_full_20260622/extracted_feats_all_layers_qwen3-vl-8b-thinking.h5
```

The Google Drive folder for the external artifact is:

```text
https://drive.google.com/drive/folders/1C7KUkZHCfQXnpkt6debhCcA4YtV8eOHX
```

The full HDF5 is intentionally not committed to GitHub. Keep the binary in Google Drive or local ignored storage, and keep only metadata/logs in the repo.

Validation summary:

```text
size:              11514713158 bytes
sha256:            c9534056c1d6c81d6c82cdf468e890d3d66072507f5d728e8f07b2cfe39e9fb4
dvc md5:           6d3747827c3094eca02e2ac74a8426c2
dvc remote:        gdrive-artifacts
tasks:             roleplaying__plain, sandbagging_v2__wmdp_mmlu
examples:          457 total
answer-token rows: 50375 total
layers:            36
storage dtype:     float16
failed chunks:     0
format:            qwen_answer_token_activations_v2
model:             Qwen/Qwen3-VL-8B-Thinking
```

The DVC pointer is:

```text
artifacts/activations/activation_full_20260622/extracted_feats_all_layers_qwen3-vl-8b-thinking.h5.dvc
```

The remote backup copy remained at:

```text
/dev/shm/intelligent_liars_activation/results/extracted_feats_all_layers_qwen3-vl-8b-thinking.h5
```

It is no longer required for correctness because the local file was transferred cleanly and validated locally. It is only useful as a temporary backup until the Vast instance is destroyed.

## What Worked

The dynamic activation scheduler was the right shape. It started persistent workers on all visible GPUs, each worker loaded Qwen once, and workers pulled small HDF5 extraction chunks from a shared queue. This avoided the rollout failure mode where one static shard gets stuck while other GPUs go idle.

For activation extraction, one worker per A100 was enough. Unlike long autoregressive generation, extraction is a deterministic forward-pass workload over already-generated transcripts. Once the model was loaded, all 61 chunks finished in about 3 minutes.

The queue should plan the largest chunks first. The first version sorted chunks in memory but wrote queue filenames without preserving that order, so filesystem ordering could defeat the priority. The final version prefixes pending filenames with a queue index, so large sandbagging chunks actually start first.

The chunk estimator should consider full rendered context length, not only answer length. Some examples have short final answers but long prompts or reasoning transcripts. Prioritizing by answer text alone underestimates runtime and memory risk.

Writing activations as `float16` is the right default for this pass. It avoids bf16-to-NumPy/HDF5 incompatibilities and keeps the all-layer output around 11 GB instead of roughly doubling it.

## What Failed

The first full run wrote shards and merge output under `/workspace`, which was a small overlay filesystem on this Vast image. That run failed with `ENOSPC`.

Future rule: large model artifacts, HDF5 shards, and merged HDF5 outputs should go under a large mounted volume or `/dev/shm` after checking `df -h`. Do not assume `/workspace` has enough disk.

The final merge was much slower than extraction. Extraction finished in about 3 minutes, while gzip recompression during merge took about 15 minutes. That is acceptable for this run but should not be ignored when artifacts get larger.

Future rule: expose a compression option for activation writes and merge writes. `gzip` is compact but slow. For iteration runs, prefer no compression or `lzf`; for archival runs, use `gzip`.

The first transfer back to local was interrupted and left a partial HDF5. A later manual append attempt produced an oversized corrupt file, likely because the earlier transfer process had continued writing longer than expected.

Future rule: never resume large HDF5 copies by appending unless all writers are proven dead and the prefix is verified. Prefer copying into `*.tmp`, checking exact remote/local byte counts, then atomically moving into place.

## Warnings To Clean Up

Worker logs repeatedly contain:

```text
Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`
```

This warning did not break extraction, and the HDF5 validated structurally. Still, the processor call should be updated before the next serious run so logs stay meaningful and future Transformers versions do not turn this into an error.

The remote environment used Python 3.12 while the project expects Python 3.11, so `uv run --no-sync` emitted compatibility warnings. This did not affect the run, but the next clean remote setup should use a Python 3.11 project env or intentionally update the project pin.

## Scope Limitations

This extraction includes only the currently binary-usable judged rollout examples:

```text
roleplaying__plain:        329 examples
sandbagging_v2__wmdp_mmlu: 128 examples
```

The insider rollout files were excluded because their labels were not usable under the extractor's binary-label logic at extraction time. Later judging those rollouts does not backfill this HDF5; using insider data for probe training requires a separate activation extraction pass over the now-labeled insider tasks.

This HDF5 is not yet a trained truth direction. It is the activation table needed for the next step.

## Recommended Next Steps

1. Commit the activation extraction code and tests, not the ignored HDF5 artifacts.
2. Destroy the Vast instance unless another immediate GPU job will use it. Billing continues while it is running, and the local HDF5 has already validated.
3. Fix the processor warning in `activations.py`.
4. Use the probe results to choose candidate layers before building steering hooks.
5. Run a separate GPU activation extraction for the now-labeled insider tasks if insider directions should be represented.
6. Add a smaller compression knob before the next full extraction run if iteration speed matters.

## Next Activation Target

The existing all-layer HDF5 contains only:

```text
roleplaying__plain
sandbagging_v2__wmdp_mmlu
```

The next GPU job should extract an insider-only HDF5 from the already-generated
and graded rollout files:

```text
data/rollouts/insider_trading__onpolicy__qwen3-vl-8b-thinking.json
data/rollouts/insider_trading_doubledown__onpolicy__qwen3-vl-8b-thinking.json
```

Loader preflight on 2026-06-23:

```text
insider_trading__onpolicy: 173 examples, 117 binary usable, labels={honest:107, deceptive:10, skip:56}
insider_trading_doubledown__onpolicy: 91 examples, 90 binary usable, labels={honest:27, deceptive:63, skip:1}
```

On an 8x A100 80GB box, use the dynamic runner rather than the direct single
process CLI:

```bash
mkdir -p /dev/shm/intelligent_liars_activation/runs /dev/shm/intelligent_liars_activation/results logs
tmux new-session -d -s insider_activation 'cd /workspace/intelligent_liars && \
  PYTHONPATH=src HF_HOME=/dev/shm/hf_home TMPDIR=/dev/shm/tmp HF_HUB_DISABLE_XET=1 \
  uv run --no-sync python scripts/run_activation_extraction_dynamic.py supervisor \
    --project-root /workspace/intelligent_liars \
    --run-dir /dev/shm/intelligent_liars_activation/runs/insider-activation-$(date -u +%Y%m%dT%H%M%SZ) \
    --path data/rollouts/insider_trading__onpolicy__qwen3-vl-8b-thinking.json \
    --path data/rollouts/insider_trading_doubledown__onpolicy__qwen3-vl-8b-thinking.json \
    --output /dev/shm/intelligent_liars_activation/results/extracted_feats_insider_qwen3-vl-8b-thinking.h5 \
    --gpus 0,1,2,3,4,5,6,7 \
    --layers all \
    --batch-size 1 \
    --chunk-chars 40000 \
    --max-examples-per-chunk 8 \
    --storage-dtype float16 \
    --overwrite-queue \
    --overwrite-output \
    2>&1 | tee logs/insider-activation.latest.log'
```

Expected plan shape with those settings is about 37 chunks. The output should
be a fresh HDF5; do not append to the validated 11 GB roleplaying/sandbagging
HDF5.

## Re-run Shape

The successful run shape was:

```text
gpus:                  0,1,2,3,4,5,6,7
workers:               8
layers:                all
batch size:            1
chunk chars:           40000
max examples/chunk:    8
planned units:         61
capture logits:        false
storage dtype:         float16
remote output storage: /dev/shm
remote project root:   /workspace/intelligent_liars_activation
remote run dir:        /dev/shm/intelligent_liars_activation/runs/full-run-20260622T061750Z/run
input paths:           data/rollouts/roleplaying__plain__qwen3-vl-8b-thinking.json
                       data/rollouts/sandbagging_v2__wmdp_mmlu__qwen3-vl-8b-thinking.json
```

The key operational invariant is to keep shards independent and merge only after all chunks finish. Each shard writes its own HDF5, and the final merge rejects duplicate `(source_index, output_index)` examples.
