# TinyLoRA model cache

This cache removes repeated model downloads without turning a temporary GPU host
into the only copy of the model. It is intentionally narrow, immutable, and
checksum-addressed.

## What is pinned

- Model: `Qwen/Qwen3-VL-8B-Thinking`
- Exact revision: `92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b`
- Selected runtime files: 14
- Expected size at that revision: 17,545,907,058 bytes (16.34 GiB)

The selection contains the four safetensor shards, their index, model and
generation configuration, tokenizer files, chat template, and image/video
preprocessor configuration. It excludes the README and Git metadata. Exact
filenames are code-owned in `REQUIRED_SNAPSHOT_FILES`; wildcards are not used.
The pinned upstream model card and Apache-2.0 license are published beside the
loader tree as legal/attribution objects, so they do not expand or contaminate
the exact runtime inventory.

This is not a quantized or altered model. It is a byte-for-byte narrow snapshot
of the approved Hugging Face revision.

## The three states

1. **Source plan.** The Hub metadata endpoint is queried at the exact commit.
   This transfers metadata, not weight blobs. The plan records every expected
   byte count and Hugging Face source object identifier.
2. **Verified local snapshot.** Every required local file must be a regular,
   self-contained file. The verifier rejects missing files, extra files,
   symlinks, wrong sizes, wrong LFS SHA-256 values, and wrong Git blob IDs. It
   then records a local SHA-256 for every file.
3. **Published S3 cache.** The verified file inventory produces a content
   SHA-256 and therefore an immutable S3 prefix. Publication is not implemented
   by this tool; a separate operations step must copy files and publish the
   completion marker last.

Planning and verification do not rent a GPU. The implementation completed here
did not download model weights or upload anything.

## Commands

Create the source plan without downloading weights:

```bash
uv run python scripts/build_tinylora_model_cache.py plan \
  --output artifacts/model-cache/qwen3-vl-8b-thinking.plan.json
```

Verify an already-present snapshot and prepare S3 metadata:

```bash
uv run python scripts/build_tinylora_model_cache.py verify \
  --plan artifacts/model-cache/qwen3-vl-8b-thinking.plan.json \
  --snapshot-dir /mnt/model-cache/qwen3-vl-8b-thinking \
  --manifest-output artifacts/model-cache/qwen3-vl-8b-thinking.manifest.json \
  --completion-output artifacts/model-cache/qwen3-vl-8b-thinking.complete.json \
  --s3-bucket YOUR_PROJECT_BUCKET
```

Explicitly download the narrow snapshot, verify it, and prepare metadata:

```bash
uv run python scripts/build_tinylora_model_cache.py download \
  --execute-download \
  --snapshot-dir /mnt/model-cache/qwen3-vl-8b-thinking \
  --plan-output artifacts/model-cache/qwen3-vl-8b-thinking.plan.json \
  --manifest-output artifacts/model-cache/qwen3-vl-8b-thinking.manifest.json \
  --completion-output artifacts/model-cache/qwen3-vl-8b-thinking.complete.json \
  --s3-bucket YOUR_PROJECT_BUCKET
```

`download` is inert without `--execute-download`. It also checks that free disk
space exceeds the expected snapshot by at least 2 GiB. Plan and manifest outputs
must live outside the snapshot directory so the runtime inventory stays exact.

Hydrate the cache layout that the existing `ModelLoadConfig` understands:

```bash
uv run python scripts/build_tinylora_model_cache.py hydrate-hf-cache \
  --plan artifacts/model-cache/qwen3-vl-8b-thinking.plan.json \
  --snapshot-dir /mnt/model-cache/qwen3-vl-8b-thinking \
  --cache-dir /mnt/huggingface/hub \
  --report-output artifacts/model-cache/hydration.json
```

Use the returned `model_load_config` fields (`model_name`, `cache_dir`, and
`revision`) directly with `ModelLoadConfig`, and set the returned
`HF_HUB_OFFLINE=1` requirement before launch. This preserves the approved Hub
model ID—`resolve_model_id` does not accept arbitrary local paths—while loading
entirely from the verified exact-revision cache. Hydration creates normal Hub
blob and snapshot links; it never downloads a file.

Materialize the small adjacent legal bundle separately:

```bash
uv run python scripts/build_tinylora_model_cache.py legal \
  --output-dir artifacts/model-cache/legal \
  --report-output artifacts/model-cache/legal.report.json
```

This fetches the canonical Apache-2.0 text and pinned upstream README (about 18
KB total), verifies their pinned SHA-256 values, and writes generated attribution
and modification notices. It does not fetch model weights.

## S3 publication contract

For a verified snapshot, the object prefix is:

```text
model-cache/v1/qwen--qwen3-vl-8b-thinking/
  <exact-revision>/<content-sha256>/
```

Under that prefix:

```text
files/<original-relative-path>
legal/LICENSE-APACHE-2.0.txt
legal/UPSTREAM_README.md
legal/ATTRIBUTION.json
legal/MODIFICATIONS.md
manifest.json
_COMPLETE.json
```

The safe publication sequence is:

1. Upload every object below `files/`, preserving its relative path.
2. Upload and verify all four legal/attribution objects below `legal/`. The
   upstream cache is marked byte-for-byte unmodified; every adapter or merged
   derivative must add an artifact-specific modification notice.
3. Verify every uploaded object's byte count and checksum against the local
   manifest. Do not rely on multipart S3 ETags as SHA-256 values.
4. Upload the exact canonical `manifest.json` bytes.
5. Upload `_COMPLETE.json` last. It names the content SHA-256 and SHA-256 of the
   canonical manifest bytes.

A consumer treats the cache as absent unless `_COMPLETE.json` exists and all of
these checks pass:

- the requested model ID and exact revision match;
- the downloaded manifest hashes to `manifest_sha256` in the completion marker;
- the manifest and completion marker name the same `content_sha256`;
- all 14 files exist locally with the manifest's exact size and SHA-256.

The prefix is immutable. Never overwrite or repair files in place. A changed
file produces a different content address and therefore a new prefix. No mutable
`latest` pointer is part of the training contract.

## Startup and failure policy

The eventual host bootstrap should fetch the small completion marker and
manifest first, measure cache download throughput, then fetch the 14 files in
parallel into durable local storage. It must verify the whole snapshot before
starting Python or counting the host as ready.

A slow cache transfer is a host/network qualification failure. A Python, CUDA,
dependency, or training failure is a software failure. Software failure must be
diagnosed on the existing machine and resumed from its durable checkpoint when
possible; it is not grounds to rent a replacement machine. If required outputs
or checkpoints have not been recovered and verified, stop/pause the machine for
recovery rather than destroy it.

The throughput threshold, upload implementation, host lifecycle wrapper, and
checkpoint publisher are separate Step 5 operational seams. This module gives
them a precise cache identity to enforce; it does not silently take those cloud
actions.
