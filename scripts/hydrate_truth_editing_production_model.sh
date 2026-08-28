#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root=/workspace/model-source
snapshot_dir="${source_root}/standalone-qwen-snapshot"
plan="${source_root}/snapshot-plan.json"
cache_dir="${repo_root}/artifacts/truth-editing/model-cache/huggingface"
manifest="${repo_root}/artifacts/truth-editing/model-cache/snapshot-manifest.json"
receipt=/workspace/outputs/model/cache-hydration-receipt.json

export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "${source_root}" "${cache_dir}" "$(dirname "${receipt}")"
python scripts/build_tinylora_model_cache.py download \
  --execute-download \
  --snapshot-dir "${snapshot_dir}" \
  --plan-output "${plan}" \
  --manifest-output "${manifest}"
python scripts/build_tinylora_model_cache.py hydrate-hf-cache \
  --plan "${plan}" \
  --snapshot-dir "${snapshot_dir}" \
  --cache-dir "${cache_dir}" \
  --report-output "${receipt}"
