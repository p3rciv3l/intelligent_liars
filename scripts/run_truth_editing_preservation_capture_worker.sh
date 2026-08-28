#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache_root=/workspace/cache
snapshot_dir="${cache_root}/standalone-qwen-snapshot"
hf_cache="${cache_root}/huggingface"
output_root=/workspace/outputs

export PYTHONPATH="${repo_root}/src"
export HF_HOME="${hf_cache}"
export HF_HUB_CACHE="${hf_cache}/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRUTH_EDITING_MODEL_CACHE_MANIFEST="${cache_root}/model-cache-manifest.json"
export TRUTH_EDITING_PRESERVATION_OUTPUT="${output_root}"

tar -xzf artifacts/truth-editing/vast/preservation-capture-v2-r4-inputs/osworld-source-v2-r4.tar.gz
tar -xzf artifacts/truth-editing/vast/preservation-capture-v2-r4-inputs/vision-media.tar.gz
mkdir -p "${cache_root}" "${hf_cache}" "${output_root}"
python docker/truth-editing/validate_runtime.py
python scripts/build_tinylora_model_cache.py download \
  --execute-download \
  --snapshot-dir "${snapshot_dir}" \
  --plan-output "${cache_root}/snapshot-plan.json" \
  --manifest-output "${cache_root}/model-cache-manifest.json"
python scripts/build_tinylora_model_cache.py hydrate-hf-cache \
  --plan "${cache_root}/snapshot-plan.json" \
  --snapshot-dir "${snapshot_dir}" \
  --cache-dir "${hf_cache}" \
  --report-output "${output_root}/cache-hydration-receipt.json"
export HF_HUB_OFFLINE=1
python scripts/run_truth_editing_preservation_capture_worker.py
