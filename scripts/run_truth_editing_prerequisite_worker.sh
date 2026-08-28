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
python scripts/run_truth_editing_prerequisite_inference.py \
  --base-config configs/truth_editing_base_known_v1.json \
  --dataset-dir datasets/truth_editing/v2 \
  --structured-view datasets/truth_editing/views/structured_semantic_v1 \
  --structured-source-root corpora/tinylora_deception_action_v1/step5_v1 \
  --refusal-config configs/truth_editing_refusal_directions_v1.json \
  --refusal-prompts configs/truth_editing_refusal_prompt_manifest_v1.json \
  --cache-dir "${hf_cache}" \
  --snapshot-manifest "${cache_root}/model-cache-manifest.json" \
  --output-dir "${output_root}"
