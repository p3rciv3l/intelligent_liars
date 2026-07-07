#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ubuntu/intelligent_liars}"
UV="${UV:-$HOME/.local/bin/uv}"
MAX_PARALLEL="${MAX_PARALLEL:-14}"

cd "$ROOT"

export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

DENSE_LAYERS="15,16,17,18,19,20,21,22,23,24,25,26,27"
SMOKE_LAYERS="15,16,17"
DENSE_CACHE="${DENSE_CACHE:-artifacts/probe_features/no_insider_dense_15_27_pooled.h5}"
EXPECTED_CACHE_SHA256="${EXPECTED_CACHE_SHA256:-}"
SWEEP_DIR="artifacts/probes/sweeps/dense_15_27"
SMOKE_RESULT="artifacts/probes/no_insider_dense_smoke_15_17_two_task_probe_results.json"
SUMMARY_DOC="docs/validation/no_insider_dense_15_27_primary_sweep_20260705.md"

NO_INSIDER_TASKS=(
  claims__definitional_gemini_600_full
  claims__evidential_gemini_600_full
  claims__fictional_gemini_600_full
  claims__logical_gemini_600_full
  ethics__commonsense
  geometry_of_truth__best
  geometry_of_truth__mixed
  internal_state__animals
  internal_state__cities
  internal_state__companies
  internal_state__elements
  internal_state__facts
  internal_state__inventions
  repe_honesty__IF_dishonest
  roleplaying__plain
  sandbagging_v2__wmdp_mmlu
  sycophancy__mmlu_stem_same_conf_all
)

NO_REPE_TASKS=(
  claims__definitional_gemini_600_full
  claims__evidential_gemini_600_full
  claims__fictional_gemini_600_full
  claims__logical_gemini_600_full
  ethics__commonsense
  geometry_of_truth__best
  geometry_of_truth__mixed
  internal_state__animals
  internal_state__cities
  internal_state__companies
  internal_state__elements
  internal_state__facts
  internal_state__inventions
  roleplaying__plain
  sandbagging_v2__wmdp_mmlu
  sycophancy__mmlu_stem_same_conf_all
)

C_VALUES=(0.01 0.03 0.1 0.3 1 3 10)
C_TAGS=(c001 c003 c01 c03 c1 c3 c10)

mkdir -p "$SWEEP_DIR" artifacts/probes docs/validation logs/probe_sweep

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

verify_cache() {
  if [[ ! -s "$DENSE_CACHE" ]]; then
    log "missing dense cache: $DENSE_CACHE"
    exit 1
  fi
  if [[ -n "$EXPECTED_CACHE_SHA256" ]]; then
    log "checking dense cache sha256"
    actual="$(sha256sum "$DENSE_CACHE" | awk '{print $1}')"
    if [[ "$actual" != "$EXPECTED_CACHE_SHA256" ]]; then
      log "cache sha256 mismatch: expected $EXPECTED_CACHE_SHA256 got $actual"
      exit 1
    fi
    log "dense cache sha256 verified"
  else
    log "EXPECTED_CACHE_SHA256 is empty; skipping cache sha verification"
  fi

  log "validating dense cache schema"
  "$UV" run --no-sync python - "$DENSE_CACHE" "$DENSE_LAYERS" "${NO_INSIDER_TASKS[@]}" <<'PY'
import sys
from pathlib import Path

import h5py

cache_path = Path(sys.argv[1])
layers = [int(part) for part in sys.argv[2].split(",")]
tasks = sys.argv[3:]

with h5py.File(cache_path, "r") as handle:
    if handle.attrs.get("format") != "qwen_answer_token_pooled_features_v1":
        raise SystemExit(f"bad cache format: {handle.attrs.get('format')!r}")
    if int(handle.attrs.get("hidden_dim", -1)) != 4096:
        raise SystemExit(f"bad hidden_dim: {handle.attrs.get('hidden_dim')!r}")
    cached_layers = [int(layer) for layer in handle.attrs["selected_layers"]]
    if cached_layers != layers:
        raise SystemExit(f"bad layer list: {cached_layers}")
    for task in tasks:
        if f"metadata/{task}" not in handle:
            raise SystemExit(f"missing metadata task: {task}")
    for layer in layers:
        for task in tasks:
            path = f"layer_{layer}/{task}"
            if path not in handle:
                raise SystemExit(f"missing cached features: {path}")
            rows, hidden_dim = handle[path].shape
            label_rows = handle[f"metadata/{task}/example_labels"].shape[0]
            if rows != label_rows or hidden_dim != 4096:
                raise SystemExit(f"bad shape for {path}: {(rows, hidden_dim)} labels={label_rows}")
print("dense cache schema verified")
PY
}

run_smoke() {
  if [[ ! -s "$SMOKE_RESULT" ]]; then
    log "training second smoke probe from dense cache"
    "$UV" run --no-sync intelligent-liars train-probes-from-cache \
      --cache "$DENSE_CACHE" \
      --output "$SMOKE_RESULT" \
      --layers "$SMOKE_LAYERS" \
      --task claims__definitional_gemini_600_full \
      --task claims__evidential_gemini_600_full \
      --c 1.0 \
      --test-size 0.25 \
      --random-seed 0 \
      --general-domain-probe
  else
    log "smoke result already exists: $SMOKE_RESULT"
  fi

  log "checking second smoke result counts"
  "$UV" run --no-sync python - "$SMOKE_RESULT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
counts = {
    "within_task": len(payload.get("within_task", [])),
    "cross_task": len(payload.get("cross_task", [])),
    "directions": len(payload.get("directions", [])),
    "general_domain_evaluations": len(payload.get("general_domain", {}).get("evaluations", [])),
    "general_domain_directions": len(payload.get("general_domain", {}).get("directions", [])),
}
expected = {
    "within_task": 6,
    "cross_task": 6,
    "directions": 6,
    "general_domain_evaluations": 6,
    "general_domain_directions": 3,
}
if counts != expected:
    raise SystemExit(f"unexpected smoke counts: {counts}, expected {expected}")
print(counts)
PY
}

throttle() {
  while (( $(jobs -pr | wc -l | tr -d ' ') >= MAX_PARALLEL )); do
    sleep 10
  done
}

run_train_job() {
  local task_set="$1"
  local c_value="$2"
  local c_tag="$3"
  shift 3
  local output="$SWEEP_DIR/${task_set}_${c_tag}.json"
  local job_log="logs/probe_sweep/${task_set}_${c_tag}.log"
  if [[ -s "$output" ]]; then
    log "skipping existing result: $output"
    return 0
  fi

  log "starting train job task_set=$task_set c=$c_value output=$output"
  cmd=(
    "$UV" run --no-sync intelligent-liars train-probes-from-cache
    --cache "$DENSE_CACHE"
    --output "$output"
    --layers "$DENSE_LAYERS"
    --c "$c_value"
    --test-size 0.25
    --random-seed 0
    --general-domain-probe
  )
  for task in "$@"; do
    cmd+=(--task "$task")
  done
  "${cmd[@]}" >"$job_log" 2>&1
  log "finished train job task_set=$task_set c=$c_value output=$output"
}

run_primary_sweep() {
  local pids=()
  for i in "${!C_VALUES[@]}"; do
    throttle
    run_train_job all_no_insider "${C_VALUES[$i]}" "${C_TAGS[$i]}" "${NO_INSIDER_TASKS[@]}" &
    pids+=("$!")

    throttle
    run_train_job no_repe "${C_VALUES[$i]}" "${C_TAGS[$i]}" "${NO_REPE_TASKS[@]}" &
    pids+=("$!")
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" != 0 ]]; then
    log "one or more train jobs failed; see logs/probe_sweep"
    exit "$status"
  fi
}

summarize() {
  log "summarizing sweep"
  "$UV" run --no-sync python scripts/summarize_probe_sweep.py \
    --results-dir "$SWEEP_DIR" \
    --output "$SUMMARY_DOC" \
    --title "No-Insider Dense 15-27 Primary Probe Sweep" \
    --cache "$DENSE_CACHE"
}

main() {
  log "cache-only dense probe sweep runner started root=$ROOT cache=$DENSE_CACHE max_parallel=$MAX_PARALLEL"
  verify_cache
  run_smoke
  run_primary_sweep
  summarize
  log "cache-only dense probe sweep runner complete"
}

main "$@"
