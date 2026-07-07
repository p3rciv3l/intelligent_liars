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

SOURCE="artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5"
EXPECTED_SIZE="62394047728"
EXPECTED_SHA256="c6e687f69256544121d0718cf8bba142ed5837221976121e1899552dc76f1a5a"

DENSE_LAYERS="15,16,17,18,19,20,21,22,23,24,25,26,27"
SMOKE_LAYERS="15,16,17"
SMOKE_CACHE="artifacts/probe_features/no_insider_dense_smoke_15_17_two_task_pooled.h5"
SMOKE_RESULT="artifacts/probes/no_insider_dense_smoke_15_17_two_task_probe_results.json"
DENSE_CACHE="artifacts/probe_features/no_insider_dense_15_27_pooled.h5"
SWEEP_DIR="artifacts/probes/sweeps/dense_15_27"
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

mkdir -p artifacts/probe_features "$SWEEP_DIR" docs/validation logs/probe_sweep

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

wait_for_source() {
  log "waiting for $SOURCE to reach $EXPECTED_SIZE bytes"
  while true; do
    if [[ -f "$SOURCE" ]]; then
      size="$(stat -c %s "$SOURCE")"
      log "source size: $size / $EXPECTED_SIZE"
      if [[ "$size" == "$EXPECTED_SIZE" ]]; then
        break
      fi
    else
      log "source is not present yet"
    fi
    sleep 60
  done
}

verify_source() {
  log "checking source sha256"
  actual="$(sha256sum "$SOURCE" | awk '{print $1}')"
  if [[ "$actual" != "$EXPECTED_SHA256" ]]; then
    log "sha256 mismatch: expected $EXPECTED_SHA256 got $actual"
    exit 1
  fi
  log "source sha256 verified"

  log "validating HDF5 metadata/readability"
  "$UV" run --no-sync python scripts/validate_activation_hdf5.py "$SOURCE" \
    --expected-task-count 20 \
    --expected-layer-count 36 \
    --expected-hidden-dim 4096 \
    --finite-check sample
}

run_smoke() {
  if [[ ! -s "$SMOKE_CACHE" ]]; then
    log "building second smoke cache"
    "$UV" run --no-sync intelligent-liars cache-pooled-features \
      --input "$SOURCE" \
      --output "$SMOKE_CACHE" \
      --layers "$SMOKE_LAYERS" \
      --task claims__definitional_gemini_600_full \
      --task claims__evidential_gemini_600_full
  else
    log "smoke cache already exists: $SMOKE_CACHE"
  fi

  if [[ ! -s "$SMOKE_RESULT" ]]; then
    log "training second smoke probe"
    "$UV" run --no-sync intelligent-liars train-probes-from-cache \
      --cache "$SMOKE_CACHE" \
      --source "$SOURCE" \
      --output "$SMOKE_RESULT" \
      --layers "$SMOKE_LAYERS" \
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

build_dense_cache() {
  if [[ ! -s "$DENSE_CACHE" ]]; then
    log "building dense 15-27 no-insider cache"
    cmd=(
      "$UV" run --no-sync intelligent-liars cache-pooled-features
      --input "$SOURCE"
      --output "$DENSE_CACHE"
      --layers "$DENSE_LAYERS"
    )
    for task in "${NO_INSIDER_TASKS[@]}"; do
      cmd+=(--task "$task")
    done
    "${cmd[@]}"
  else
    log "dense cache already exists: $DENSE_CACHE"
  fi
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
    --source "$SOURCE"
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
    --source "$SOURCE" \
    --cache "$DENSE_CACHE"
}

main() {
  log "dense probe sweep runner started at root=$ROOT max_parallel=$MAX_PARALLEL"
  wait_for_source
  verify_source
  run_smoke
  build_dense_cache
  run_primary_sweep
  summarize
  log "dense probe sweep runner complete"
}

main "$@"
