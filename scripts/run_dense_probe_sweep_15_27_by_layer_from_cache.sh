#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ubuntu/intelligent_liars}"
UV="${UV:-$HOME/.local/bin/uv}"
MAX_PARALLEL="${MAX_PARALLEL:-20}"

cd "$ROOT"

export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

DENSE_CACHE="${DENSE_CACHE:-artifacts/probe_features/no_insider_dense_15_27_pooled.h5}"
EXPECTED_CACHE_SHA256="${EXPECTED_CACHE_SHA256:-}"
SWEEP_DIR="${SWEEP_DIR:-artifacts/probes/sweeps/dense_15_27_by_layer}"
SUMMARY_DOC="${SUMMARY_DOC:-docs/validation/no_insider_dense_15_27_primary_sweep_20260705_by_layer.md}"
DENSE_LAYER_LIST="${DENSE_LAYER_LIST:-15 16 17 18 19 20 21 22 23 24 25 26 27}"

read -r -a DENSE_LAYERS <<<"$DENSE_LAYER_LIST"

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

mkdir -p "$SWEEP_DIR" "$SWEEP_DIR/.locks" docs/validation logs/probe_sweep

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
  local layer="$4"
  shift 4
  local output="$SWEEP_DIR/${task_set}_layer${layer}_${c_tag}.json"
  local job_log="logs/probe_sweep/${task_set}_layer${layer}_${c_tag}.log"
  local lock_dir="$SWEEP_DIR/.locks/${task_set}_layer${layer}_${c_tag}.lock"

  if [[ -s "$output" ]]; then
    log "skipping existing result: $output"
    return 0
  fi

  if ! mkdir "$lock_dir" 2>/dev/null; then
    log "skipping in-progress result: $output"
    return 0
  fi

  log "starting train job task_set=$task_set layer=$layer c=$c_value output=$output"
  cmd=(
    "$UV" run --no-sync intelligent-liars train-probes-from-cache
    --cache "$DENSE_CACHE"
    --output "$output"
    --layers "$layer"
    --c "$c_value"
    --test-size 0.25
    --random-seed 0
    --general-domain-probe
  )
  for task in "$@"; do
    cmd+=(--task "$task")
  done
  local status=0
  "${cmd[@]}" >"$job_log" 2>&1 || status=$?
  rmdir "$lock_dir" 2>/dev/null || true
  if [[ "$status" != 0 ]]; then
    log "failed train job task_set=$task_set layer=$layer c=$c_value output=$output status=$status"
    return "$status"
  fi
  log "finished train job task_set=$task_set layer=$layer c=$c_value output=$output"
}

run_primary_sweep() {
  local pids=()
  for layer in "${DENSE_LAYERS[@]}"; do
    for i in "${!C_VALUES[@]}"; do
      throttle
      run_train_job all_no_insider "${C_VALUES[$i]}" "${C_TAGS[$i]}" "$layer" "${NO_INSIDER_TASKS[@]}" &
      pids+=("$!")

      throttle
      run_train_job no_repe "${C_VALUES[$i]}" "${C_TAGS[$i]}" "$layer" "${NO_REPE_TASKS[@]}" &
      pids+=("$!")
    done
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

check_result_count() {
  local expected=0
  local missing=0
  local layer
  local c_tag
  local path
  for layer in "${DENSE_LAYERS[@]}"; do
    for c_tag in "${C_TAGS[@]}"; do
      for task_set in all_no_insider no_repe; do
        path="$SWEEP_DIR/${task_set}_layer${layer}_${c_tag}.json"
        expected=$((expected + 1))
        if [[ ! -s "$path" ]]; then
          log "missing expected result: $path"
          missing=1
        fi
      done
    done
  done
  if [[ "$missing" != 0 ]]; then
    exit 1
  fi

  local count
  count="$(find "$SWEEP_DIR" -maxdepth 1 -name '*.json' -type f | wc -l | tr -d ' ')"
  log "verified selected result set: expected=$expected current_json_count=$count"
}

summarize() {
  log "summarizing by-layer sweep"
  "$UV" run --no-sync python scripts/summarize_probe_sweep.py \
    --results-dir "$SWEEP_DIR" \
    --output "$SUMMARY_DOC" \
    --title "No-Insider Dense 15-27 Primary Probe Sweep (By-Layer Execution)" \
    --cache "$DENSE_CACHE"
}

main() {
  log "by-layer dense probe sweep runner started root=$ROOT cache=$DENSE_CACHE max_parallel=$MAX_PARALLEL"
  verify_cache
  run_primary_sweep
  check_result_count
  summarize
  log "by-layer dense probe sweep runner complete"
}

main "$@"
