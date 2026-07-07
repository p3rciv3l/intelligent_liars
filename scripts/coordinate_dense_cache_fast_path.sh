#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-ubuntu@45.250.254.57}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/ubuntu/intelligent_liars}"
LOCAL_CACHE="${LOCAL_CACHE:-artifacts/probe_features/no_insider_dense_15_27_pooled.h5}"
REMOTE_CACHE="$REMOTE_ROOT/$LOCAL_CACHE"
SOURCE="$REMOTE_ROOT/artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5"
EXPECTED_SOURCE_SIZE="${EXPECTED_SOURCE_SIZE:-62394047728}"
CACHE_BUILD_SCREEN_PATTERN="${CACHE_BUILD_SCREEN_PATTERN:-local_dense_cache_build_}"
HDF5_UPLOAD_SCREEN_PATTERN="${HDF5_UPLOAD_SCREEN_PATTERN:-hdf5_parallel_upload_}"
OLD_TMUX_SESSION="${OLD_TMUX_SESSION:-probe_sweep_15_27}"
CACHE_TMUX_SESSION="${CACHE_TMUX_SESSION:-probe_sweep_15_27_cache}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

remote_source_complete() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
    "test -f '$SOURCE' && test \$(stat -c %s '$SOURCE') -eq '$EXPECTED_SOURCE_SIZE'" >/dev/null 2>&1
}

screen_session_matching() {
  local pattern="$1"
  screen -ls | awk -v pattern="$pattern" '$0 ~ pattern {print $1; exit}' || true
}

stop_screen_matching() {
  local pattern="$1"
  local session
  session="$(screen_session_matching "$pattern")"
  if [[ -n "$session" ]]; then
    log "stopping local screen session $session"
    screen -S "$session" -X quit || true
  fi
}

stop_full_hdf5_upload_processes() {
  pkill -TERM -f 'parallel_hdf5_upload_to_latitude.py --workers 8 --chunk-size-mib 64 --cleanup-chunks' || true
  pkill -TERM -f '.chunks_extracted_feats_all_text_qwen3-vl-8b-thinking.h5' || true
}

validate_local_cache() {
  "$PYTHON_BIN" - "$LOCAL_CACHE" <<'PY'
import sys
from pathlib import Path

import h5py

path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size <= 0:
    raise SystemExit(f"cache missing or empty: {path}")
with h5py.File(path, "r") as handle:
    if handle.attrs.get("format") != "qwen_answer_token_pooled_features_v1":
        raise SystemExit(f"unexpected cache format: {handle.attrs.get('format')!r}")
    layers = [int(layer) for layer in handle.attrs["selected_layers"]]
    expected_layers = list(range(15, 28))
    if layers != expected_layers:
        raise SystemExit(f"unexpected layers: {layers}")
    if int(handle.attrs.get("hidden_dim", -1)) != 4096:
        raise SystemExit(f"unexpected hidden dim: {handle.attrs.get('hidden_dim')!r}")
print("local cache schema ok")
PY
}

wait_for_local_cache() {
  log "waiting for local dense cache: $LOCAL_CACHE"
  while [[ ! -s "$LOCAL_CACHE" ]]; do
    if remote_source_complete; then
      log "remote source HDF5 completed first; leaving original runner in control"
      exit 0
    fi
    if ! { screen -ls || true; } | grep -q "$CACHE_BUILD_SCREEN_PATTERN"; then
      log "cache build screen is not running and cache does not exist"
      exit 1
    fi
    log "local cache still building"
    sleep "$WAIT_SECONDS"
  done
  validate_local_cache
}

upload_cache() {
  local cache_sha
  cache_sha="$(shasum -a 256 "$LOCAL_CACHE" | awk '{print $1}')"
  local cache_size
  cache_size="$(stat -f %z "$LOCAL_CACHE")"
  log "local cache ready size=$cache_size sha256=$cache_sha"

  if remote_source_complete; then
    log "remote source HDF5 completed before cache upload; leaving original runner in control"
    exit 0
  fi

  stop_screen_matching "$HDF5_UPLOAD_SCREEN_PATTERN"
  stop_full_hdf5_upload_processes
  log "copying cache-only runner to remote"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" "mkdir -p '$REMOTE_ROOT/scripts' '$REMOTE_ROOT/logs/probe_sweep'"
  scp -q scripts/run_dense_probe_sweep_15_27_from_cache.sh "$REMOTE:$REMOTE_ROOT/scripts/run_dense_probe_sweep_15_27_from_cache.sh"

  log "uploading dense cache to $REMOTE_CACHE"
  python3 scripts/parallel_hdf5_upload_to_latitude.py \
    --local "$LOCAL_CACHE" \
    --remote "$REMOTE" \
    --remote-root "$REMOTE_ROOT" \
    --workers 8 \
    --chunk-size-mib 64 \
    --expected-sha256 "$cache_sha" \
    --cleanup-chunks

  log "stopping old source-wait runner if still present"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
    "tmux kill-session -t '$OLD_TMUX_SESSION' 2>/dev/null || true"

  local runner_log
  runner_log="dense_15_27_cache_runner_$(date -u +%Y%m%dT%H%M%SZ).log"
  log "starting cache-only remote runner session=$CACHE_TMUX_SESSION log=logs/probe_sweep/$runner_log"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
    "cd '$REMOTE_ROOT' && tmux new-session -d -s '$CACHE_TMUX_SESSION' \"cd '$REMOTE_ROOT' && EXPECTED_CACHE_SHA256='$cache_sha' bash scripts/run_dense_probe_sweep_15_27_from_cache.sh 2>&1 | tee logs/probe_sweep/$runner_log\""
}

main() {
  wait_for_local_cache
  upload_cache
  log "cache fast path coordinator complete"
}

main "$@"
