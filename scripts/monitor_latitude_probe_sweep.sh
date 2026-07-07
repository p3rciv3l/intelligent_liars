#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-ubuntu@45.250.254.57}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/ubuntu/intelligent_liars}"
SOURCE="${SOURCE:-$REMOTE_ROOT/artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5}"
EXPECTED_SIZE="${EXPECTED_SIZE:-62394047728}"
TMUX_SESSION="${TMUX_SESSION-probe_sweep_15_27}"
TRANSFER_SCREEN_PATTERN="${TRANSFER_SCREEN_PATTERN:-hdf5_}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
STALL_LIMIT="${STALL_LIMIT:-5}"
SUMMARY_DOC="${SUMMARY_DOC:-$REMOTE_ROOT/docs/validation/no_insider_dense_15_27_primary_sweep_20260705.md}"

last_size=0
last_time=0
stall_count=0
notified_complete=0

notify() {
  local title="$1"
  local body="$2"
  printf '[%s] ALERT %s: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$title" "$body"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"${body//\"/\\\"}\" with title \"${title//\"/\\\"}\"" >/dev/null 2>&1 || true
  fi
}

remote_size() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
    "SOURCE='$SOURCE' python3 - <<'PY'
from pathlib import Path
import os

path = Path(os.environ['SOURCE'])
final_size = path.stat().st_size if path.exists() else 0
chunk_dir = path.parent / f'.chunks_{path.name}'
chunk_bytes = sum(chunk.stat().st_size for chunk in chunk_dir.glob('chunk_*')) if chunk_dir.exists() else 0
tmp_bytes = sum(chunk.stat().st_size for chunk in chunk_dir.glob('.chunk_*')) if chunk_dir.exists() else 0
print(final_size, chunk_bytes + tmp_bytes)
PY"
}

remote_tmux_alive() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
    "tmux has-session -t '$TMUX_SESSION' 2>/dev/null"
}

remote_rsync_alive() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
    "ps -eo cmd | grep -q '[r]sync --server'"
}

local_transfer_alive() {
  pgrep -f '[p]arallel_hdf5_upload_to_latitude.py|[r]sync -a --partial --append' >/dev/null 2>&1
}

remote_summary_exists() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
    "test -s '$SUMMARY_DOC'"
}

while true; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  now="$(date +%s)"
  raw_progress="$(remote_size || echo '0 0')"
  final_size="$(printf '%s\n' "$raw_progress" | awk '{print $1}')"
  chunk_bytes="$(printf '%s\n' "$raw_progress" | awk '{print $2}')"
  size="$final_size"
  if (( chunk_bytes > 0 )) && local_transfer_alive; then
    size="$chunk_bytes"
  elif (( chunk_bytes > size )); then
    size="$chunk_bytes"
  fi
  percent="$(python3 - <<PY
size = int("$size")
expected = int("$EXPECTED_SIZE")
print(f"{size / expected * 100:.2f}")
PY
)"
  if (( last_size > 0 && now > last_time && size > last_size )); then
    rate_eta="$(
      python3 - <<PY
size = int("$size")
last_size = int("$last_size")
expected = int("$EXPECTED_SIZE")
dt = int("$now") - int("$last_time")
rate = (size - last_size) / dt
eta = (expected - size) / rate if rate > 0 else 0
print(f"rate_mib_s={rate / 1024**2:.2f} eta_hours={eta / 3600:.2f}")
PY
    )"
  else
    rate_eta="rate_mib_s=unknown eta_hours=unknown"
  fi
  printf '[%s] final_size=%s chunk_bytes=%s progress_size=%s percent=%s %s\n' \
    "$timestamp" "$final_size" "$chunk_bytes" "$size" "$percent" "$rate_eta"

  if [[ -n "$TMUX_SESSION" ]] && ! remote_tmux_alive; then
    notify "Latitude probe sweep" "remote tmux session $TMUX_SESSION is not running"
  fi

  if (( final_size < EXPECTED_SIZE )); then
    if ! remote_rsync_alive && ! local_transfer_alive && ! screen -ls | grep -q "$TRANSFER_SCREEN_PATTERN"; then
      notify "Latitude HDF5 transfer" "no local transfer process/screen or remote rsync process is visible before HDF5 completed"
    fi
    if (( size <= last_size )); then
      stall_count=$((stall_count + 1))
      if (( stall_count >= STALL_LIMIT )); then
        notify "Latitude HDF5 transfer" "remote HDF5 size has not advanced for $stall_count checks"
        stall_count=0
      fi
    else
      stall_count=0
    fi
  elif (( notified_complete == 0 )); then
    notify "Latitude HDF5 transfer" "source HDF5 reached expected size; remote runner should begin validation"
    notified_complete=1
  fi

  if remote_summary_exists; then
    notify "Latitude probe sweep" "primary dense sweep summary doc exists"
    break
  fi

  last_size="$size"
  last_time="$now"
  sleep "$INTERVAL_SECONDS"
done
