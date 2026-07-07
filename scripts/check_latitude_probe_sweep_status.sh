#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-ubuntu@45.250.254.57}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/ubuntu/intelligent_liars}"
SOURCE="$REMOTE_ROOT/artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5"
EXPECTED_SIZE="${EXPECTED_SIZE:-62394047728}"
CACHE="$REMOTE_ROOT/artifacts/probe_features/no_insider_dense_15_27_pooled.h5"
EXPECTED_CACHE_SIZE="${EXPECTED_CACHE_SIZE:-11186928745}"
TMUX_SESSION="${TMUX_SESSION:-probe_sweep_15_27}"

echo "== local transfer screen =="
screen -ls || true

echo
echo "== remote source artifact (abandoned full-HDF5 route) =="
ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
  "SOURCE='$SOURCE' EXPECTED_SIZE='$EXPECTED_SIZE' python3 - <<'PY'
import os
import time
from pathlib import Path

path = Path(os.environ['SOURCE'])
expected = int(os.environ['EXPECTED_SIZE'])
if not path.exists():
    print('missing')
    raise SystemExit
size = path.stat().st_size
mtime = path.stat().st_mtime
chunk_dir = path.parent / f'.chunks_{path.name}'
chunk_sizes = [chunk.stat().st_size for chunk in chunk_dir.glob('chunk_*')] if chunk_dir.exists() else []
chunk_bytes = sum(chunk_sizes)
progress_bytes = chunk_bytes if chunk_bytes else size
print(f'size={size}')
print(f'expected={expected}')
print(f'percent={size / expected * 100:.2f}')
print(f'remaining_gib={(expected - size) / 1024**3:.2f}')
print('updated_utc=' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(mtime)))
print(f'chunk_count={len(chunk_sizes)}')
print(f'chunk_bytes={chunk_bytes}')
print(f'chunk_percent={chunk_bytes / expected * 100:.2f}')
print(f'progress_percent={progress_bytes / expected * 100:.2f}')
print(f'progress_remaining_gib={(expected - progress_bytes) / 1024**3:.2f}')
PY"

echo
echo "== remote dense cache artifact =="
ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
  "SOURCE='$CACHE' EXPECTED_SIZE='$EXPECTED_CACHE_SIZE' python3 - <<'PY'
import os
import time
from pathlib import Path

path = Path(os.environ['SOURCE'])
expected = int(os.environ['EXPECTED_SIZE'])
if not path.exists():
    print('missing')
    chunk_dir = path.parent / f'.chunks_{path.name}'
    if chunk_dir.exists():
        chunk_sizes = [chunk.stat().st_size for chunk in chunk_dir.glob('chunk_*')]
        tmp_sizes = [chunk.stat().st_size for chunk in chunk_dir.glob('.chunk_*')]
        progress_bytes = sum(chunk_sizes) + sum(tmp_sizes)
        print(f'chunk_count={len(chunk_sizes)}')
        print(f'chunk_bytes={sum(chunk_sizes)}')
        print(f'tmp_count={len(tmp_sizes)}')
        print(f'tmp_bytes={sum(tmp_sizes)}')
        print(f'progress_bytes={progress_bytes}')
        if expected:
            print(f'progress_percent={progress_bytes / expected * 100:.2f}')
    raise SystemExit
size = path.stat().st_size
mtime = path.stat().st_mtime
chunk_dir = path.parent / f'.chunks_{path.name}'
chunk_sizes = [chunk.stat().st_size for chunk in chunk_dir.glob('chunk_*')] if chunk_dir.exists() else []
tmp_sizes = [chunk.stat().st_size for chunk in chunk_dir.glob('.chunk_*')] if chunk_dir.exists() else []
chunk_bytes = sum(chunk_sizes)
tmp_bytes = sum(tmp_sizes)
progress_bytes = chunk_bytes + tmp_bytes if chunk_bytes or tmp_bytes else size
print(f'size={size}')
if expected:
    print(f'expected={expected}')
    print(f'percent={size / expected * 100:.2f}')
    print(f'chunk_percent={chunk_bytes / expected * 100:.2f}')
    print(f'progress_percent={progress_bytes / expected * 100:.2f}')
else:
    print('expected=unknown')
print('updated_utc=' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(mtime)))
print(f'chunk_count={len(chunk_sizes)}')
print(f'chunk_bytes={chunk_bytes}')
print(f'tmp_count={len(tmp_sizes)}')
print(f'tmp_bytes={tmp_bytes}')
print(f'progress_bytes={progress_bytes}')
PY"

echo
echo "== remote tmux =="
ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" "tmux ls 2>/dev/null || true"

echo
echo "== runner tail =="
ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
  "cd '$REMOTE_ROOT' && if tmux has-session -t probe_sweep_15_27_cache 2>/dev/null; then latest=\$(ls -t logs/probe_sweep/dense_15_27_cache_runner_*.log 2>/dev/null | head -1); elif tmux has-session -t probe_sweep_15_27 2>/dev/null; then latest=\$(ls -t logs/probe_sweep/dense_15_27_runner_*.log 2>/dev/null | head -1); else latest=''; fi; if [ -n \"\$latest\" ]; then echo \"log=\$latest\"; tail -20 \"\$latest\"; else echo no-active-runner-yet; fi"

echo
echo "== generated artifacts =="
ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
  "cd '$REMOTE_ROOT' && find artifacts/probe_features artifacts/probes docs/validation -maxdepth 3 -type f 2>/dev/null | grep -v '/.chunks_no_insider_dense_15_27_pooled.h5/' | sort | sed -n '1,160p'"
