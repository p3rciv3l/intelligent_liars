#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-ubuntu@45.250.254.57}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/ubuntu/intelligent_liars}"
SWEEP_DIR="${SWEEP_DIR:-artifacts/probes/sweeps/dense_15_27_by_layer}"
SUMMARY_DOC="${SUMMARY_DOC:-docs/validation/no_insider_dense_15_27_primary_sweep_20260705_by_layer.md}"
CACHE="${CACHE:-artifacts/probe_features/no_insider_dense_15_27_pooled.h5}"
EXPECTED_CACHE_SHA256="${EXPECTED_CACHE_SHA256:-8bfc57d8f74cdc687f3f8e73720479c585bac6f16b72ce6e2bcd18fb5245ad77}"
EXPECTED_RESULTS="${EXPECTED_RESULTS:-182}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-10}"

PAUSE_ACTIVE="${PAUSE_ACTIVE:-45}"
PAUSE_AVAIL_GIB="${PAUSE_AVAIL_GIB:-65}"
RESUME_ACTIVE="${RESUME_ACTIVE:-42}"
RESUME_AVAIL_GIB="${RESUME_AVAIL_GIB:-70}"

FILL_SESSION="${FILL_SESSION:-probe_sweep_26_27_controlled_fill}"
FILL_LAYERS="${FILL_LAYERS:-26 27}"
FILL_MAX_PARALLEL="${FILL_MAX_PARALLEL:-3}"
FILL_SUMMARY_DOC="${FILL_SUMMARY_DOC:-docs/validation/no_insider_dense_15_27_controlled_fill_partial_IGNORE.md}"

notify() {
  local title="$1"
  local body="$2"
  printf '[%s] ALERT %s: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$title" "$body"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"${body//\"/\\\"}\" with title \"${title//\"/\\\"}\"" >/dev/null 2>&1 || true
  fi
}

remote_status() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
    "cd '$REMOTE_ROOT' && SWEEP_DIR='$SWEEP_DIR' python3 - <<'PY'
from pathlib import Path
import datetime as dt
import os
import subprocess

root = Path(os.environ['SWEEP_DIR'])
locks = root / '.locks'
sets = ['all_no_insider', 'no_repe']
c_tags = ['c001', 'c003', 'c01', 'c03', 'c1', 'c3', 'c10']

ps = subprocess.check_output(['ps', '-eo', 'args='], text=True)
active_outs = []
for line in ps.splitlines():
    if '.venv/bin/python' in line and 'train-probes-from-cache' in line:
        toks = line.split()
        if '--output' in toks:
            active_outs.append(toks[toks.index('--output') + 1])

active_set = set(active_outs)
dup_active = len(active_outs) - len(active_set)
count = sum(1 for path in root.glob('*.json') if path.is_file())
plain_total = 0
locked_total = 0
active_expected = 0
complete_expected = 0
for layer in range(15, 28):
    for task_set in sets:
        for c_tag in c_tags:
            rel = f'{task_set}_layer{layer}_{c_tag}.json'
            out = root / rel
            if out.exists() and out.stat().st_size > 0:
                complete_expected += 1
            elif str(out) in active_set:
                active_expected += 1
            elif (locks / f'{task_set}_layer{layer}_{c_tag}.lock').exists():
                locked_total += 1
            else:
                plain_total += 1

critical = 0
log_root = Path('logs/probe_sweep')
if log_root.exists():
    needles = ('Traceback', 'Exception', 'MemoryError', 'No space', 'Killed')
    for log in log_root.glob('*.log'):
        try:
            text = log.read_text(errors='ignore')
        except OSError:
            continue
        critical += sum(text.count(needle) for needle in needles)

mem_available_kib = 0
with open('/proc/meminfo') as f:
    for line in f:
        if line.startswith('MemAvailable:'):
            mem_available_kib = int(line.split()[1])
            break
avail_gib = mem_available_kib / 1024 / 1024
tmux = subprocess.run(['tmux', 'ls'], text=True, capture_output=True)
tmux_probe = sum(1 for line in tmux.stdout.splitlines() if 'probe_sweep_' in line)

print('time=' + dt.datetime.now(dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'))
print(f'count={count}')
print(f'complete_expected={complete_expected}')
print(f'active={len(active_outs)}')
print(f'active_expected={active_expected}')
print(f'locked_total={locked_total}')
print(f'plain_total={plain_total}')
print(f'dup_active={dup_active}')
print(f'critical={critical}')
print(f'avail_gib={avail_gib:.1f}')
print(f'tmux_probe={tmux_probe}')
PY"
}

value_of() {
  local key="$1"
  awk -F= -v k="$key" '$1 == k {print $2}' <<<"$STATUS"
}

pause_lockaware_launchers() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" "set -euo pipefail
for s in \$(tmux ls 2>/dev/null | awk -F: '/probe_sweep_/ {print \$1}'); do
  case \"\$s\" in
    probe_sweep_15_27_by_layer|probe_sweep_20_27_supplemental) continue ;;
  esac
  pane_pid=\$(tmux list-panes -t \"\$s\" -F '#{pane_pid}' 2>/dev/null || true)
  if [ -n \"\$pane_pid\" ]; then
    launcher=\$(pgrep -P \"\$pane_pid\" -f 'bash scripts/run_dense_probe_sweep_15_27_by_layer_from_cache.sh' | head -1 || true)
    if [ -n \"\$launcher\" ]; then kill -STOP \"\$launcher\" || true; fi
  fi
done" >/dev/null
}

resume_controlled_filler() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" "set -euo pipefail
if tmux has-session -t '$FILL_SESSION' 2>/dev/null; then
  pane_pid=\$(tmux list-panes -t '$FILL_SESSION' -F '#{pane_pid}' 2>/dev/null || true)
  if [ -n \"\$pane_pid\" ]; then
    launcher=\$(pgrep -P \"\$pane_pid\" -f 'bash scripts/run_dense_probe_sweep_15_27_by_layer_from_cache.sh' | head -1 || true)
    if [ -n \"\$launcher\" ]; then kill -CONT \"\$launcher\" || true; fi
  fi
fi" >/dev/null
}

start_controlled_filler() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" "set -euo pipefail
cd '$REMOTE_ROOT'
if tmux has-session -t '$FILL_SESSION' 2>/dev/null; then
  exit 0
fi
log='logs/probe_sweep/${FILL_SESSION}_'\$(date -u +%Y%m%dT%H%M%SZ)'.log'
tmux new-session -d -s '$FILL_SESSION' \"cd '$REMOTE_ROOT' && EXPECTED_CACHE_SHA256='$EXPECTED_CACHE_SHA256' MAX_PARALLEL='$FILL_MAX_PARALLEL' DENSE_LAYER_LIST='$FILL_LAYERS' SWEEP_DIR='$SWEEP_DIR' SUMMARY_DOC='$FILL_SUMMARY_DOC' bash scripts/run_dense_probe_sweep_15_27_by_layer_from_cache.sh > \\\$log 2>&1\"
echo started"
}

run_final_summary() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
    "cd '$REMOTE_ROOT' && PYTHONPATH=src ~/.local/bin/uv run --no-sync python scripts/summarize_probe_sweep.py --results-dir '$SWEEP_DIR' --output '$SUMMARY_DOC' --title 'No-Insider Dense 15-27 Primary Probe Sweep (By-Layer Execution)' --cache '$CACHE'"
}

while true; do
  STATUS="$(remote_status)"
  time="$(value_of time)"
  count="$(value_of count)"
  complete_expected="$(value_of complete_expected)"
  active="$(value_of active)"
  active_expected="$(value_of active_expected)"
  locked_total="$(value_of locked_total)"
  plain_total="$(value_of plain_total)"
  dup_active="$(value_of dup_active)"
  critical="$(value_of critical)"
  avail_gib="$(value_of avail_gib)"
  tmux_probe="$(value_of tmux_probe)"

  printf '[%s] count=%s/%s complete_expected=%s active=%s active_expected=%s locked=%s plain=%s dup=%s critical=%s avail_gib=%s tmux=%s\n' \
    "$time" "$count" "$EXPECTED_RESULTS" "$complete_expected" "$active" "$active_expected" "$locked_total" "$plain_total" "$dup_active" "$critical" "$avail_gib" "$tmux_probe"

  if (( critical > 0 )); then
    notify "Latitude probe sweep" "critical log signature detected; inspect logs/probe_sweep"
    exit 1
  fi
  if (( dup_active > 0 )); then
    pause_lockaware_launchers || true
    notify "Latitude probe sweep" "duplicate active output detected; lock-aware launchers paused"
    exit 1
  fi

  if (( count >= EXPECTED_RESULTS && complete_expected >= EXPECTED_RESULTS && active == 0 && plain_total == 0 && locked_total == 0 )); then
    printf '[%s] grid complete; running final summarizer\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    run_final_summary
    notify "Latitude probe sweep" "dense 15-27 sweep complete and summarized"
    break
  fi

  if (( active >= PAUSE_ACTIVE )); then
    pause_lockaware_launchers || true
  elif python3 - "$avail_gib" "$PAUSE_AVAIL_GIB" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)
PY
  then
    pause_lockaware_launchers || true
  elif (( plain_total > 0 && active <= RESUME_ACTIVE )); then
    if python3 - "$avail_gib" "$RESUME_AVAIL_GIB" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
    then
      start_controlled_filler >/dev/null || true
      resume_controlled_filler || true
    fi
  fi

  sleep "$INTERVAL_SECONDS"
done
