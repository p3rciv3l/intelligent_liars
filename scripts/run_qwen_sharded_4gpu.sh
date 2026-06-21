#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-/workspace/intelligent_liars}"

mkdir -p logs data/rollout_shards data/rollouts

GPUS_CSV="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPUS_ARRAY <<< "$GPUS_CSV"
TASKS_CSV="${TASKS:-roleplaying__plain,sandbagging_v2__wmdp_mmlu}"
IFS=',' read -r -a TASKS_ARRAY <<< "$TASKS_CSV"
SHARDS="${SHARDS:-${#GPUS_ARRAY[@]}}"
BATCH_SIZE="${BATCH_SIZE:-2}"
FLUSH_EVERY="${FLUSH_EVERY:-1}"
POLL_SECONDS="${POLL_SECONDS:-30}"
PROMPT_SET="${PROMPT_SET:-truth-spec-llama-70b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/rollout_shards}"
FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-data/rollouts}"
MODEL_SLUG="${MODEL_SLUG:-qwen3-vl-8b-thinking}"

if [ "${#GPUS_ARRAY[@]}" -ne "$SHARDS" ]; then
  echo "GPUS count (${#GPUS_ARRAY[@]}) must equal SHARDS (${SHARDS})." >&2
  exit 2
fi

mapfile -t SHARD_LINES < <(
  SHARDS="$SHARDS" PROMPT_SET="$PROMPT_SET" TASKS="$TASKS_CSV" PYTHONPATH=src uv run --no-sync python - <<'PY'
import os
from pathlib import Path
from intelligent_liars.rollouts import load_rollout_prompt_examples

project_root = Path(".").resolve()
prompt_set = os.environ["PROMPT_SET"]
tasks = [task for task in os.environ["TASKS"].split(",") if task]
shards = int(os.environ["SHARDS"])
total = max(
    len(load_rollout_prompt_examples(task, project_root=project_root, prompt_set=prompt_set))
    for task in tasks
)
for shard in range(shards):
    start = (total * shard) // shards
    end = (total * (shard + 1)) // shards
    print(f"{shard} {start} {end - start}")
PY
)

echo "starting ${SHARDS} qwen shards on GPUs ${GPUS_CSV} tasks=${TASKS_CSV}"
printf '%s\n' "${SHARD_LINES[@]}"

task_args=()
for task in "${TASKS_ARRAY[@]}"; do
  task_args+=(--task "$task")
done

pids=()
for line in "${SHARD_LINES[@]}"; do
  read -r shard start limit <<< "$line"
  gpu="${GPUS_ARRAY[$shard]}"
  shard_dir="${OUTPUT_ROOT}/shard-${shard}"
  log="logs/qwen-shard-${shard}.log"
  mkdir -p "$shard_dir"
  echo "gpu=${gpu} shard=${shard} start=${start} limit=${limit} log=${log}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export INTELLIGENT_LIARS_PROGRESS=1
    export PYTHONPATH=src
    uv run --no-sync intelligent-liars generate-rollouts \
      "${task_args[@]}" \
      --batch-size "$BATCH_SIZE" \
      --flush-every "$FLUSH_EVERY" \
      --prompt-set "$PROMPT_SET" \
      --start "$start" \
      --limit "$limit" \
      --output-dir "$shard_dir"
  ) >"$log" 2>&1 &
  pids+=("$!")
done

while :; do
  alive=0
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) alive_shards=${alive}/${#pids[@]}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  find "$OUTPUT_ROOT" -maxdepth 2 -type f -name '*.json' -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort || true
  [ "$alive" -eq 0 ] && break
  sleep "$POLL_SECONDS"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
[ "$failed" -eq 0 ] || exit 1

for task in "${TASKS_ARRAY[@]}"; do
  mapfile -t files < <(find "$OUTPUT_ROOT" -maxdepth 2 -type f -name "${task}__*.json" | sort)
  if [ "${#files[@]}" -eq 0 ]; then
    echo "no shard files found for ${task}" >&2
    exit 3
  fi
  uv run --no-sync intelligent-liars merge-rollouts \
    --output-path "${FINAL_OUTPUT_DIR}/${task}__${MODEL_SLUG}.json" \
    "${files[@]}"
done

echo "merged outputs:"
find "$FINAL_OUTPUT_DIR" -maxdepth 1 -type f -name '*.json' -printf '%p %s bytes\n' | sort
