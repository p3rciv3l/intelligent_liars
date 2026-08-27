# Vast Qwen Rollout Postmortem

> **Archived snapshot — do not execute.** Hosts, paths, commands, and runtime
> assumptions record a June 2026 run and are not current instructions.

Date: 2026-06-21

Artifact root: `artifacts/vast-runs/vast-41988240-20260621T184553Z`

This report reconstructs the long Vast.ai Qwen3-VL rollout and judging run from the local fetched artifacts, repo code, memory from the active session, and five read-only subagent analyses. It also records the later dedicated insider-trading rollout rescue on the 8x A100 instance. The important distinction is that the final outputs are complete, but the local artifact bundle does not include every original intermediate shard JSON. It does include final rollout JSONs, logs, rebalance scripts, rescue scripts, and enough state to understand what happened.

## Executive Summary

The run succeeded. The final Qwen rollout set is downloaded locally:

- Roleplaying: `371/371` rows, `371` graded values, no missing source indices.
- Sandbagging: `366/366` rows, `366` graded values, no missing source indices.
- Insider trading on-policy: `173/173` rows, no missing source indices.
- Insider trading doubledown on-policy: `91/91` rows, no missing source indices.

Roleplaying and sandbagging are judged. The dedicated insider rollout files are generated and merged; they are not graded by `grade-rollouts` because the repo currently only implements rollout graders for `roleplaying` and `sandbagging_v2`. That is fine for activation extraction if the prompt labels are the target labels.

The main engineering lesson is that the model workload scales across independent examples, not by splitting one row's autoregressive decode across GPUs. The original fixed sharding was a decent first step, but it created tail risk: when a few examples ran very long, many GPUs were idle. The successful intervention was to re-chunk missing rows into single-example jobs, run those on idle GPUs, and merge only when coverage hit 100%.

The main operational lesson is that dynamic queueing should be first-class. A future run should not start with equal static shards. It should start with a resumable work queue, one persistent Qwen process per GPU, small chunks, heartbeats, missing-index monitoring, and automatic hedging for stragglers.

The later insider run improved this further: two persistent Qwen workers per A100 kept all eight A100s at roughly `99-100%` utilization. One worker per GPU used only about `17.6 GB` and roughly `45%` GPU utilization. Two workers per GPU used about `35-41 GB` per GPU and saturated the cards. Once the workers were hot and fully saturated, the remaining rollout finished and merged in just under 30 minutes:

```text
first stable saturated poll: 2026-06-21T22:11:32Z
coverage complete:          2026-06-21T22:41:14Z
merge complete:             2026-06-21T22:41:24Z
steady-state wall time:     29m52s
```

The main setup lesson is that the instance environment was usable but messy. The 8x instance had Python 3.12 in `/venv/main` while the project expected Python 3.11. `uv run --no-sync` made that work, but every command warned about the mismatch. There was also a stale editable-install `.pth` warning. The run succeeded anyway, but future setups should use a fresh Python 3.11 env or intentionally update the project pin.

## Final Output State

Final local files:

- `data/rollouts/roleplaying__plain__qwen3-vl-8b-thinking.json`
- `data/rollouts/sandbagging_v2__wmdp_mmlu__qwen3-vl-8b-thinking.json`
- `data/rollouts/insider_trading__onpolicy__qwen3-vl-8b-thinking.json`
- `data/rollouts/insider_trading_doubledown__onpolicy__qwen3-vl-8b-thinking.json`

Final local row counts:

```text
roleplaying__plain:                    371
sandbagging_v2__wmdp_mmlu:             366
insider_trading__onpolicy:             173
insider_trading_doubledown__onpolicy:   91
```

The older transcript-style insider sweep remains separate and is not one of these rollout files:

- `data/insider_trading/qwen3-vl-8b-thinking-generations.json`

Final roleplaying label counts:

```text
1: 93
2: 13
3: 8
4: 9
5: 13
6: 21
7: 202
skip: 12
```

Final sandbagging label counts:

```text
skip: 211
honest: 74
deceptive: 55
ambiguous: 26
```

Interpretation for activation use:

- Roleplaying has `329` likely binary usable rows if labels `1` and `2` are honest and labels `5`, `6`, `7` are deceptive. There are `42` non-binary or skip-like rows.
- Sandbagging has `129` binary rows: `74` honest and `55` deceptive. It has `237` skip or ambiguous rows.

One nuance: the 12 roleplaying `skip` rows are judge-error skips, not necessarily semantic skips. Their source indices are:

```text
18, 139, 163, 213, 246, 343, 346, 347, 348, 349, 350, 353
```

These are already marked as graded by the current schema. If we want to retry only those rows, use `--overwrite` and explicit `--source-index` filters. A full roleplaying rejudge is not required unless we want cleaner labels for those 12 rows.

## Evidence Pointers

Important local evidence:

- `logs/setup-new-8x.latest.log`: setup on the new 8-GPU instance.
- `logs/qwen-sharded.latest.log`: initial 4-shard generation.
- `logs/sandbagging-supervisor.latest.log`: early progress monitoring.
- `data/rebalance_runs/rebalance-20260621T092739Z/run.sh`: queue-style rebalance script.
- `logs/qwen-rebalance.latest.log`: original rebalance progress.
- `logs/qwen-rebalance-resume.latest.log`: resumed rebalance on the new 8x box.
- `data/rescue_runs/rescue-20260621T181400Z/run.sh`: single-row rescue script.
- `logs/qwen-single-rescue.latest.log`: six single-row rescue workers.
- `data/rescue_runs/hedge-20260621T182648Z/run.sh`: duplicate hedge script for final missing rows.
- `logs/qwen-hedge-rescue.latest.log`: hedge status.
- `logs/qwen-merge-monitor.sh`: monitor that waited for full coverage, killed redundant sessions, merged, and graded.
- `logs/qwen-merge-monitor.latest.log`: final merge and grading evidence.
- `logs/roleplay-judge.latest.log`: roleplaying judge API failures and final count.
- `logs/sandbagging-grade.latest.log`: deterministic sandbagging grade.
- `insider_dynamic_run/qwen-insider-dynamic.latest.log`: dedicated insider dynamic run, prefetch, 16-slot execution, hedging, and merge.
- `insider_dynamic_run/run_metadata.json`: dynamic insider run settings and GPU slot metadata.
- `insider_dynamic_run/worker_logs/`: worker logs from the 16-slot insider run.
- `final_remote_logs/`: final remote log snapshot copied before pausing the instance.

Key repo files:

- `scripts/run_qwen_sharded_4gpu.sh`: checked-in fixed sharder.
- `src/intelligent_liars/models.py`: Qwen loading path, `device_map="auto"`, flash attention, bf16.
- `src/intelligent_liars/rollouts.py`: generation settings, `model.generate`, atomic JSON writes, merge behavior.
- `src/intelligent_liars/cli.py`: `generate-rollouts`, `merge-rollouts`, `grade-rollouts`.
- `src/intelligent_liars/judging.py`: roleplaying and sandbagging grading behavior.
- `PROJECT.md`: Vast.ai operational notes.
- `scripts/insider_dynamic_runner.py` on the remote: temporary dynamic queue runner used for the insider rollout rescue. The local scratch copy was removed during cleanup.

## Timeline

All log times below are UTC.

### 1. Before the 8x rescue

The project was supposed to run on a 4x A100 Vast instance using one Qwen process per GPU, not tensor parallel. That design was correct for the repo as written: one process per GPU, separate output dirs, and a later merge step.

The old data was copied from the earlier instance to the new instance through Vast's instance-copy flow. This preserved partial rollouts, logs, and rebalance state.

Roleplaying was already complete and judged by the time the final rescue work was being done. Sandbagging was the remaining problem.

### 2. Initial 4-GPU static sharding

`qwen-sharded.latest.log` shows the original static sharder started at:

```text
2026-06-21T06:16:46Z
```

It launched four shards:

```text
0 0 91
1 91 92
2 183 91
3 274 92
```

That means:

- Shard 0 handled source indices `0..90`.
- Shard 1 handled source indices `91..182`.
- Shard 2 handled source indices `183..273`.
- Shard 3 handled source indices `274..365`.

This was better than one monolithic process, but it was still fixed equal sharding. It assumed the runtime per source row would be roughly uniform. That assumption was false.

By `09:25`, the supervisor showed only `160/366` sandbagging rows present:

```text
SHARD_STATUS 160 /366 shard-0=50; shard-1=38; shard-2=20; shard-3=52
```

The fixed shards were moving, but unevenly. Some examples were much slower because Qwen Thinking can generate very long reasoning/output traces.

### 3. Rebalance queue

At `09:27:39`, a rebalance runner started:

```text
REBALANCE_START 2026-06-21T09:27:39+00:00
chunks=53
```

The rebalance script created small chunks for missing source-index ranges. The chunk plan included gaps like:

```text
chunk-0000 50 4
chunk-0011 129 4
chunk-0025 203 4
chunk-0043 328 4
...
chunk-0051 360 4
chunk-0052 364 2
```

This was the right idea: stop treating the four original contiguous shards as the scheduling unit, and instead schedule smaller missing ranges.

By `12:13:55`, rebalance had reached:

```text
done=51 running=2 pending=0 failed=0
```

The two stuck/running chunks were:

```text
chunk-0048 348 4
chunk-0051 360 4
```

This left six final missing source indices:

```text
350, 351, 360, 361, 362, 363
```

Why those six? `chunk-0048` wrote `348` and `349` but not `350` and `351`. `chunk-0051` covered `360..363` and had not produced the needed rows in the archived state.

### 4. Old instance issue and move to 8x

The old 4x instance got paused. The UI showed the copied data had landed on the new instance. The new instance was an 8x A100 SXM4 box.

This was the right moment to re-evaluate the scheduler. Simply restarting the old four-worker rebalance script on an 8-GPU machine would not use all 8 GPUs, because the script was hardcoded to:

```bash
for gpu in 0 1 2 3; do
```

Also, only two long chunks remained. Even if the script used 8 GPUs, there were only two chunks to run. This is the key reason GPU utilization looked bad.

### 5. New 8x setup

`setup-new-8x.latest.log` shows setup began:

```text
PIP_SETUP_START 2026-06-21T17:37:14+00:00
```

The setup used:

```text
Python 3.12.13 environment at: /venv/main
```

Important setup events:

- Project installed into `/venv/main`.
- `ninja` installed.
- `flash-attn==2.8.3.post1` built in `22m14s`.
- `decord==0.6.0` installed.
- Final validation printed:

```text
SETUP_OK 2.12.0+cu130 True 8
```

That means Torch was `2.12.0+cu130`, CUDA was available, and 8 GPUs were visible.

Setup ended:

```text
PIP_SETUP_DONE 2026-06-21T17:59:41+00:00
```

The setup succeeded, but two things were messy:

1. The repo's `.python-version` wants Python 3.11, while `/venv/main` was Python 3.12. Because commands used `uv run --no-sync`, uv kept warning that the environment was incompatible.
2. There was a non-fatal stale editable install warning:

```text
ModuleNotFoundError: No module named 'editables'
```

The run survived this, but a clean future run should not start from a stale mixed editable/non-editable env.

### 6. Resume rebalance on 8x

The resumed rebalance began at:

```text
REBALANCE_START 2026-06-21T18:00:12+00:00
chunks=2
```

It had:

```text
done=51 running=2 pending=1
```

Shortly after, it became:

```text
alive=2 done=51 running=2 pending=0
```

Only two workers were actually useful because only two chunk tasks existed. This is where the 8-GPU machine was mostly idle from the perspective of useful scheduling. The machine was strong enough; the queue shape was not.

### 7. Single-row rescue

At `18:14:00`, we started six independent single-row rescue workers:

```text
RESCUE_START 2026-06-21T18:14:00+00:00
indices=350 351 360 361 362 363
gpus=1 3 4 5 6 7
```

This was the right intervention. It did not destroy any existing outputs, because each source index wrote to its own isolated output directory:

```text
data/rescue_runs/rescue-20260621T181400Z/outputs/source-350
data/rescue_runs/rescue-20260621T181400Z/outputs/source-351
...
```

No two workers wrote the same file. Merge happened only after coverage was complete.

Single-row rescue durations:

```text
source 362: 18:14:00 -> 18:15:33, about 1.5 min
source 363: 18:14:00 -> 18:17:56, about 4.0 min
source 361: 18:14:00 -> 18:18:19, about 4.3 min
source 350: 18:14:00 -> 18:18:28, about 4.5 min
source 351: 18:14:00 -> 18:33:42, about 19.7 min
source 360: 18:14:00 -> 18:44:57, about 31.0 min
```

This is the clearest evidence of the tail problem. Most missing rows were fast. Two were slow. One was very slow.

### 8. Hedge rescue

At `18:26:48`, the merge monitor still saw:

```text
unique=364 missing=351,360
```

So we launched a hedge rescue:

```text
indices=351 360 351 360
gpus=1 5 6 7
```

This duplicated only the still-missing rows on idle GPUs, again with separate output directories. The hedge jobs were not used in the final merge because the ordinary rescue row `360` landed first and the merge monitor killed redundant sessions.

This was safe because duplicate attempts did not write to shared files. It was also the right kind of intervention for a tail-latency workload.

### 9. Capped probe

A separate capped probe for source `360` used `--max-new-tokens 8192` and completed one row in about `6:05`.

It was not used in the final merge.

Why this mattered:

- It showed that row `360` was slow because the default max generation budget was huge.
- It showed a possible future speed knob.
- It did not alter the final data, preserving consistency with the model-card/default generation settings.

Source `360` in the final uncapped rollout had about `61,421` output characters. That explains why it dominated the tail.

### 10. Merge and grade

The merge monitor loop watched all original shard, rebalance, and rescue outputs. It printed:

```text
MERGE_MONITOR_STATUS 2026-06-21T18:45:08+00:00 unique=366 missing=
MERGE_MONITOR_READY 2026-06-21T18:45:08+00:00
```

Then it killed redundant sessions:

```text
MERGE_MONITOR_KILL qwen_rebalance_resume
MERGE_MONITOR_KILL qwen_single_rescue
MERGE_MONITOR_KILL qwen_hedge_rescue
```

Then it merged:

```text
MERGE_MONITOR_FILE_COUNT 62
Merged rollout shards task=sandbagging_v2__wmdp_mmlu examples=366 duplicates=0
MERGE_MONITOR_MERGED 366 missing []
```

Then it graded:

```text
Graded rollout file task=sandbagging_v2__wmdp_mmlu graded=366 skipped=0
FINAL_SANDBAGGING 366 graded 366 unknown 0 labels {'skip': 211, 'honest': 74, 'deceptive': 55, 'ambiguous': 26}
MERGE_MONITOR_DONE 2026-06-21T18:45:09+00:00
```

Sandbagging grading was fast because it is deterministic/local. It is not GPU-bound and does not need OpenRouter.

### 11. Dedicated insider dynamic sharding

After the original roleplaying and sandbagging work, we found an important missing scope: the dedicated insider rollout-prompt tasks had not been generated. These are separate from roleplaying prompts that happen to mention insider trading.

The missing rollout-prompt tasks were:

```text
insider_trading__onpolicy: 173 examples
insider_trading_doubledown__onpolicy: 91 examples
total: 264 examples
```

The first dynamic insider launch exposed a storage problem on the 8x A100 instance. The container overlay had only `16 GB`, while the Qwen cache needs about `21 GB`. Eight workers tried to hydrate the same Hugging Face cache under `/workspace/hf_home` and failed with:

```text
OSError: [Errno 28] No space left on device
```

No rollout outputs were lost. The run still had:

```text
pending=264
completed=0
failed=0
```

The fix was to move model cache and temp space to the large memory-backed filesystem:

```bash
/workspace/hf_home -> /dev/shm/hf_home
/workspace/tmp     -> /dev/shm/tmp
HF_HOME=/dev/shm/hf_home
TMPDIR=/dev/shm/tmp
```

Then Qwen was prefetched once before worker launch. Prefetch took `1m46s`:

```text
PREFETCH_START 2026-06-21T21:57:38Z
PREFETCH_DONE  path=/dev/shm/hf_home/hub/models--Qwen--Qwen3-VL-8B-Thinking/...
PREFETCH_END   2026-06-21T21:59:24Z
```

The temporary dynamic runner used a per-source queue:

- one queue unit per source row,
- persistent worker processes that load Qwen once,
- isolated per-source output directories,
- `pending/`, `running/`, `done/`, `failed/`, `completed/`, `heartbeats/`, and `logs/` directories,
- completion markers keyed by task and source index,
- merge only after coverage is complete,
- automatic hedge attempts for the tail once ordinary pending work is empty.

The first healthy version used one worker per A100:

```text
workers=8
GPU memory: about 17.6 GB / 80 GB each
GPU utilization: about 43-46%
```

That was safe but left throughput on the table. Because each A100 had enough memory headroom, we patched the runner to support unique worker slot IDs and relaunched with two Qwen workers per GPU:

```text
--gpus 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7
workers=16
```

This doubled the number of active decode streams while staying well under the 80 GB VRAM limit:

```text
GPU memory: about 35-41 GB / 80 GB each
GPU utilization: 99-100%
heartbeats: 16 active worker slots
failed tasks: 0
```

The key steady-state timeline:

```text
16-slot supervisor start:       2026-06-21T22:04:29Z
first stable saturated poll:    2026-06-21T22:11:32Z, completed=57/264
coverage complete:              2026-06-21T22:41:14Z
merge complete:                 2026-06-21T22:41:24Z
steady-state saturated runtime: 29m52s
```

The end-to-end wall time from the prefetch launch to merge was longer:

```text
launch start:     2026-06-21T21:57:38Z
merge complete:   2026-06-21T22:41:24Z
end-to-end time:  43m46s
```

The practical result is still very good: after model cache setup and worker warmup, the actual saturated rollout production finished in about 30 minutes while all GPUs were fully used.

Tail hedging worked as intended. When normal pending work emptied, the supervisor created duplicate attempts for slow remaining source rows. This is why the log showed more completed task files than unique source rows:

```text
done_tasks=277
unique expected rows=264
```

That was not corruption. Duplicate attempts wrote to separate output dirs and final merge deduped by source index:

```text
MERGE_START task=insider_trading__onpolicy files=173
MERGE_DONE  task=insider_trading__onpolicy rows=173 missing=[]

MERGE_START task=insider_trading_doubledown__onpolicy files=102
MERGE_DONE  task=insider_trading_doubledown__onpolicy rows=91 missing=[]
```

Final insider files:

```text
data/rollouts/insider_trading__onpolicy__qwen3-vl-8b-thinking.json            173 rows
data/rollouts/insider_trading_doubledown__onpolicy__qwen3-vl-8b-thinking.json  91 rows
```

There was one stale `running/` marker left on the remote from a terminated hedge attempt, but final coverage and the merged JSONs are complete. The stale marker is a cleanup issue, not a data integrity issue.

## How Dynamic Sharding Maxed Out The GPUs

The important distinction is that we did not shard one model across GPUs. We sharded the examples across many independent Qwen processes.

There are two different ideas that sound similar:

```text
model/tensor sharding:
  one model call split across multiple GPUs
  needs distributed inference support
  higher integration risk mid-run

example/work sharding:
  many independent examples split across many workers
  each worker owns a normal single-GPU Qwen process
  simple, robust, and ideal for a rollout dataset
```

This run used example/work sharding. Each worker set `CUDA_VISIBLE_DEVICES` to one physical GPU before loading Qwen. Inside that process, Transformers saw only one GPU, so `device_map="auto"` placed the whole Qwen model on that visible GPU. That gave us isolation without tensor parallelism:

```text
physical GPU 0 -> worker slot gpu0-slot0 -> one Qwen process
physical GPU 0 -> worker slot gpu0-slot1 -> one Qwen process
physical GPU 1 -> worker slot gpu1-slot0 -> one Qwen process
physical GPU 1 -> worker slot gpu1-slot1 -> one Qwen process
...
physical GPU 7 -> worker slot gpu7-slot0 -> one Qwen process
physical GPU 7 -> worker slot gpu7-slot1 -> one Qwen process
```

The final command shape was intentionally repetitive:

```text
--gpus 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7
```

That means "create 16 worker slots, two slots per physical A100." It does not mean the model was split across duplicate GPU IDs. It means two separate Python processes were allowed to share each A100.

### Why one worker per A100 was not enough

One Qwen decode stream is autoregressive. The next token depends on the previous token, so a single request cannot expose unlimited parallel work to an A100. Even though attention and matrix kernels are large, the overall decode loop still has CPU scheduling, token sampling, cache updates, and per-step synchronization.

That showed up directly in the hardware counters:

```text
8 workers, 1 per A100:
  VRAM:        about 17.6 GB / 80 GB per GPU
  utilization: about 43-46%
```

The model fit easily, but each GPU was waiting too often. The free VRAM was a symptom: the process was not memory-constrained, and it was not keeping the A100 fully occupied.

### Why two workers per A100 fixed it

The 8B Qwen model plus KV cache was small enough that a second independent process could fit on the same A100. Two decode streams on the same GPU gave the CUDA scheduler more ready work. When one stream was between kernels, sampling, updating state, or waiting on framework overhead, the other stream could keep the GPU busier.

The observed change was the important proof:

```text
16 workers, 2 per A100:
  VRAM:        about 35-41 GB / 80 GB per GPU
  utilization: 99-100%
  active slots: 16
  failed tasks: 0
```

This is what "maxed out" meant in practice: the bottleneck moved from "not enough concurrent decode streams" to "the GPUs are continuously busy." We did not need to fill all 80 GB. We needed enough concurrent work to saturate compute, and two Qwen processes per A100 did that.

We did not push to three or four workers per A100 because two had already saturated utilization. More workers would have increased risk without a clear benefit:

- larger aggregate KV cache,
- more memory fragmentation,
- more CPU/process scheduling overhead,
- more disk and logging pressure,
- higher chance of out-of-memory if an example produced a long answer.

### How dynamic claiming avoided idle GPUs

The earlier static sharding approach split work into fixed ranges. That is fine when every row has similar runtime. It is bad when rows vary a lot, because one shard can get stuck on long generations while the other GPUs finish and sit idle.

The dynamic runner changed the unit of scheduling from "big shard" to "one source row":

```text
pending/source-row-N
  -> worker atomically claims it
  -> running/source-row-N.attempt-K
  -> worker writes isolated output
  -> completed/source-row-N marker
  -> merge dedupes by source index
```

Workers did not own fixed ranges. They repeatedly pulled the next available source row from the queue. Fast rows freed a worker quickly; slow rows did not block unrelated work. This is why the 16 worker slots could stay fed until the tail.

The queue was file-backed because that was the simplest reliable control plane on the Vast box. The important safety rule was that a source row became complete only after the output was written and the completion marker existed. That made recovery and status checks straightforward:

```text
expected rows  = source row count
completed rows = unique completion markers
missing rows   = expected - completed
```

### How tail hedging worked without destroying outputs

At the end of a dynamic run, the ordinary queue can empty while a few long rows are still running. If all idle workers wait for one slow attempt, the GPUs go underused again. The runner handled that by launching hedge attempts for slow missing rows.

Hedging means:

```text
source row 173 has one slow attempt still running
an idle slot starts a second attempt for source row 173
both attempts write to separate attempt directories
the first successful completion marker wins
merge keeps one final row per source index
```

This is why `done_tasks` exceeded the number of unique source rows:

```text
done_tasks=277
unique rows=264
```

That was expected. It meant duplicate attempts existed for tail rows. It did not mean duplicate final data. The final merge deduped by source index:

```text
insider_trading__onpolicy:
  attempted files: 173
  merged rows:     173
  missing rows:    []

insider_trading_doubledown__onpolicy:
  attempted files: 102
  merged rows:      91
  missing rows:     []
```

The doubledown task had 11 extra attempt files, and the merge correctly reduced them to 91 unique source rows.

### The throughput result

The clearest throughput window is after cache prefetch and worker warmup:

```text
first stable saturated poll:    2026-06-21T22:11:32Z, completed=57/264
coverage complete:              2026-06-21T22:41:14Z
steady-state saturated runtime: 29m52s
```

That is the basis for the "about 30 minutes" result. The full launch-to-merge time was `43m46s`, but that includes model-cache setup and worker warmup. Once the system reached the 16-slot saturated state, the actual rollout production finished in just under 30 minutes.

## Why 8 GPUs Still Looked Underused

The critical mental model:

- GPUs can be used in parallel across independent examples.
- A single autoregressive decode is mostly sequential at the token level.
- Token `n+1` depends on token `n`.
- The current repo does not use tensor-parallel or pipeline-parallel inference.
- The current repo uses one Transformers process per GPU with `CUDA_VISIBLE_DEVICES` isolation.

The model loading path uses:

```python
Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

When the shell sets `CUDA_VISIBLE_DEVICES=4`, that process sees only one GPU, so `device_map="auto"` places the model on that one visible GPU. This is good isolation. It avoids file races. It is not tensor parallel.

Why not split one row across all 8 A100s? In principle, tensor parallel inference can split model layers or matrix multiplies across GPUs. In practice, that would require a different runtime stack and integration path, such as vLLM or a Transformers distributed/tensor-parallel setup. Mid-run, switching to that would have risked more time than it saved, especially with only a few rows left.

For this run, the fastest safe optimization was row-level parallelism plus hedging. The later insider run proved the stronger version of that idea: two independent Qwen workers per A100 can saturate an 8x A100 box for this workload, as long as the queue is dynamic enough to keep every worker fed.

## Why VRAM Was Not 80 GB Full All The Time

A100 80 GB gives memory headroom. It does not guarantee full utilization. For this workload, VRAM usage comes from:

- model weights,
- activations needed during generation,
- KV cache for the active sequence,
- batch size,
- sequence length,
- framework overhead.

Extra free VRAM does not automatically convert into more tokens/sec. To use it, we need more batch concurrency, more concurrent decode streams, a persistent worker queue, or a different serving runtime. The dedicated insider run showed this clearly:

```text
8 workers, 1 per A100:  about 17.6 GB per GPU, about 45% GPU utilization
16 workers, 2 per A100: about 35-41 GB per GPU, about 99-100% GPU utilization
```

This is the most important throughput lesson from the final run. On A100 80GB, Qwen3-VL 8B Thinking can fit more than one independent worker per GPU for these text-only rollout prompts. That does not mean every future run should blindly use two workers per GPU; it means we should benchmark worker slots per GPU and select the highest stable setting that keeps utilization high without OOMs.

We used `BATCH_SIZE=2` for normal sharded generation. For rescue and dynamic queueing, we used `BATCH_SIZE=1` because we wanted isolated single-row jobs and safe partial completion.

The logs show memory usage ranging from roughly 17 GB during early loading to 50-78 GB during long active generation. The 80 GB class was still justified because long thinking outputs can push memory high. A 48 GB GPU would be risky for this exact default `max_new_tokens=40960` setup.

## What Failed

### 1. Static equal sharding created tail risk

The checked-in helper split by source-index count, not expected runtime. That is simple and correct enough for a first pass, but it assumes all examples take similar time. Qwen Thinking violates that assumption.

The result:

- some shards advanced quickly,
- some source rows generated enormous outputs,
- after most rows completed, a tiny number of examples controlled wall-clock time,
- extra GPUs sat idle unless we manually re-chunked or hedged.

### 2. The rebalance script reloaded the model per chunk

The rebalance script's scheduling idea was good: a `pending/running/done/failed` queue.

But each chunk invoked:

```bash
uv run --no-sync intelligent-liars generate-rollouts ...
```

That reloads Qwen for each chunk. Model loading was usually only a few seconds once cached, but the pattern is still inefficient and makes monitoring noisier.

The future runner should load Qwen once per GPU and process many chunks in that persistent process.

### 3. The 8-GPU box inherited a 4-GPU script shape

The resumed rebalance used a script hardcoded to GPUs `0..3`. Even if changed to `0..7`, only two chunks were left. The real problem was not only GPU count. It was scheduling granularity.

### 4. Environment setup was not clean

The 8x instance setup worked, but with warnings:

- `/venv/main` was Python 3.12.13.
- The project expected Python 3.11.
- `uv run --no-sync` warned every time.
- A stale editable install `.pth` file tried to import missing `editables`.

These did not stop the run, but they are a reproducibility smell.

### 5. FlashAttention build time was a real tax

On the new 8x instance, `flash-attn==2.8.3.post1` took `22m14s` to build. This was the biggest setup-time cost in the fetched setup log.

The A100-specific build pin helped:

```bash
FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=8 uv pip install flash-attn --no-build-isolation
```

But the deeper fix is to use a prebuilt image or cached environment when possible.

### 6. Roleplaying judge had transient OpenRouter failures

`roleplay-judge.latest.log` shows many:

```text
Judge error on attempt N: OpenRouterAPIError
Structured roleplaying judge failed
```

The final roleplaying JSON still has `371` graded values, but 12 are `skip` due to judge errors. The good part is that `--continue-on-error` prevented the entire roleplaying file from failing. The bad part is that judge-error skips are mixed into the same label surface as ordinary skip labels.

### 7. Local artifact provenance is incomplete

The final rollout JSONs are complete.

But the local artifact bundle does not include the original `data/rollout_shards/` JSONs. The merge monitor did use them remotely. Locally, we cannot reconstruct the final sandbagging merge from intermediate files alone because some original shard JSONs were not fetched.

This is not a data-loss issue for final outputs. It is a provenance issue.

## What Went Well

### 1. One process per GPU was the right baseline

The repo does not use tensor parallelism. The right baseline was one independent process per GPU, separate output dirs, and a merge step. That avoided file-write races and used multiple GPUs where there was enough independent work.

### 2. `flush_every=1` protected partial progress

Frequent flushing made it possible to recover partial rows after pauses and interrupts. It costs some I/O, but for long cloud runs it is worth it.

### 3. Source-index metadata made recovery possible

Every rollout row has `metadata.source_index`. That made it possible to:

- count unique coverage,
- identify exact missing rows,
- create rescue chunks,
- merge by source index,
- avoid duplicating already completed work.

Without this, recovery would have been much more fragile.

### 4. Merge dedupe behavior was useful

`merge-rollouts` dedupes by `metadata.source_index` and prefers the record with more populated grading. That made it safe to merge existing final files, shard outputs, rebalance outputs, and rescue outputs.

### 5. Isolated rescue output dirs avoided destructive intervention

The intervention did not overwrite existing outputs. Each rescue/hedge attempt wrote to its own directory. The final merge was the only step that produced the canonical final file.

### 6. The merge monitor was effective

The custom merge monitor did the right operational work:

- repeatedly counted unique source indices,
- printed missing indices,
- waited for `366/366`,
- killed redundant sessions,
- merged,
- verified missing list was empty,
- graded.

This should become a real command.

### 7. Sandbagging grading was straightforward

Sandbagging grading was deterministic and local. Once generation finished, grading took essentially no time and produced complete labels.

### 8. Two workers per A100 was the right throughput setting for insider

The final insider run proved that one worker per GPU underfilled the A100s for this text rollout workload. Moving to two persistent workers per GPU:

- doubled active decode streams from 8 to 16,
- raised utilization from about `45%` to `99-100%`,
- kept VRAM comfortably below 80 GB,
- preserved output isolation,
- completed the saturated generation window in about 30 minutes.

This is the clearest positive lesson from the run. The best unit of parallelism was not "one process per GPU"; it was "as many isolated decode workers per GPU as fit and keep utilization saturated."

## Data Integrity Notes

Final files are complete by source-index coverage:

- Roleplaying: `0..370`, no missing.
- Sandbagging: `0..365`, no missing.
- Insider trading on-policy: `0..172`, no missing.
- Insider trading doubledown on-policy: `0..90`, no missing.

No empty outputs were found in the final files. The final files have the expected rollout schema:

- `input_messages`
- `expected_label`
- `input_str`
- `output_messages`
- `output_strs`
- `grader_reasoning`
- `graded_deception`
- `deception_percentage`
- `metadata`

Important caveats:

1. Roleplaying has 12 judge-error skip labels.
2. Sandbagging has 5 rows without `qwen_thinking_content`: source indices `43`, `129`, `197`, `227`, `360`. Their outputs are present; this only matters if we need thinking traces.
3. Intermediate provenance is incomplete because original `data/rollout_shards/` JSONs were not fetched.
4. `deception_percentage` is not enough for downstream filtering. Use schema-aware labels from `graded_deception`.

## Why Source 360 Mattered So Much

Most rescue rows finished in 1.5 to 4.5 minutes. Source `351` took about 20 minutes. Source `360` took about 31 minutes.

Final source `360` had:

```text
output_chars: 61421
thinking_chars: 0
label: skip
```

That means it likely generated a very long visible answer rather than a normal split `<think>...</think>` plus final answer. It was the tail.

The capped probe for `360` completed in about 6 minutes with `--max-new-tokens 8192`. We did not use it in the final merge because the run's default generation setting was:

```text
max_new_tokens=40960
do_sample=True
temperature=1.0
top_p=0.95
top_k=20
seed=0
```

Changing max tokens for one final row would have mixed generation regimes. The probe was useful as a speed diagnosis, not as final data.

## Recommended Code Changes

### 1. Add a real rollout-run orchestrator

Add `src/intelligent_liars/rollout_runs.py` with concepts like:

```python
@dataclass(frozen=True)
class RolloutChunk:
    id: str
    task: str
    source_indices: tuple[int, ...]
    attempt: int
    output_dir: Path
```

Core functions:

```python
def plan_rollout_chunks(..., chunk_size: int, existing_paths: Sequence[Path]) -> list[RolloutChunk]: ...
def run_rollout_worker(..., gpu: str, slot_id: str, chunks: ChunkQueue, bundle: ModelBundle) -> WorkerSummary: ...
def summarize_rollout_run(run_dir: Path) -> RolloutRunSummary: ...
```

Add CLI:

```bash
intelligent-liars run-rollout-workers \
  --task sandbagging_v2__wmdp_mmlu \
  --gpus 0,1,2,3 \
  --workers-per-gpu 1 \
  --chunk-size 4 \
  --run-dir data/rollout_runs/<run-id> \
  --stale-after 45m \
  --hedge-after 20m \
  --max-attempts 2 \
  --merge-when-complete
```

### 2. Keep workers persistent

Do not reload Qwen for each chunk. The future worker should:

1. set `CUDA_VISIBLE_DEVICES`,
2. load model once,
3. pull chunks from queue,
4. write per-chunk or per-source JSON,
5. heartbeat progress,
6. exit cleanly when no work remains.

Also make worker slots explicit. On A100 80GB, the insider rollout saturated the machine with:

```text
workers_per_gpu=2
total_workers=16
```

The status output should show both `gpu` and `slot_id`, because duplicate GPU IDs are valid when multiple isolated workers share one physical GPU.

### 3. Add first-class status and missing-index commands

Add:

```bash
intelligent-liars rollout-status --run-dir data/rollout_runs/<run-id> --expect-complete
```

It should print:

- expected total,
- unique completed,
- missing source indices,
- running chunks and age,
- failed chunks,
- per-GPU worker state,
- output file counts,
- last heartbeat time.

This replaces ad hoc Python snippets inside shell monitors.

### 4. Add automatic hedging

If a source index has been running longer than a threshold and an idle GPU exists, launch a duplicate attempt in a separate output dir.

Rules:

- Hedge only missing source indices.
- Never write to the same file as the original worker.
- Merge only after coverage is complete.
- Preserve duplicate-attempt metadata for audit.

### 5. Improve merge validation

Extend `MergeSummary` with:

- `missing_source_indices`
- `duplicate_source_indices`
- `conflicting_duplicates`
- `input_file_manifest`

Make merge fail if expected coverage is incomplete unless an explicit `--allow-missing` flag is passed.

For sampled generation, duplicate source indices can produce different answers. The merge command should emit a duplicate audit and require an explicit duplicate policy:

```bash
--duplicate-policy prefer-graded
--duplicate-policy prefer-first
--duplicate-policy prefer-newest
```

### 6. Add a run manifest

Every remote run should write:

```text
run_id
instance_id
git_commit
model_id
generation_settings
prompt_set
tasks
expected_source_counts
input_prompt_sha256
output_files
output_sha256
label_counts
unknown_counts
judge_error_counts
intermediate_file_manifest
```

Fetch this manifest with the artifacts.

### 7. Separate judge-error skip from semantic skip

Roleplaying currently encodes failed judge calls as `skip`, which is operationally convenient but analytically ambiguous.

Better:

```json
"graded_deception": ["skip"],
"grader_status": ["error"],
"grader_error": ["OpenRouterAPIError ..."]
```

or:

```json
"graded_deception": ["unknown"],
"grader_status": ["error"]
```

Then downstream code can distinguish:

- actual skip,
- non-binary label,
- judge failure,
- unknown/ungraded.

### 8. Add sharded grading

Do not have multiple judge workers write one JSON. Instead:

```bash
intelligent-liars grade-rollouts-sharded \
  --path data/rollouts/roleplaying__plain__qwen3-vl-8b-thinking.json \
  --source-chunk-size 20 \
  --output-dir data/grading_shards/<run-id> \
  --max-workers-per-shard 5
```

Then merge graded shard copies by source index.

Also add adaptive OpenRouter backoff if too many transient failures occur.

### 9. Add generation profiles

Do not silently change model-card defaults. Add named profiles:

```text
model-card: max_new_tokens=40960, do_sample=true, temperature=1.0, top_p=0.95, top_k=20
smoke: max_new_tokens=1024
fast-eval: max_new_tokens=8192
deterministic-extract: do_sample=false
```

Then commands can be explicit:

```bash
--generation-profile model-card
--generation-profile smoke
```

### 10. Measure tokens/sec and row runtime

Every generated source row should record:

- prompt token count,
- generated token count,
- wall time,
- tokens/sec,
- GPU id,
- worker id,
- output chars,
- thinking chars,
- finish reason if available.

This would have identified source `360` as a straggler immediately.

## Recommended Operational Runbook

### Dedicated insider rollout status

The dedicated insider-trading rollout tasks are separate from roleplaying and sandbagging. They are now complete locally:

```text
data/rollouts/insider_trading__onpolicy__qwen3-vl-8b-thinking.json             173 rows
data/rollouts/insider_trading_doubledown__onpolicy__qwen3-vl-8b-thinking.json   91 rows
```

Do not rerun them for the current experiment scope. If they need to be regenerated later, use the dynamic queue pattern, not the older static helper. The successful operating shape was:

```bash
HF_HOME=/dev/shm/hf_home
TMPDIR=/dev/shm/tmp
HF_HUB_DISABLE_XET=1
PYTHONPATH=src
UV_PROJECT_ENVIRONMENT=/venv/main

uv run --no-sync python scripts/insider_dynamic_runner.py supervisor \
  --run-dir data/rollout_runs/insider-dynamic-<timestamp> \
  --gpus 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7 \
  --batch-size 1 \
  --max-new-tokens 40960 \
  --poll-seconds 30 \
  --hedge-after 900 \
  --max-attempts 2
```

That temporary runner was uploaded during the run as:

```text
/workspace/intelligent_liars_run/scripts/insider_dynamic_runner.py
```

The repo should grow this into a real CLI command before the next large rollout.

Current caveat: `grade-rollouts` only implements graders for `roleplaying` and `sandbagging_v2`. It will not grade `insider_trading__onpolicy` or `insider_trading_doubledown__onpolicy` rollout files without adding a rollout grader for those task bases. The generated rollout files may still be useful if their prompt-set labels are sufficient for activation extraction, but they are not currently covered by the same `grade-rollouts` command.

The older transcript-style insider-trading sweep is another separate command:

```bash
tmux new-session -d -s insider_sweep 'cd /workspace/intelligent_liars && \
  export HF_HOME=/workspace/hf_home TMPDIR=/workspace/tmp HF_HUB_DISABLE_XET=1 PYTHONPATH=src && \
  uv run --no-sync intelligent-liars run-insider-trading-sweep \
  2>&1 | tee logs/insider-sweep.latest.log'
```

That writes to `data/insider_trading/qwen3-vl-8b-thinking-generations.json`, not `data/rollouts/`.

To grade that transcript-style output separately:

```bash
uv run --no-sync intelligent-liars grade-insider-trading \
  --input-path data/insider_trading/qwen3-vl-8b-thinking-generations.json
```

### Before renting

Use the repo commit from main and verify it includes the sharding/merge changes:

```bash
git rev-parse --short HEAD
```

Use 80 GB GPUs for full default Qwen Thinking rollouts. For full runs, prefer:

- 4x A100 80GB with dynamic queue from the start, or
- 8x A100 80GB only if the runner can keep 8 GPUs fed, or
- H100 80GB if the price/runtime tradeoff is better.

Avoid 48 GB GPUs for full default settings unless a smoke test proves the exact max-token/batch settings fit.

### Setup

Use a clean Python 3.11 environment. Avoid stale `/venv/main`.

Preferred shape:

```bash
cd /workspace
git clone git@github.com:p3rciv3l/intelligent_liars.git || true
cd /workspace/intelligent_liars
git pull --ff-only
git rev-parse --short HEAD

mkdir -p /workspace/hf_home /workspace/tmp
export HF_HOME=/workspace/hf_home
export TMPDIR=/workspace/tmp
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=src

UV_PROJECT_ENVIRONMENT=/venv/main uv sync --frozen --dev --no-editable --python 3.11
uv pip install ninja packaging psutil
uv pip install torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install decord
FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=8 uv pip install flash-attn --no-build-isolation
```

Then validate imports:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync intelligent-liars check-env \
  --require-cuda \
  --check-model \
  --check-openrouter
```

Also explicitly check:

```bash
uv run --no-sync python - <<'PY'
import torch, torchvision, decord, flash_attn, flash_attn_2_cuda
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
PY
```

### Generation

Until the Python queue runner exists, prefer a dynamic queue script over fixed equal shards. If using the existing helper, understand its limitation:

```bash
GPUS=0,1,2,3 BATCH_SIZE=2 FLUSH_EVERY=1 scripts/run_qwen_sharded_4gpu.sh
```

This launches one process per GPU, but it uses static contiguous source ranges.

For future robust operation:

```bash
tmux new-session -d -s qwen_rollouts 'cd /workspace/intelligent_liars && \
  export HF_HOME=/workspace/hf_home TMPDIR=/workspace/tmp HF_HUB_DISABLE_XET=1 PYTHONPATH=src && \
  intelligent-liars run-rollout-workers \
    --task sandbagging_v2__wmdp_mmlu \
    --gpus 0,1,2,3,4,5,6,7 \
    --workers-per-gpu 2 \
    --chunk-size 4 \
    --stale-after 45m \
    --hedge-after 20m \
    --merge-when-complete \
    2>&1 | tee logs/qwen-rollout-workers.latest.log'
```

### Monitoring

Monitor both GPU state and source-index coverage:

```bash
watch -n 30 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits && echo && tail -n 80 logs/qwen-rollout-workers.latest.log'
```

For current artifacts, the equivalent custom monitor was:

```bash
tail -f logs/qwen-merge-monitor.latest.log
```

### Grading

Sandbagging:

```bash
uv run --no-sync intelligent-liars grade-rollouts \
  -p data/rollouts/sandbagging_v2__wmdp_mmlu__qwen3-vl-8b-thinking.json
```

Roleplaying:

```bash
tmux new-session -d -s roleplay_judge 'cd /workspace/intelligent_liars && \
  export PYTHONPATH=src && \
  uv run --no-sync intelligent-liars grade-rollouts \
    -p data/rollouts/roleplaying__plain__qwen3-vl-8b-thinking.json \
    --max-workers 10 \
    --continue-on-error \
    2>&1 | tee logs/roleplay-judge.latest.log'
```

If only retrying the 12 current roleplaying judge-error skips:

```bash
uv run --no-sync intelligent-liars grade-rollouts \
  -p data/rollouts/roleplaying__plain__qwen3-vl-8b-thinking.json \
  --overwrite \
  --continue-on-error \
  --source-index 18 \
  --source-index 139 \
  --source-index 163 \
  --source-index 213 \
  --source-index 246 \
  --source-index 343 \
  --source-index 346 \
  --source-index 347 \
  --source-index 348 \
  --source-index 349 \
  --source-index 350 \
  --source-index 353
```

### Fetching

Always fetch into a versioned local artifact dir:

```bash
ID=<vast-instance-id>
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DEST="$PWD/artifacts/vast-$ID-$STAMP"
mkdir -p "$DEST/data" "$DEST/logs"
```

Fetch:

```bash
rsync -avP -e "ssh -p $VAST_SSH_PORT" "$VAST_HOST:$REMOTE_REPO/data/rollouts/" "$DEST/data/rollouts/"
rsync -avP -e "ssh -p $VAST_SSH_PORT" "$VAST_HOST:$REMOTE_REPO/logs/" "$DEST/logs/"
rsync -avP -e "ssh -p $VAST_SSH_PORT" "$VAST_HOST:$REMOTE_REPO/data/rollout_shards/" "$DEST/data/rollout_shards/"
rsync -avP -e "ssh -p $VAST_SSH_PORT" "$VAST_HOST:$REMOTE_REPO/data/rebalance_runs/" "$DEST/data/rebalance_runs/"
rsync -avP -e "ssh -p $VAST_SSH_PORT" "$VAST_HOST:$REMOTE_REPO/data/rescue_runs/" "$DEST/data/rescue_runs/"
rsync -avP -e "ssh -p $VAST_SSH_PORT" "$VAST_HOST:$REMOTE_REPO/results/" "$DEST/artifacts/activations/" || true
```

The missing item from this run's local artifact bundle is `data/rollout_shards/`. Fetch it next time.

### Local verification after fetch

Run:

```bash
python3 - <<'PY'
import json, collections
from pathlib import Path
base = Path("artifacts/vast-runs/vast-41988240-20260621T184553Z/data/rollouts")
for p in sorted(base.glob("*.json")):
    d = json.load(open(p))
    rows = d["rollouts"]
    idx = [r["metadata"]["source_index"] for r in rows]
    vals = [v for r in rows for v in r.get("graded_deception", [])]
    print(p.name)
    print("rows", len(rows), "unique", len(set(idx)), "missing", [i for i in range(max(idx)+1) if i not in set(idx)])
    print("labels", dict(collections.Counter(map(str, vals))))
PY
```

Only pause or destroy the instance after this passes.

## Hardware Guidance

For this exact workload:

- Minimum comfortable VRAM: 80 GB.
- Disk: do not rely on a 16 GB container filesystem. Use enough disk or move HF cache to `/workspace` or `/dev/shm` if large enough.
- CPU/RAM: not the bottleneck, but enough RAM helps with multiple Python workers and model loading.
- Network: useful for model/cache downloads and artifact fetches, but not the main runtime bottleneck once weights are cached.
- GPU count: only useful if the scheduler has enough independent chunks to feed the GPUs.

Recommended next full-run setup:

1. 4x A100 80GB with dynamic queue from the start.
2. 8x A100 80GB only if wall-clock is more important than cost and the runner can keep all GPUs fed.
3. H100 80GB if available at a price where faster tokens/sec offsets cost.

The current run proved that simply renting more GPUs is not enough. The scheduler must expose enough independent work.

## Key Lessons

1. Fixed source-count sharding is better than one process, but dynamic queues are the real solution.
2. Tail examples dominate wall-clock when `max_new_tokens=40960` and Qwen Thinking can generate huge outputs.
3. Extra GPU memory is headroom, not automatic throughput.
4. Extra GPUs help only when there are enough independent examples or hedge attempts to run.
5. Single-row rescue is safe if every attempt writes to a separate output dir.
6. Merge should be coverage-aware and fail if expected source indices are missing.
7. Run manifests matter. Final outputs are complete, but incomplete intermediate provenance is still annoying.
8. Roleplaying judging needs better failure-status encoding.
9. FlashAttention build time should be treated as part of runtime/cost planning.
10. A clean Python env matters. Warnings that do not break today still create confusion during a long run.

## What To Change Before The Next Run

Highest-impact code work:

1. Build the persistent per-GPU queue runner.
2. Add `rollout-status`.
3. Add automatic stale chunk hedging.
4. Add merge expected-coverage validation.
5. Add run manifests.
6. Fetch `data/rollout_shards/` every time.
7. Separate judge-error status from semantic labels.

Highest-impact ops work:

1. Use a clean Python 3.11 env.
2. Use a prebuilt/cached FlashAttention environment if possible.
3. Always launch long jobs in remote tmux.
4. Monitor source-index coverage, not only `nvidia-smi`.
5. Do not evaluate success by GPU utilization alone. Evaluate by completed unique source indices per hour.

## Bottom Line

The run was successful, but too manual. The manual interventions were technically sound: they preserved outputs, isolated file writes, used idle GPUs for missing rows, and only merged after full coverage. The future improvement is to make those interventions the default orchestration layer.

The next run should feel boring: start a dynamic queue, watch coverage climb, hedge stragglers automatically, merge only after expected coverage is complete, grade in isolated shards, fetch a complete artifact manifest, then destroy the instance.
