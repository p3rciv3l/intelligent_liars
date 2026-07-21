---
name: vast-gpu-experiments
description: Use when a task involves fast experiments, paper reproduction, model training, inference benchmarks, CUDA workloads, Vast.ai, cheap GPU rental, remote GPU acceleration, or "run this faster". Helps Codex choose the cheapest sufficient Vast.ai GPU, dry-run the lifecycle, execute only with explicit approval, fetch results, and clean up.
---

# Vast GPU Experiments

Use this skill when local compute is likely too slow or unsuitable for an ML, CUDA, benchmark, or paper reproduction workload. The default goal is the cheapest successful remote run, not the strongest GPU.

## Core Policy

- Default to one GPU.
- Prefer the cheapest offer that satisfies the workload's memory, disk, CUDA, reliability, and network needs.
- Escalate GPU class only for a concrete reason: VRAM, CUDA capability, tensor throughput/runtime economics, or an explicit multi-GPU requirement.
- Dry-run before spending. Do not rent unless the user explicitly asks to execute or approves the printed create command.
- Clean up rented instances after fetching outputs unless the user asks to keep them.
- Before any rental, confirm with the user using concrete instance and price details: offer ID, GPU type/count, VRAM, hourly price, estimated total cost, image, disk, workload command, and cleanup behavior.
- Treat shutdown as part of the experiment, not an optional afterthought. Default to destroying the instance after artifacts are fetched, then verify no matching labeled stale instances remain.
- Do not store API keys or secrets in the skill or generated metadata.
- Preserve durable run knowledge. After each run, write or update the project-local run record when a repo has one, and otherwise create a local run record outside the skill. The record should tie together commands, metadata, logs, artifacts, performance lessons, and cleanup state.

## Workflow

1. Inspect the repo or paper enough to infer framework, setup command, workload command, dataset size, expected VRAM, runtime, and output artifacts.
2. Use `scripts/vast_find_offer.py` to search and rank Vast offers. Start with single GPU and the smallest credible `--min-vram`.
3. Show the user the recommended offer, hourly price, estimated total cost, Docker image, create command, workload command, fetch paths, and cleanup command.
4. Ask for explicit confirmation before spending. The user must approve the instance and price details, not just say "go ahead" to an unspecified run.
5. Use `scripts/vast_run_workload.py --dry-run` to print the lifecycle for create, sync, run, fetch, metadata, and cleanup.
6. Execute only with both `--execute` and `--confirmed-cost-approval` after explicit approval.
7. If the user asks to keep the GPU running, require a concrete reason and use `--keep-instance --confirm-keep-instance`; otherwise destroy it automatically.
8. After completion or interruption, run `scripts/vast_cleanup.py --dry-run` to check for stale `codex-vast-` instances. Use `--execute` when cleanup is confirmed or clearly part of the approved run.
9. Update the durable run knowledge store before final handoff. Prefer the active repo's existing docs convention, usually a central run note plus one flat run-specific note. Use a per-run folder only when the run has multiple useful sidecars. If the active task has no repo-level store, create a timestamped run folder under `~/Documents/Codex/vast-runs/` and mention it in the final answer.

## Bundled Scripts

- `scripts/vast_find_offer.py`: searches `vastai search offers --raw`, filters sufficient machines, and ranks by expected total cost.
- `scripts/vast_run_workload.py`: dry-run-first lifecycle wrapper for create, sync, SSH run, fetch, metadata, and optional destroy.
- `scripts/vast_cleanup.py`: finds stale labeled Vast instances and destroys only matching stale instances when executed.

The scripts default to `/Users/student/Library/Python/3.10/bin/vastai`. Override with `--vastai-path` or `VASTAI_PATH`.

## References

- Read `references/experiment-workflow.md` for paper/repo-to-GPU workflow and cost guardrails.
- Read `references/vast-cli.md` for practical Vast CLI command patterns and troubleshooting notes.
