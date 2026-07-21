# Paste this into the existing Capy Captain thread `SCR-001`

Continue as the primary Captain for this project. This message adds required repository, VM, and Vast.ai operating context. It does **not** authorize implementation, Build-agent delegation, a Capy capability test, a GPU rental, or an evaluation run. First absorb and verify the context, then report back and wait for Avinash's explicit approval.

## Immediate scientific objective

The immediate comparison is:

1. Unmodified `Qwen/Qwen3-VL-8B-Thinking` baseline.
2. Full 5-shot MMLU over all 57 subjects and 14,042 test questions.
3. Official OSWorld 369-task suite, initially at a declared maximum of 15 steps per task.
4. Preserve complete OSWorld trajectories: initial and per-step screenshots, raw model response/reasoning, parsed action, execution result, reward, timestamps, errors/retries, final score, runtime logs, and recording.
5. Only after the baseline is complete and validated, rerun the identical frozen harness against one or more named layer-21 abliteration or direction-reflection checkpoints.

Capy is the orchestrator. Capy must not choose the desktop actions being evaluated. The OSWorld harness must send observations to the Qwen endpoint, parse Qwen's responses, execute Qwen's actions, and score the resulting desktop state with OSWorld's task-specific evaluator. KL divergence is not the OSWorld score; it would be an optional, separate model-drift diagnostic.

## Read the repository documentation first

Use GitHub `main` at or after `c1a73d3c4cf011bf33492b5694d6926fe8d78a5b` as your filesystem source of truth. Read these tracked files before proposing work, in this order:

1. `PROJECT.md` — model contract, research thesis, experiment loop, steering notes, Vast.ai guidance, benchmark framing, evaluation infrastructure, and success criteria.
2. `README.md` — current implementation state, supported commands, code map, data/artifact conventions, development checks, and run-note conventions.
3. `docs/validation/no_insider_seed_robustness_20260709.md` and `docs/validation/no_insider_gpu_sensitivity_20260709.md` — the completed multi-seed and Apple-MPS robustness evidence for the selected layer-21 candidate.
4. `docs/validation/no_insider_dense_15_27_primary_sweep_20260705_by_layer.md` — the checked-in evidence that ranks `no_repe_layer21_c001` as the primary layer-21 candidate.
5. `docs/validation/no_insider_sparse_probe_20260628.md` — the earlier sparse baseline and selection context.
6. `docs/activation_full_20260622.md` — activation-extraction artifact state, failures, integrity lessons, and pre-probe gates.
7. `docs/vast_qwen_rollouts_20260621.md` — the prior Vast/Qwen run, dynamic work sharding, worker-per-GPU measurements, failure recovery, artifact fetching, and cleanup lessons.
8. `docs/handoffs/20260707_github_dvc_drive_durability_handoff.md` — DVC and remote-artifact durability requirements.
9. `docs/probe_sweep_knob_universe_20260701.md` and `sweep_plan.md` — broader sweep context; do not let them expand the immediate paper-shaped baseline-versus-edited evaluation.

The local OSWorld scaffolding under `src/intelligent_liars/osworld_eval.py` plus `tests/test_osworld_eval.py` is intentionally not published because it is unfinished. Do not independently recreate or overwrite it during orientation.

## Capy VM concerns that must be proved

Before Capy can replace OSWorld's official AWS host/client arrangement, design one bounded, force-stopped capability probe that proves all of the following:

1. The Capy task VM exposes `/dev/kvm` or another adequately accelerated virtualization path.
2. Docker Engine and Compose work, and OSWorld's Docker/QEMU desktop can obtain the required privileges, devices, mounts, networking, and capabilities.
3. The official OSWorld desktop image boots successfully and can be reset between tasks.
4. CPU, RAM, disk, and `/dev/shm` are adequate; identify the actual large writable mount rather than assuming `/workspace` is large.
5. Containers have reliable outbound access to an authenticated OpenAI-compatible Qwen endpoint hosted on Vast.
6. Required ports and controller-to-desktop networking work without assuming undocumented public inbound access to the Capy VM.
7. A task can run for the required duration, and the usable concurrent-session/environment limits are known.
8. `traj.jsonl`, screenshots, runtime logs, `result.txt`, and `recording.mp4` can be incrementally exported to durable storage before the Capy VM is destroyed.
9. Forced stop and cleanup leave no Docker/QEMU processes, containers, volumes, or billable resources behind.

The probe must distinguish a missing KVM device, insufficient privilege, image/setup failure, network failure, artifact-egress failure, and benchmark/agent failure. Do not weaken the official evaluator or silently substitute a different desktop environment merely to make the probe pass.

## Vast GPU skill policy

Avinash's GPU lifecycle skill is published in this repository at `.agents/skills/vast-gpu-experiments/`, the location recommended by Capy's skills documentation. Load that skill before any Vast planning or action. Its policy includes:

- Default to one GPU and the cheapest offer that satisfies VRAM, CUDA capability, disk, reliability, ports, and network requirements.
- Escalate GPU class or GPU count only for a measured VRAM, compatibility, throughput, or total-cost reason.
- Inspect the repository and workload before searching: framework, image/CUDA requirements, model revision, endpoint command, caches, data size, output paths, expected VRAM, expected runtime, and cleanup needs.
- Before spending, present the exact Vast offer ID, GPU type/count, VRAM, reliability, disk, location when relevant, hourly price, maximum estimated runtime and total cost, Docker image, create command, workload command, artifact-fetch paths, and destroy command.
- A vague “go ahead” is not priced-offer approval. Do not create an instance until Avinash approves those concrete details.
- Dry-run the full lifecycle first: create, sync/setup, start endpoint, health check, benchmark concurrency, run, fetch, validate, record metadata, destroy, and confirm no matching stale instances remain.
- Never print, commit, or bake Vast, Hugging Face, model-endpoint, or artifact-store secrets into an image or repository file.
- Unless Avinash explicitly approves keeping an instance alive for a concrete reason, destroy it immediately after validated artifacts are fetched. Billing continues until destruction, not merely shutdown.
- Use unique `codex-vast-*` labels and unique immutable run paths. Never overwrite valid outputs.
- Record offer ID, instance ID, hardware, price, image, disk, Git commit, exact commands, timestamps, runtime, exit code, artifact checksums/paths, and verified destruction state in the repository's established run-note convention.
- For Qwen autoregressive inference, measure useful completed requests per hour, latency, OOMs, and parser success rather than chasing GPU utilization alone. Benchmark concurrency progressively.
- Prior project evidence showed that one isolated model worker per A100 underfilled the earlier rollout workload, while two workers per A100 used roughly twice the VRAM and reached near-full utilization. That is a prior measurement, not permission to hardcode two workers for vLLM/OSWorld; re-benchmark the actual serving stack.
- Prefer a dynamic queue with small resumable units, explicit task identity, coverage validation, no-overwrite semantics, and straggler recovery for tail-heavy generation.
- API judging is network-bound and should not be conflated with GPU inference.

Use the bundled helpers from the repository skill package:

- `.agents/skills/vast-gpu-experiments/scripts/vast_find_offer.py`
- `.agents/skills/vast-gpu-experiments/scripts/vast_run_workload.py`
- `.agents/skills/vast-gpu-experiments/scripts/vast_cleanup.py`

Do not recreate or fork those scripts in this orientation phase. Confirm that Capy discovered the project skill and report any loading or runtime incompatibility exactly.

## Billing and authority boundary

- GPT-5.6 Sol is routed through Avinash's connected Codex subscription.
- External-subscription fallback is set to **Error immediately**; never switch to Capy model credits.
- Capy platform/VM runtime costs still apply.
- Do not start a Build task, paid Capy probe, Vast instance, AWS resource, model download, MMLU run, or OSWorld run without Avinash explicitly authorizing that next concrete action.
- Preserve unrelated work and do not create a PR merely to persist planning context.

## Required response

Respond with a concise orientation report containing:

1. Which listed Markdown files you successfully read on GitHub `main` and the most relevant facts from each.
2. Confirmation that the project-local `vast-gpu-experiments` skill loaded successfully, plus acknowledgement that the unfinished local OSWorld scaffolding is unavailable.
3. The exact Capy VM capability-probe design, including commands/checks, success criteria, hard deadline, expected Capy VM charge, artifact-egress proof, and cleanup proof—but do not execute it.
4. The exact information you would require before recommending a Vast offer.
5. Any compatibility adjustment required to use the bundled Vast helpers inside Capy's Ubuntu VM; do not make that adjustment yet.
6. Any conflicts, missing evidence, or questions that genuinely block the next decision.
7. A final explicit statement that no Build task, VM probe, GPU rental, or evaluation has been started.

Then wait for Avinash's next message.

## Suggested skills

- Repository inspection and handoff reading.
- Official OSWorld harness/provider research.
- Cloud lifecycle and spend-safety review.
- `vast-gpu-experiments` only after the actual package is installed or attached.
- `code-review` for future implementation diffs, not for this orientation response.
