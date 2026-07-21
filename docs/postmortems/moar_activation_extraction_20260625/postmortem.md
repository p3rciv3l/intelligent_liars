# Moar Activation Extraction Postmortem

Date: 2026-06-25

Workspace: `/Users/student/Desktop/ai/intelligent_liars`

Target session: `019ef6d5-d4bb-7853-9ecc-d22df1b9fe5f`, the long `moar activation extraction...` Codex session. The exact-title older session `019edc8b-5c0f-70a3-953f-42cfbd7af8e3` was much smaller and was not treated as the target.

## Executive Summary

This postmortem only analyzed 0-55% and 98-100% as chunks of the whole codex session, given that sections 55%-98% were just polling jobs. That is generally what the 55% refers to here.

The goal was mostly achieved for the pre-probe gate: all text activation artifacts were eventually merged into a final all-text HDF5, validated, DVC-uploaded, and no new probe training was started. The final-tail evidence supports "all text activation data is ready before probes," not "probe/truth-direction training is done."

The technical architecture moved in the right direction. The activation pipeline had processor-aware masking, HDF5 metadata, shard merging, duplicate-key checks, NNsight/Transformers backend boundaries, safe fetch helpers, validators, and run documentation. Those are the durable wins.

The operational control plane was the failure. The session repeatedly mixed code edits, remote GPU work, DVC uploads, artifact validation, remote recovery, and handoff writing in one live stream. That produced duplicate writers, stale queue state, destructive cleanup of live temp files, wrong commands, weak locks, and high-pressure overclaims before evidence was complete.

The best one-sentence diagnosis is:

> Sound activation-extraction architecture, weak long-running operations discipline.

## Final Observable Outcome

From the allowed final tail, the endpoint was good for the pre-probe objective.
The durable evidence report is
`docs/postmortems/validation/activation_all_text_20260624.md`; it records the canonical
artifact, DVC pointer, SHA/size/task/layer validation, and the commands that
produced the proof without committing raw logs or transcript dumps.

- Final HDF5: `artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5`.
- Size: `62,394,047,728` bytes.
- DVC pointer: `artifacts/activations/activation_all_text_20260624/extracted_feats_all_text_qwen3-vl-8b-thinking.h5.dvc`.
- DVC md5: `b6e82b698513d2372949f2752e17005a`.
- SHA-256: `c6e687f69256544121d0718cf8bba142ed5837221976121e1899552dc76f1a5a`.
- HDF5 validator: `OK`, `20` tasks, `36` layers, hidden dim `4096`, sample finite check over `720` sampled datasets, `0` non-finite.
- DVC remote status: final tail reports `Cache and remote 'gdrive-artifacts' are in sync.`
- Insider JSON validator: `520` records, `ungraded_count=0`, `binary_report_usable=6`.
- Probe boundary: no new probe process/artifact modification was observed. There is one older probe artifact from 2026-06-22, so the precise claim is "no new probe training," not "no probe artifacts exist."
- Goal status: the old goal was marked complete after the final validation/upload audit.

The important residual risk is not artifact correctness, but repository hygiene: the final repo state remained broad and dirty, with many modified/untracked scripts, tests, logs, `.dvc` pointers, and docs not frozen into a commit.

## State At The 55% Cutoff

At the last allowed early point, the work was not yet in a clean handoff shape.

- GPU work was stopped or unreachable.
- Static DVC push was alive and holding DVC locks.
- Rollout4, static Truth Spec, and fresh Qwen sycophancy activations had local validation evidence.
- Fresh sycophancy was the intended final merge input, not the older sycophancy artifact unless explicitly selected.
- The first insider upscale run was unusable for direction extraction: only `26` records, `explicit=1`, `concealed=0`.
- Corrective insider `s20` state was supposed to be `520` planned, `406` valid outputs, `114` pending, `0` running, `0` failed.
- A dangerous mismatch had appeared: plan/preflight expected `done=406`, but actual local snapshot/plan-only status reported `done=812`, `outputs=406`, `pending=114`.
- Probe training was still blocked.

That was the point where the work should have stopped and produced a frozen handoff: artifact table, live processes, exact locks, known mismatches, exact next command, and explicit "do not run" commands.

Instead, the session kept polling, hardening code, checking DVC, and preparing recovery. Some of that work was useful, but it increased cognitive load and made the handoff harder to trust.

## What Worked

### 1. The Goal Was Correctly Protected

The session repeatedly preserved the central constraint: do not train probes yet. The work stayed pointed at completing text activation data first, then leaving probe/direction training to a later phase.

This was the right scientific boundary. Training probes before the all-text activation store was complete would have created partial, hard-to-interpret results and would have encouraged premature conclusions.

### 2. The Core Activation Architecture Was Substantive

The code direction was not superficial.

- `ActivationBackend` separated data/masking from model-hooking.
- Transformers and NNsight backends shared detection/logit mask semantics.
- Qwen chat rendering used processor-aware text rendering and token offset mapping.
- Detection masks stayed extraction-only; they were not model attention masks.
- HDF5 metadata preserved task, labels, raw labels, token positions, logit positions, decoded/detected text, char spans, rendered text, model/backend/processor metadata, and source identity.
- Merge validation rejected duplicate `(task, source_index, output_index)` rows.

This was the right foundation for later truth-direction/probe work.

### 3. Validation Became Real, Not Decorative

Several validations were meaningful:

- Smoke activation runs before full extraction.
- HDF5 layer/task/hidden-dim checks.
- Balanced sycophancy checks: `2,500` positive and `2,500` negative examples.
- Duplicate example checks before/at merge time.
- SHA and size checks for fetched artifacts.
- DVC remote status after upload.
- No-new-probe audit before marking the goal complete.

The validator was wrong at first in several ways, but those failures were handled by fixing the validator rather than blindly distrusting valid artifacts.

### 4. Artifact Transfer Discipline Improved

The eventual safe fetch pattern was correct:

- Verify remote size and SHA.
- Fetch into a fresh sibling temp file.
- Verify temp size/SHA locally.
- Atomically promote only after validation.
- Do not append-resume large HDF5s.

That pattern should remain the only approved path for multi-GB activation artifacts.

### 5. Some Bad Results Were Correctly Rejected

The session correctly refused to extract an insider direction from the initial one-sided/one-example-ish insider output. It also treated duplicate writers and failed grading attempts as audit problems rather than silently pressing forward.

## What Failed

### 1. The Control Plane Was Missing

The biggest failure was not the model code. It was the absence of a first-class run control plane.

Symptoms:

- Multiple supervisors or tmux sessions could target the same run/output.
- Merge writers could overlap on the same final HDF5.
- Queue repair was based on unit/chunk identity before it was based on semantic output validity.
- `done` markers could diverge from actual output shards.
- Run directories could contain shards from mixed queue plans.
- Hidden temp HDF5 files were deleted while workers were still alive.

Consequence: live production runs required manual surgery, and manual surgery created new failures.

### 2. Runbooks Were Consulted Too Late

The session initially conflated lessons from autoregressive rollout generation with activation extraction. The rollout runbook's high-utilization advice did not directly apply to deterministic forward-pass extraction. The activation runbook already suggested one worker per A100 could be enough.

This late consultation led to unnecessary tuning churn: stopping clean runs for low `nvidia-smi` utilization, changing chunk sizes midstream, relaunching with different worker counts, and later repairing the resulting state.

### 3. Live Cleanup Broke Live Work

The worst concrete operational mistake: hidden temp shards were deleted while workers were still alive, causing `FileNotFoundError` failures. That is a straightforward invariant:

> Never clean temp shard files unless the run lock proves no worker can still own them.

This should be enforced in code, not left to operator memory.

### 4. Duplicate Writers And Mixed Plans Got Too Far

Two merge writers targeted the same output. A polluted run directory contained shards from multiple queue plans. Deduping by chunk id was insufficient because different plans can encode the same examples under different chunk ids.

The right invariant is semantic:

> Completion derives from validated output shards and `(task, source_index, output_index)` coverage, not from `done/` marker count alone.

### 5. Several Handoff Commands Were Not Copy-Paste Safe

Specific command and CLI failures recurred:

- Bare `python` failed because it was not always on `PATH`.
- `pytest` was not always available as a bare command.
- A nonexistent `run_activation_extraction_dynamic.py merge-activation-shards` command was written before being corrected to the Typer CLI surface.
- Validator examples used invalid `--expected-task-rows` syntax before correction.
- `rg` treated a pattern beginning with `--compression` as an option.
- macOS/GNU shell differences, such as `find -printf`, caused avoidable failures.
- Remote heredocs and nested quoting failed.

The project needs a "known-good command surface" that is tested before being put in a handoff.

### 6. Remote/Vast Handling Was Too Ad Hoc

Remote work had several fragile spots:

- The Vast login banner reportedly told agents to read `/etc/vast-agents-guide.md`; there was no clear evidence this happened before actions.
- `/dev/shm` was treated too much like durable state during a restart scare.
- Remote disk and cache preflight was incomplete before relaunching many workers.
- A broad `pkill -f` likely killed the remote shell or unrelated matching process.
- Vast CLI was unusable due login `403`, and SSH endpoint loss blocked recovery.

The correct mental model is:

> `/dev/shm` is a fast cache, not the source of truth. Persistent checkpoints, local snapshots, and validated output shards are the source of truth.

### 7. Overclaims Appeared During Stress

During the restart/panic phase, "No progress was lost permanently" was stated before enough evidence had been gathered. The safer statement would have been:

> The local snapshot appears intact; I am verifying remote state now.

The same pattern appeared with "I fixed it" before commands were actually run and confirmed.

### 8. DVC State Was Sometimes Reported As More Certain Than It Was

At some points, the only DVC evidence was an active process or lock, not cloud completion. The final tail eventually proved sync success, but earlier slices should have used stricter wording:

- "DVC pointer exists" is not "DVC pushed."
- "DVC process still running" is not "remote is in sync."
- "Lock exists" should be a hard stop unless the owning process is known dead.

### 9. Final Repo State Was Not Committed

The final artifact state was good, but the repo state was not frozen. Current `git status` shows modified and untracked files across docs, scripts, source, tests, artifact pointers, logs, and `plan.md`.

That means the final evidence is reproducible on this machine, but not yet safely preserved in Git. For this project, large bytes can stay ignored, but `.dvc` pointers, validators, run scripts, tests, and final docs should be committed deliberately.

## Root Causes

### Root Cause 1: No State Machine For Long Runs

The task involved multiple asynchronous systems: GPU workers, queues, HDF5 shards, DVC locks, rclone upload, remote state, local snapshots, and Codex agents. The repo had useful scripts, but not a single authoritative state machine.

Without a state machine, the assistant had to infer state from directories, process lists, partial logs, and ad hoc commands. That is how stale `done` markers, duplicate writers, and mixed plans became possible.

### Root Cause 2: Too Much Work In One Active Thread

The session tried to do all of these at once:

- Implement activation code.
- Launch remote GPU extraction.
- Repair remote queues.
- Validate HDF5s.
- Monitor DVC.
- Fix validators.
- Write docs.
- Answer user panic.
- Prepare next-step commands.

That is too much for one live conversational control loop. Long-running GPU/data work needs phase boundaries.

### Root Cause 3: Optimization Was Driven By Live Utilization Instead Of End-To-End Throughput

The run chased GPU utilization instead of stable completed examples per hour and merge reliability. For activation extraction, low instantaneous GPU utilization can be acceptable if the workflow is deterministic, stable, and I/O-bound.

### Root Cause 4: The System Let Dangerous Operations Be Easy

It was easy to:

- Start a second supervisor.
- Start a second merge.
- Delete temp files.
- Rerun a destructive launcher.
- Generate a giant `--plan-only` dump.
- Use `--no-verify-masks` in production.

These should be structurally hard or require explicit force flags plus audit records.

## What Should Have Been Done

### Before Remote Work

1. Read the relevant local runbooks and remote Vast guide first.
2. Produce a preflight artifact table: source inputs, expected tasks, expected examples, output paths, DVC target paths, and required secrets.
3. Run one representative smoke task through the exact production path with mask verification enabled.
4. Use one worker per A100 unless a bounded benchmark proves a higher worker count improves completed examples/hour without increasing failure rate.
5. Create a run manifest with `run_id`, `queue_plan_id`, git commit, command, model, processor, task list, chunk settings, output root, and expected counts.

### During Remote Work

1. Use a run-dir lock with owner PID, host, tmux session, command, and started-at.
2. Refuse a second supervisor or merge by default.
3. Treat `done/` as advisory; treat validated output shards as authoritative.
4. Write temp shards in worker-private temp paths or with ownership metadata.
5. Never clean temp files while a worker lock/process exists.
6. Keep monitoring low-frequency and evidence-oriented.

### Before Handoff

1. Stop adding new work.
2. Write a frozen state table.
3. Separate validated, live, blocked, and unknown states.
4. Include exact next commands and exact commands not to run.
5. Include current DVC lock owner and whether cloud sync is proven.
6. Include queue/output mismatch warnings at the top.

### Before Marking Complete

1. Validate final HDF5 with task/layer/dim counts.
2. Compute and persist final SHA-256.
3. Verify DVC remote sync, not just pointer creation.
4. Validate source JSONs and label distributions.
5. Audit no new probe training.
6. Commit or explicitly stage a follow-up freeze commit for all `.dvc`, docs, scripts, tests, and validation logs needed to reproduce the result.

## Concrete Improvements

### Add Run Locking

Every dynamic run should have a lock file:

```text
run.lock
owner_pid=<pid>
host=<host>
tmux=<session>
started_at=<iso timestamp>
command=<full command>
run_id=<uuid>
queue_plan_id=<hash>
```

Commands should refuse to run if the lock exists and the owner is alive, unless a `--force-stale-lock` path proves the process is dead.

### Add Plan IDs Everywhere

Every unit JSON and shard HDF5 should record:

- `run_id`
- `queue_plan_id`
- source task list hash
- chunk settings
- source example key range or manifest

Merge should refuse mixed plan IDs unless an explicit repair command creates a new clean manifest.

### Make Completion Output-Based

Resume status should report:

```text
planned=520
valid_outputs=406
pending=114
running=0
failed=0
done_without_output=0
output_without_done=0
duplicate_output_keys=0
```

If `done` differs from `valid_outputs`, the status must be red and resume must repair from outputs, not marker counts.

### Add Merge Locks

`merge_activation_hdf5_shards` should create an output-level merge lock. A second merge should refuse the same target unless the lock is stale and proven safe.

### Keep Mask Verification On

`--no-verify-masks` should be diagnostic only. Production extraction should require verification or an explicit sampled verification policy with an audit record.

### Make The Validator Friendlier

The validator should print examples of correct syntax:

```bash
--expected-task-rows task_a=10 --expected-task-rows task_b=20
```

or:

```bash
--expected-task-rows task_a=10,task_b=20
```

It should also persist a JSON report containing SHA, task counts, layer counts, finite-check mode, and exact file path.

### Separate DVC States

Reports should always distinguish:

- file exists locally
- `.dvc` pointer exists
- DVC cache contains object
- remote contains object
- remote sync was freshly checked

Never collapse these into "DVC done."

### Use Known-Good Commands

Handoffs should avoid bare `python` and untested shell fragments. Use repo-known commands, for example:

```bash
UV_NO_SYNC=1 PYTHONPATH=src uv run --no-sync pytest ...
UV_NO_SYNC=1 PYTHONPATH=src uv run --no-sync intelligent-liars ...
```

Do not call `merge-activation-shards` through the dynamic runner; that command belongs to the project CLI surface.

### Add A Frozen Handoff Template

Use this template before every pause:

````markdown
## Frozen State

### Validated Artifacts
| artifact | local path | size | sha256 | validator | dvc status |

### Live Processes
| host | pid | command | started | lock | safe action |

### Blockers
| blocker | evidence | required next action |

### Queue State
| run | planned | valid outputs | pending | running | failed | mismatches |

### Do Not Run
- command and reason

### Next Command
```bash
...
```
````

## Subagent Findings By Slice

### Slice 01: 0.000-6.111%

The goal was framed correctly: no probes, complete all text activation data first. The first orchestration already showed the pattern that later dominated the run: subagents and live operations were useful, but too much concurrency caused backend disconnects and `Too many open files`. Planning and static-loader preflight were valuable, and missing model-dependent sources were caught early.

### Slice 02: 6.111-12.222%

The static extraction path was technically reasonable: smoke tests, dynamic scheduler, persistent workers, HDF5 shards, and verifier checks. The failures were operational: runbooks were read late, chunk sizing changed after launch, hidden temp shards were cleaned while workers were alive, and queue/output repair became necessary.

### Slice 03: 12.222-18.333%

Architecture looked good: processor-aware Qwen masking, backend abstraction, provenance-rich HDF5 data, and duplicate-key merge checks. Operations were weak: two merge writers targeted the same output, mixed queue plans polluted the run directory, and a fresh large extraction was launched with `--no-verify-masks`.

### Slice 04: 18.333-24.444%

The agent recovered from a wrong config-path assumption, copied the correct root `model_deployments.yaml`, worked around OpenRouter structured-output failures, and rejected unusable one-sided insider extraction. Risks were duplicate grader/generator launches, fragile remote scripts, static merge opacity, and no full-suite validation in that slice.

### Slice 05: 24.444-30.556%

Static and sycophancy artifacts were validated and fetched safely. The static 45 GB artifact used size/SHA checks and DVC pointer creation. Insider `s20` was frozen at `406/520`. DVC locks were respected, but DVC became a serialized bottleneck and remote/local state drifted.

### Slice 06: 30.556-36.667%

The restart scare showed why `/dev/shm` cannot be the source of truth. The local snapshot was authoritative. The user explicitly paused GPU work and redirected to optimization/docs, but the agent kept juggling recovery and implementation. The user's safety question should have received a risk matrix immediately.

### Slice 07: 36.667-42.778%

This was the clearest scope-overload slice. The agent mixed local code fixes, remote resume, remote cache migration, artifact validation, DVC supervision, and planning. It produced lasting safeguards, but also made command mistakes, guessed expected task lists, used broad `pkill -f`, and relaunched workers before enough remote preflight.

### Slice 08: 42.778-48.889%

Validation evidence was strong: repeated pytest runs, HDF5 hashes, task/layer/dim checks, and focused suites. But command usability lagged: bad merge command, bare `python`, invalid validator syntax, and failed full insider grading. DVC was alive, not proven complete, in this slice.

### Slice 09: 48.889-55.000%

The right state was known but not frozen. Static DVC push was still alive, activation artifacts were validated, insider `s20` was incomplete, and the local snapshot had stale `done=812` with only `406` outputs. This mismatch should have become the top handoff warning.

### Slice 10: 98.500-100.000%

The final tail showed a successful endpoint after the ignored middle: rclone/upload verifier completed with exit `0`, DVC remote sync was clean, final all-text HDF5 validated with SHA, insider JSON validated, no new probes ran, and the goal was marked complete. The remaining issue was an unfrozen dirty repo and a weak persisted SHA-validation artifact.

## Next Best Actions

1. Make a freeze commit for the completed pre-probe gate.
2. Include `.dvc` pointers, validation scripts, safe fetch scripts, dynamic runner hardening, tests, `plan.md`, and final docs.
3. Persist a final validation JSON for the all-text HDF5 that includes SHA-256.
4. Add run locks, merge locks, plan IDs, and output-based queue status before the next long GPU run.
5. Start the next phase with probe preflight only: split policy, layer selection, class imbalance handling for `insider_trading__upscale`, and I/O/memory estimate for the 58 GiB HDF5.

## Bottom Line

The project got the thing it needed most: a validated, DVC-synced all-text activation store before probe training. That is a real win.

The cost was too high because the session lacked a disciplined operations layer. Next time, the agent should spend more effort up front making state hard to corrupt, and less effort live-repairing state after corruption becomes possible.
