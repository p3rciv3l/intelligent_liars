# OSWorld Qwen3-VL run: lessons learned and future-run specification

**Run root:** `live-c6be945-resume-20260723T050624Z`  
**Scope:** checksum-verified run artifacts, completed evidence-driven postmortem, and completed cost audit. This document is an operational specification, not a proposal to mutate the historical run.

## Decision

The run is not a valid paper-compatible Qwen3-VL reproduction. Its 361 genuine evaluator results are durable and useful as failure evidence, but its 2.5239% mean score must not be interpreted as model capability: the generic Qwen agent/parser corrupted intended actions, silently converted malformed actions to `DONE`, and used a materially different history/image contract. Full-scale execution continued after Small already showed a 1/33 (3.03%) score, contrary to the frozen `scale_up_allowed=false` context. Future scaling is blocked until the paper-baseline gate and every invariant below pass.

## Evidence and provenance

The terminal receipt records 361/361 genuine evaluators, score sum 9.1111, 10 positive tasks, 6,903 remotely checksum-verified files, and final task-map SHA-256 `b78a802...dfa99` [E1]. The postmortem rehydrated only files named by that final map, checked attempt files against `checksums.json`, and verified run ledgers against resumable manifests [E2]. Its upload receipt and manifest are content-addressed [E3]. The cost audit independently rehydrated 1,710/1,710 checksum-addressed objects with zero failures [E4]. A local recheck of every downloaded checksum-addressed file used here verified 370/370 hashes; see the companion local `hash-verification.json` (working evidence, not a repository deliverable).

Evidence citations use immutable S3 keys. Source citations include frozen line ranges.

- **[E1] Terminal acceptance:** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/terminal/pre-pause-final-receipt-sha256-e3764ed034e0eb88a94c50d505d2d25a01f6d32876efddc1211be11dcfd183df.json` (`results`, `manifest`).
- **[E2] Postmortem:** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/verification/postmortem/reports/sha256-7930e0231afcd1a97f0b30a857bbf341910365f2e7c64a20e1efa420640ad1bc-POSTMORTEM.md#L3-L38`.
- **[E3] Postmortem receipt:** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/verification/postmortem/receipts/sha256-2ee98b98d0fb94294b39e7c50eb620c18e58a29802b6f2d9da288faeabad186d-postmortem-upload-receipt.json`.
- **[E4] Cost audit:** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/verification/cost-audit/cost-audit-sha256-001546c7cc4a919cf3b184d5b35e28ade9d8c85a2ca88ec9eb7ada49ac314185.json` (`rehydration`, `canonical`, `terminal_reconciliation`).

## Frozen identity of the analyzed run

- Intelligent-liars commit: `c6be945d39e89b612abdc18f33d49ae78d124ad1` (manifest recorded clean).
- OSWorld commit: `b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf`.
- Model/processor/tokenizer: `Qwen/Qwen3-VL-8B-Thinking` revision `92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b`.
- Canonical final run: `osworld-de2bd7f9c0e1f1c57375`, 361 no-GDrive tasks, max 100 steps; terminal manifest SHA-256 `de2bd7f9c0e1f1c573758486d711bfdaa789ac2a46b9206c50eb85b59bf7d771`.
- Final verified task map SHA-256: `b78a802adeb3994bfac8de7858494cc330a479980215f82e0ea01015b40dfa99`.
- Frozen action parser SHA-256: `5bf7d6919726b11d74433716a2d59b1ae3b80e64a8250867add60231c14bfdbf`.

These values identify what ran; they do not make its generic-agent contract paper-compatible. Future manifests must carry the same categories with the corrected agent bundle.

## Non-negotiable invariants

1. **Paper-baseline score gate.** Before any full run, execute a frozen, stratified 20-task gate using the paper-compatible public contract: max 15 steps (a max-50 extension reported separately), ≥99% syntactically valid tool calls, ≥98% raw-to-executed coordinate/key/text retention, zero malformed-action-to-termination conversions, ≤5% terminal model timeouts, 100% required artifact checksums, and score ≥20% with Wilson interval and domain breakdown. The target reference remains paper 33.9%; the gate is deliberately lower but must reject the observed 3% early signal [E2, lines 339-355].
2. **Malformed action fail-closed.** Missing, unknown, ill-typed, out-of-bounds, or unparsable tool calls are retryable `INVALID_ACTION`; they can never execute a default action and can never become `DONE`/`FAIL`. Only an explicit, schema-valid `terminate(status=...)` may terminate.
3. **Exact task/commit/model manifest.** Before paid work, freeze and checksum: ordered task IDs and each task config/evaluator; intelligent-liars commit and dirty state; OSWorld commit; agent class; prompt, parser, processor, tokenizer and wheel hashes; model ID and immutable revision; endpoint image/body/token limits; generation parameters; history policy; proxy mode; desktop image/region/screen geometry; retry/checkpoint policy. Runtime code must match the manifest byte-for-byte.
4. **Per-transition status and evaluator metrics.** Every `queued → running → {retrying|failed|genuine_evaluator_complete}` transition is immutable and checksum-addressed and includes task ID, attempt, timestamps, reason/error class, worker/resource IDs, manifest hash, and artifact/checkpoint references. Every metrics snapshot includes evaluator count, score sum/mean/positive count, per-domain values, action/termination distribution, timeout rate, GPU utilization, resource state, and reconciled spend/projection. A mutable `latest` pointer may reference, but never replace, immutable events.
5. **Checksum durability.** An attempt is not complete until evaluator, trajectory, raw response, runtime log, initial and transition screenshots, recording, attempt manifest, and checksum file exist remotely; remote bytes and metadata hash must re-verify. Content-addressed writes use `If-None-Match: *`; collision is success only when remote bytes hash identically.
6. **Authoritative billing reconciliation.** Runtime estimates are safety controls, not final spend. Reconcile Vast from authoritative charge history across every lineage-owned instance and AWS from billing APIs when available, otherwise from an append-only lifecycle ledger with explicit one-sided uncertainty. Include compute, stop lag, disks/EBS, IP, S3 storage/requests, data transfer, and retained monthly burn. Never call a partial estimate “actual.”
7. **Resource ownership.** Every resource has `owner`, `run_root`, `managed_by`, `created_at`, `billing_scope`, `cleanup_policy`, and `preexisting` fields. Only lineage-owned resources may be stopped/terminated. Preexisting resources are excluded from run cost and preserved unless separately authorized.
8. **Pause/cleanup order.** Stop admission; quiesce workers; persist final transitions/ledgers; upload and remote-verify all required artifacts; produce final task map and acceptance receipt; reconcile owned resources; stop paid GPU; terminate ephemeral clients; stop controller; verify zero running owned resources; then optionally delete only explicitly disposable resources. Never destroy evidence-bearing disks before retrieval verification.
9. **No scale-up when early score diverges.** Scale-up is mechanically disabled unless the gate receipt is present and signed off. If cumulative score or retention metrics fall below the gate at any checkpoint, stop admission and pause. Budget headroom, throughput, or schedule cannot override quality divergence.

## Failure register

Each entry is normative: its guardrail, automated test, and runbook step are required for the next run.

### F-01 — Wrong Qwen agent/parser contract

- **Symptom:** Intended coordinate/key/text actions became bare clicks, empty typing, zero scroll, `(0,0)` drags, or termination.
- **Evidence:** The run used generic `QwenAgent` with nested XML while the paper-era public integration used `Qwen3VLAgent` JSON `<tool_call>`. Of 2,188 predictions, 1,784 coordinate tags were malformed, 1,668 became bare clicks, and only 47 retained coordinates [E2, lines 78-124]. The frozen regex accepts only `<parameter=name>` [S1]; fallback action builders execute defaults when required parameters are absent [S2].
- **Root cause:** Agent prompt/tool schema and parser were selected independently rather than as one versioned contract; contract compatibility was never tested on real model emissions.
- **Impact:** Direct event-level action corruption across all domains; the score is not a valid capability estimate.
- **Guardrail:** Pin one agent contract bundle (class + prompt + schema + parser + normalizer + action mapper); Qwen3 runs use the Qwen3 JSON tool-call agent, not a permissive XML repair.
- **Automated preflight/test:** Golden raw responses for every action type plus malformed/unknown/missing fields; assert exact normalized call and executed command, ≥98% retention on a live 100-transition sample, and zero default coordinates/text/scroll.
- **Future runbook:** Freeze bundle hashes; run unit corpus; run one-task trajectory diff; manually compare raw/normalized/executed actions; only then start the stratified gate.

### F-02 — Silent `DONE` fallback

- **Symptom:** Tasks declared completion after an invalid call, including malformed key actions.
- **Evidence:** 315 tasks ended `DONE` with zero score; 30 stopped after one action and 76 within three; 214 unrecognized-action predictions and 98 malformed key calls became `DONE` [E2, lines 126-138]. Frozen source appends `DONE` whenever parsing yields no action [S2].
- **Root cause:** Empty parse was conflated with successful completion and broad natural-language matching could also select `FAIL`.
- **Impact:** Premature terminal states concealed parser failures and prevented retries.
- **Guardrail:** Fail closed as `INVALID_ACTION`; termination requires an explicit schema-valid terminate call and is separately logged.
- **Automated preflight/test:** Property/fuzz tests prove all malformed inputs produce `INVALID_ACTION`; a static check forbids parser branches that synthesize `DONE`/`FAIL`; end-to-end injection confirms retry without environment action.
- **Future runbook:** Monitor invalid-action rate and termination provenance after each gate task; halt on any implicit termination.

### F-03 — History/image mismatch

- **Symptom:** At step 5 all prior screenshots disappeared and only the current image remained; prompt tokens collapsed.
- **Evidence:** Paper-era public behavior retains four prior screenshots plus current; frozen `history_n=20,image_max=4,fold_size=4` collapsed history at step 5. Isolated KL was 0.913 and mean input length fell 11,535→3,390 tokens [E2, lines 140-192]. Frozen history construction is visible at [S3].
- **Root cause:** History length and endpoint image cap were configured independently; no per-step rendered-message equivalence test existed.
- **Impact:** Demonstrated distribution shift and lost visual state, compounding action errors.
- **Guardrail:** Use paper-era history 4 with capacity for five images, or explicitly validate history 3 with four-image cap; never discontinuously collapse all history.
- **Automated preflight/test:** Snapshot rendered roles, image count/order, token hashes, and prompt length for steps 1–10 against golden fixtures; reject discontinuities or endpoint-cap violations.
- **Future runbook:** Record rendered-request digest and image lineage at every transition; inspect step 5 before gate continuation.

### F-04 — Proxy mismatch

- **Symptom:** Some Chrome tasks showed blank/black/white pages; Chrome scored 2/46.
- **Evidence:** Frozen manifest set `enable_proxy=false` while paper-era runner constructs the environment with proxy enabled [S4]. Five Chrome tasks described blank views, though cross-domain parser failures prove proxy was not the primary cause [E2, lines 246-257].
- **Root cause:** Proxy mode was treated as incidental infrastructure rather than benchmark contract.
- **Impact:** Probable domain-limited web failures and reduced comparability; magnitude is unmeasured.
- **Guardrail:** Freeze proxy configuration, health endpoint, geography, and credential mode in the manifest; run matched proxy-on/off Chrome controls.
- **Automated preflight/test:** Page-health suite for all proxy-dependent domains validates DNS/TLS/status/content/screenshot; matched controls must meet a declared success threshold.
- **Future runbook:** Verify proxy before model requests, store health receipt, and quarantine network-invalid tasks rather than scoring them as agent failures.

### F-05 — Request-body limit

- **Symptom:** Representative multimodal requests exceeded the prior 8 MiB body limit and would return HTTP 413.
- **Evidence:** Endpoint contract records a 8,915,511-byte representative request, configured limit 35,959,180, status 200, and exactly one byte over the new limit returning 413 [E5].
- **Root cause:** Server body limit was not derived from maximum rendered history/image payload plus encoding overhead.
- **Impact:** Earlier endpoints could reject valid requests before inference, creating retry storms or zero-action attempts.
- **Guardrail:** Compute a deterministic upper bound from image cap, PNG/base64 expansion, prompt and output envelope; configure headroom and reject locally with a typed error.
- **Automated preflight/test:** Send boundary fixtures: representative, declared maximum, and maximum+1; expect 200/200/413 and verify proxy/load-balancer/server limits agree.
- **Future runbook:** Persist body-limit receipt before warmup and monitor request-byte percentiles; pause at >90% limit.

### F-06 — Mutable environment and in-run patching

- **Symptom:** Multiple controller hashes (`run-small` variants, rescue, full remediation) and runtime policy changes were introduced during the paid run.
- **Evidence:** Immutable receipts identify successive controller hashes for signal and retry remediation [E6, E7], while the original manifest declared commit `c6be945...`, dirty false, model revision `92f3c4...`, and specific parser/history settings [S5].
- **Root cause:** Environment reproducibility covered repository/model pins but not the complete deployed controller, base image, dependency lock, endpoint and runbook state as one immutable release.
- **Impact:** Attempts were not all produced by one execution implementation; audit and causal interpretation became harder.
- **Guardrail:** Build a signed run release containing source commit, wheel/container digest, lockfile, controller, endpoint image, OSWorld commit and manifest. Any code change creates a new run ID and gate.
- **Automated preflight/test:** Remote attestation compares all file/image/dependency hashes to release manifest; periodic drift detector fails admission on mismatch.
- **Future runbook:** Never hot-patch workers. Pause, preserve, issue a new release/run root, rerun gate, and document lineage.

### F-07 — IAM and SSH-key bootstrap

- **Symptom:** Billing/CloudTrail/CloudWatch reads were denied, and controller access required later run-scoped public-key bootstrap.
- **Evidence:** Cost audit records `AccessDenied` for Cost Explorer, CloudTrail and CloudWatch [E4]. Managed controller receipt records `run_scoped_public_key_bootstrap=true`, restricted SSH, and no credential/private key in user data [E8].
- **Root cause:** Required read-only audit permissions and access bootstrap were not proven before launch; access and billing roles were conflated with runtime permissions.
- **Impact:** Delayed control-plane startup, incomplete authoritative AWS billing, and increased manual recovery risk.
- **Guardrail:** Separate least-privilege runtime, lifecycle, and billing-auditor roles; use SSM where possible; pre-create/test ephemeral run key, never inject private keys or secrets into user data.
- **Automated preflight/test:** Dry-run IAM matrix for required API calls; SSH/SSM connect-and-command test; verify key fingerprint, CIDR/SG restriction, rotation and cleanup policy.
- **Future runbook:** Block paid launch until IAM/access receipt passes; record denied optional APIs explicitly and predeclare fallback accounting.

### F-08 — Watchdog detachment/liveness

- **Symptom:** Safety depended on a watchdog whose survival could be coupled to the launching session until a detached implementation and remote heartbeat stream were verified.
- **Evidence:** Later receipt explicitly verifies `detached_watchdog_code`; runtime extensions changed maximum paid runtime while keeping resource IDs [E9]. Checksum-addressed heartbeats exist through shutdown.
- **Root cause:** Process detachment, independent credentials, restart policy and stale-heartbeat response were not initially an acceptance gate.
- **Impact:** A lost shell/controller could have disabled budget enforcement and teardown.
- **Guardrail:** Run watchdog under an independent service manager/control plane with restart-on-failure, lease/heartbeat, immutable config, and authority over only owned resources.
- **Automated preflight/test:** Kill launcher shell, controller process and network path separately; watchdog must remain alive or independently pause resources within SLO. Alert on two missed heartbeats.
- **Future runbook:** Verify PID/service identity and remote heartbeat before paid admission; never extend runtime without a new signed budget/config receipt.

### F-09 — Vast offer/capacity misreporting

- **Symptom:** Initial capacity logic treated aggregate multi-GPU performance as independent serving replicas and risked misreading per-GPU versus total VRAM and host price.
- **Evidence:** Corrected audit proves `dph_total` is total host hourly price, `gpu_ram` is per GPU, `gpu_total_ram=num_gpus*gpu_ram`, and one machine/offer is rentable once [E10]. It distinguishes independent replicas from tensor-parallel hosts and labels aggregate scaling optimistic [E11].
- **Root cause:** Marketplace field semantics and topology constraints were assumed rather than validated against API definitions and raw snapshots.
- **Impact:** Offers, throughput, host counts and projected cost were misreported; scaling feasibility could be false.
- **Guardrail:** Preserve raw offer JSON; normalize units; enforce one offer per machine; model TP as one serving replica unless measured concurrency proves otherwise; use `dph_total` as host cost.
- **Automated preflight/test:** Schema/property tests for empirical field equalities, duplicate machine IDs, availability and cost formulas; benchmark selected topology before purchase.
- **Future runbook:** Produce checksum-verified capacity receipt with raw snapshot, formulas, frontier and rehydration before launch; reject stale offers.

### F-10 — Spend accounting and billing reconciliation

- **Symptom:** Terminal pre-pause estimate reported $69.7527 combined, but authoritative lineage reconciliation was $78.6483 best estimate with AWS uncertainty up to $14.9079 and Vast authoritative charges of $69.189.
- **Evidence:** Terminal estimate included only one current Vast elapsed-rate estimate and then-visible AWS records [E1]. Audit found 510 transient clients, earlier Vast lineage instances, stop lag/disk/bandwidth, and purged EC2 records; Vast `GET /charges` was authoritative [E4].
- **Root cause:** Safety projection, partial live inventory and actual billing were presented in overlapping concepts without a canonical lineage ledger.
- **Impact:** Spend understated by about $8.90 at the best reconciled point; retained EBS/S3/Vast disks continued monthly burn.
- **Guardrail:** Append every resource lifecycle event before mutation; assign ownership and lineage; maintain separate `estimated`, `accrued_best`, `accrued_upper`, `authoritative`, and `ongoing_monthly` fields.
- **Automated preflight/test:** Reconcile provider inventory, lifecycle ledger and billing/charges by resource ID; arithmetic/schema tests reject missing lineage, overlaps, negative deltas, or unlabeled uncertainty.
- **Future runbook:** Use conservative upper bound for admission; after stop convergence, fetch authoritative charges, publish reconciliation and retained-cost inventory.

### F-11 — Retry, staging and checkpoint defects

- **Symptom:** Repeated attempts collided in staging, reused attempt numbers, omitted failed attempts from ledgers, or resumed tasks that already had genuine evaluators.
- **Evidence:** Preserved failures include 18 staging `FileExistsError`s, 30 missing-screenshot checkpoints, 17 missing-video checkpoints and 70 retry transitions [E2, lines 230-244]. Rescue receipt lists content-address/move-before-retry, append-failed-attempt, increment attempt number, wait for initial screenshot, and resume-only-missing-evaluator fixes [E7].
- **Root cause:** Attempt identity, staging lifecycle, checkpoint completeness and terminal evaluator selection were not transactional/idempotent.
- **Impact:** Wasted clients/GPU time, tail amplification, collision risk, and possible duplicate or lost attempt history.
- **Guardrail:** Allocate immutable attempt ID before execution; stage in unique temp dir; atomically seal/content-address; append ledger before retry; resume strictly from verified evaluator/task map.
- **Automated preflight/test:** Fault-inject at every write/move/upload boundary and restart repeatedly; assert no overwrite, no duplicate attempt number, complete ledger, deterministic resume set and checksum-valid final selection.
- **Future runbook:** Classify errors, preserve failed bundle first, then retry with new ID; quarantine repeated infrastructure failures after bounded attempts.

### F-12 — Signal registration from worker threads

- **Symptom:** Official AWS manager attempted `signal.signal` outside the main thread and failed.
- **Evidence:** Remediation receipt names the cause and preserves real signal behavior only on main thread [E6]. Frozen controller wrapper checks `threading.main_thread()` before registration [S6].
- **Root cause:** A process-oriented library was embedded in threaded workers without validating signal assumptions.
- **Impact:** Small startup failures and emergency cleanup risk.
- **Guardrail:** Lifecycle/signal handling stays in main process; workers use queues/events and cannot register process signals.
- **Automated preflight/test:** Start environment in production concurrency topology; send SIGINT/SIGTERM; assert graceful quiesce, artifact flush and owned-resource cleanup.
- **Future runbook:** Run signal test before paid endpoint admission; never monkey-patch signal behavior without a new release and gate.

### F-13 — Recording and file-retrieval failures

- **Symptom:** Completed attempts lacked locally retrieved screenshots/videos or failed checkpoint retrieval despite remote task execution.
- **Evidence:** 30 missing-screenshot and 17 missing-video events occurred; postmortem found them recovered in final checksum-verified bundles and did not attribute direct final score loss [E2, lines 230-244]. Manifest required initial screenshot and recording [S5].
- **Root cause:** Recording finalization, remote availability, local retrieval and attempt completion were not one verified transaction; eventual consistency/readiness was treated as absence.
- **Impact:** Retries and slow tail; risk of destroying resources before forensic files were durable.
- **Guardrail:** Completion waits for recording finalization, stable size, download, checksum, remote upload and re-download verification; retrieval errors are infrastructure failures, never evaluator zeros.
- **Automated preflight/test:** Delay/truncate/interrupt recording and screenshot endpoints; assert bounded polling, resumable retrieval, checksum detection and no completion receipt until all required files pass.
- **Future runbook:** Flush recorder first, retrieve with backoff, seal manifest, remote-verify, then release desktop/client.

### F-14 — Underutilized GPUs

- **Symptom:** Eight A100s held ~20.8 GiB each but fleet average SM utilization was 19.23%; GPUs 3, 4 and 6 sampled 0%.
- **Evidence:** Sustained-production receipt reports per-GPU memory/utilization, one inference lock per replica, no dynamic batching, 0.5862 action cycles/s and no throughput gain from same-replica concurrent requests [E12].
- **Root cause:** End-to-end task orchestration, AWS desktop latency and one-request endpoint locks starved inference; GPU count was optimized independently of request supply.
- **Impact:** High Vast spend for idle capacity and no six-minute feasibility.
- **Guardrail:** Size fleet from measured end-to-end arrival/service rates; require queue depth and per-replica utilization; support measured batching/concurrency or scale down.
- **Automated preflight/test:** Load test complete action cycles, not isolated tokens; require declared utilization/throughput floor per replica and verify queue-to-GPU affinity.
- **Future runbook:** Ramp 1→2→N replicas only when prior stage is saturated and score-valid; scale down idle replicas after two windows.

### F-15 — Slow tail and retry amplification

- **Symptom:** After fast early progress, a small set of repeated failures kept workers/resources alive for hours; queue priority later changed to run zero/fewer-attempt tasks first.
- **Evidence:** Full run started 08:32 and final transition completed 21:56. Priority-remediation receipt explicitly deprioritized repeatedly failing tasks without restarting the epoch [E13]; final run contained 37 terminal timeouts and 143 preserved infrastructure-failure attempts [E2, lines 194-244].
- **Root cause:** Static worker partitions, long outer timeouts, retries in the critical path, and no early straggler quarantine/speculative policy.
- **Impact:** Disproportionate runtime and spend with low marginal score recovery.
- **Guardrail:** Global durable queue, bounded per-class retries, attempt deadlines, straggler quarantine, work stealing and tail budget; preserve evaluator-complete tasks immediately.
- **Automated preflight/test:** Synthetic mix of fast, timeout and checkpoint-failing tasks; assert healthy tasks are never blocked, p95/p99 SLOs, bounded retries and deterministic terminal remainder.
- **Future runbook:** At each epoch, run unseen/fewer-attempt tasks first; pause and review when tail cost exceeds marginal-value threshold.

### F-16 — Premature scale-up after Small diverged

- **Symptom:** Full 361-task controller launched after Small had 33 genuine evaluators with score 1.0/33 (3.03%) and 32 zero-score `DONE`s.
- **Evidence:** Frozen run context says `scale_up_allowed=false` [E14]. Small metrics at 08:15 recorded the divergence; full receipt was created at 08:31 and full execution began 08:32 [E15, E16]. Final score remained 2.52% [E1].
- **Root cause:** Scale authorization was advisory and throughput/budget readiness was allowed to override missing quality acceptance.
- **Impact:** Most of the $78.65 reconciled spend produced invalid-contract evidence already predicted by Small.
- **Guardrail:** Admission controller requires a checksum-addressed gate receipt whose score/retention/timeout predicates pass; no manual override within the same run root.
- **Automated preflight/test:** Attempt full launch with absent, stale, wrong-manifest and failing gate receipts; all must be denied. Live sequential probability/quality checks pause on divergence.
- **Future runbook:** Stop after Small, inspect failures, correct under a new run ID, rerun gate; scale only on explicit pass.

### F-17 — Teardown and retained-cost control

- **Symptom:** Teardown required a multi-stage pause; controller EBS, security groups and both Vast disks were intentionally retained, continuing monthly cost.
- **Evidence:** Pre-pause receipt planned Vast stop and evidence preservation [E1]. Pause receipt verifies zero running AWS/Vast, controller stopped, clients terminated, resources not destroyed, and retained volume/security groups [E17]. Cost audit reports $2.40/month EBS, ~$5.40/month Vast retained disk and $0.699/month S3 [E4].
- **Root cause:** Pause, preserve, destroy and cost-finalization semantics were not defined as one ownership-aware state machine from launch.
- **Impact:** Risk of premature evidence destruction on one side and unnoticed recurring spend on the other.
- **Guardrail:** Explicit resource terminal policy (`terminate`, `stop-retain-until`, `preserve-preexisting`) with owner, deadline and approver; verify stop convergence and recurring-cost inventory.
- **Automated preflight/test:** Teardown simulation with preexisting and managed resources; assert order, evidence gate, zero running owned resources, no mutation of preexisting resources, and retained-cost alerts.
- **Future runbook:** Follow invariant 8 exactly; publish pause receipt, billing reconciliation and dated deletion decision; re-audit until retained resources are disposed or renewed.

## Required future-run lifecycle

1. **Release:** create immutable release/manifest and ownership ledger; verify IAM, access, body limit, proxy, endpoint fingerprint, watchdog detachment and authoritative accounting paths.
2. **Contract tests:** parser fuzz/golden tests, history rendering snapshots, action execution dry-run, signal/teardown fault tests, and artifact transaction tests.
3. **One-task canary:** prove genuine evaluator, action retention, complete remotely rehydrated bundle, transition metrics and cost/resource status.
4. **20-task gate:** run stratified paper-compatible gate and apply all invariant thresholds. Any divergence pauses admission.
5. **Scale ramp:** add replicas only after valid score and measured saturation; verify each stage's throughput, utilization and spend projection.
6. **Production:** immutable queue and per-transition events; prioritize unseen tasks; quarantine stragglers; continuously enforce quality, budget and liveness.
7. **Finalize:** stop admission, reconcile every task and artifact, generate final verified map, pause in ownership-safe order, reconcile authoritative billing, and track retained cost to deletion.

## Citation index

- **[E5]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/endpoint/contract.json`.
- **[E6]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/verification/controller-remediation-sha256-8afa18cee645d016d19a6125a8534b290afa03b71cb25438617a527ac09b9431.json`.
- **[E7]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/verification/retry-remediation-sha256-bf99b786ebbe29ae395f5afc41aa5145a8fae995c3e414e9eff5b8bec687f886.json`.
- **[E8]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/resources/aws-controller-launch-sha256-601208fad237db197b7e446b52d1bf6d5a105c69a32b2589599c62bd5a66d076.json`.
- **[E9]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/verification/watchdog-code-remote-verification-sha256-f9602df79f91de8cd2c15d697874fadd2e66cefee5a6d5fd15a39ab15dbbe6c0.json`; `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/verification/watchdog-runtime-extension-sha256-9e48b2fb4e9269107b2e8f4ddb1c6bd82b1db7be58e90ce2f0fca52263cfdde2.json`.
- **[E10]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/verification/corrected-capacity-audit-receipt-sha256-1ed6ef2fea04660c36f86e25736aa1fbfc246098612c666a0e7dabb297e49981.json`.
- **[E11]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/verification/corrected-capacity-audit-formulas-sha256-b3727641d09374c0bab8e9e46b722f57f74a055c7c98461369547cfdc2c194e8.json#L50-L96`.
- **[E12]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/verification/sustained-throughput-correction-sha256-a7a4839cd645611e86a3366824fae450229fb82f1f5e11a18b9fd548eab712c1.json`.
- **[E13]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/verification/full-queue-remediation-sha256-bdadb753871f3d791869935593e412b9198e1126857766d7a331069d40dd444f.json`.
- **[E14]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/validation/run-context.json#L21-L35`.
- **[E15]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/status/metrics/000029-sha256-678be7777c1cf593d0614fb87353f20da7927a43b8fc2c0286d5edda374974a4.json` (canonical total 39; 33 evaluators; score sum 1.0; timestamp 08:15:36; immutable key is discoverable from the checksum-addressed status index).
- **[E16]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/verification/full-controller-receipt-sha256-1b3ece72d0b7e410e7fdc085c349d5fbf772dd2419775bc381b90c3beb28b875.json`; first full event `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/status/events/000001-queued-sha256-d72c724c78c5bd9fd09eea384769c547d6b4b872f22c778611fa1fdb82b9c4b2.json`.
- **[E17]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/terminal/pause-only-resource-state-sha256-e0c8fb5f010dba07f11733e7b35a1e5ba40ae65d5b1ab61c2ad4f649335983ac.json`.
- **[S1]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/verification/postmortem/sources/sha256-de2b16e9cb08aaebb0734f01285aff74844938bbd3b1a4c891cee2e8746468d6-qwen-parser.py#L7-L30`.
- **[S2]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/verification/postmortem/sources/sha256-5bf7d6919726b11d74433716a2d59b1ae3b80e64a8250867add60231c14bfdbf-qwen-actions.py#L185-L326`.
- **[S3]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/verification/postmortem/sources/sha256-c9bb34d3ad822168c66133cd97c607d4645b7eff08072e1e45c49dbbad8491b4-our-qwen3vl-agent.py#L241-L334`.
- **[S4]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/verification/postmortem/sources/sha256-2b21aef6afbbbd1ab5dcbb8c1d948b06800098d27c495c59ac7d73c551a88cfa-paper-era-run_multienv_qwen3vl.py#L180-L195`.
- **[S5]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/manifests/one_task_50.json#L1-L105`.
- **[S6]** `s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/runs/live-c6be945-resume-20260723T050624Z/controller/run-small-rescue-sha256-a6df72c0fa60daa1f33807b9b36d92b4302424942129079d92ef9e228067179e.py#L108-L135`.
