# Truth-editing optimization: zero-scored-trials recovery handoff

**Status:** urgent execution handoff

**Repository:** `/Users/student/Desktop/ai/intelligent_liars`

**Prepared:** 2026-08-30

**Timebox for the receiving agent:** three hours of aggressive, parallel diagnosis and implementation

**Primary objective:** make the next large truth-editing optimization run reliably produce real objective scores instead of operational failures.

## Start here

You are taking over a failed production optimization run. Do not write another plan-only handoff. Diagnose, implement, test, push ready fixes to `main`, and leave the repository and cloud state ready for the next run.

The last run completed 224 trial records but produced **zero successful scored trials**:

- 189 operational failures
- 35 scientifically infeasible trials
- 0 successful scored trials
- broad coverage incomplete
- final selection correctly refused to run

The dominant issue is not that OpenRouter was entirely unavailable. The production judge completed more than one thousand individual calls. The system instead amplified a relatively small number of unresolved judge records into whole-trial failures, and a global budget circuit allowed one ambiguous request to poison unrelated work in the same batch.

Your target is:

> Every operationally valid trial receives a complete, durable score. Genuine scientific infeasibility remains a legitimate scientific outcome and must not be relabeled as success.

Do not weaken scientific gates, invent judge outputs, silently accept malformed JSON, or count operational failures as successful coverage merely to make the dashboard green.

## Operator mandate

The user explicitly authorizes:

- extensive agent fanout during this three-hour recovery session;
- cheap OpenRouter calls using the existing key;
- a small Vast.ai instance, such as one RTX GPU, if GPU-backed validation is genuinely useful;
- GitHub access and pushing reviewed, tested production fixes directly to `main`;
- AWS access through the existing automatically renewed role credentials;
- rapid implementation and adversarial testing rather than further permission-seeking.

The user’s intended launch message is:

> Go fucking crazy over the next three hours. You have GPT-5.6 Sol Ultra and should use as much economically useful fanout as possible. Fix the OpenRouter/judge problem and every related scoring failure. Cheap API tests are authorized, and a one-GPU Vast instance is authorized if needed. The next big run must not collapse into operational failures. Have fun, but solve the problem.

This authorization does **not** authorize exposing secrets, force-pushing, deleting audit records, accessing sealed audit data, uploading model weights, or concealing scientific failures.

## Suggested skills

Use these skills in this order where available:

1. `diagnosing-bugs` — reproduce and classify the real failure chain before changing semantics.
2. `tdd` — lock each failure class into a failing test, then fix it.
3. `codebase-design` — design a durable per-record judge-completion seam rather than another local patch.
4. `implement` — execute the bounded implementation.
5. `vast-gpu-experiments` — only if remote GPU validation is needed; account for instance state, cost, evidence, and cleanup.
6. `code-review` — adversarial review of billing identity, cache identity, replay, and scientific integrity before pushing.

Parallelize independent work. Useful agent lanes are listed later in this document.

## Mandatory reading order

Read these before editing. They contain the frozen contracts and experiment boundaries; do not re-invent them from chat history.

1. `AGENTS.md`, if present in the checkout. It may be local and untracked; do not commit it blindly.
2. `docs/heretic_modification_pipeline.md`
3. `docs/heretic_optuna_port.md`
4. `docs/llm_judge_harness.md`
5. `docs/simple_truth_direction_experiment_protocol_20260825.md`
6. `docs/simple_truth_direction_run_plan_20260826.md`
7. `README.md`
8. `docs/truth_editing_live_run_history.md` — append-only operational history.
9. `docs/adr/0001-freeze-response-healing-in-json-judge.md`
10. `docs/adr/0004-use-batched-optuna-suggestions.md`
11. `docs/adr/0011-run-optuna-without-a-separate-fixed-screen.md`
12. `docs/adr/0015-use-240-human-labels-for-live-judge-calibration.md`

Also inspect the implementation and tests named under **Primary code seams** below.

## Frozen judge contract

Do not change these settings inside the existing study identity:

- model: `z-ai/glm-5.3-flash`
- provider: official `z-ai/fp8`
- provider fallback: disabled
- reasoning effort: high
- temperature: `0`
- top-p: `1`
- output-token ceiling: `2048`
- JSON response format: required
- response healing: enabled
- local schema validation: fail closed

The judge handles semantic ambiguity. Deterministic eligibility and truth checks own hard facts. If evidence proves that the frozen contract itself must change, create a new versioned judge and a new study identity. Never silently mix changed judge behavior into the old study.

Response healing already exists. It performs one explicit JSON-only correction attempt followed by deterministic normalization. The task is not merely to “turn healing on.” Verify that it works against the actual OpenRouter response shapes and schema failures, and make incomplete work resumable without hiding malformed output.

## Repository and delivery state

At handoff creation:

- local checkout branch: `codex/simple-transform-experiments`
- local `HEAD`: `e55e7a0fdafc56ca41674d056d80f9f6a57e661a`
- remote `origin/main`: the same commit
- tracked changes: none before this handoff was added
- the worktree contains hundreds of unrelated untracked files; preserve every one of them

Relevant recent production commits already on `main`:

- `c8e4ba5` — harden judge response healing and adaptive tier validation
- `98f3ab4` — scale judge timeout with bundle size
- `cfec374` — remove universal judge timeout cap
- `2eed972` — enable safe restart after minimum-guarantee abort
- `9a194f6` — carry forward host spend when rearming
- `82dff40` — make adaptive rearm restart-safe
- `47417bd` — allow compatible predecessor judge-cache receipts
- `e55e7a0` — propagate adaptive tier boundaries through lineage validation

Do not assume a previous patch solved the system just because its unit tests passed. Reproduce the production failure topology using the stored receipts.

Before committing:

1. inspect the intended diff;
2. preserve unrelated and untracked work;
3. run focused and broader relevant tests;
4. exclude secrets, `.env`, `.local`, temporary output, receipts containing sensitive prompts, and model weights;
5. push the reviewed commit directly to `main` without force;
6. verify remote `main` resolves to the expected commit;
7. for substantial changes, verify from a fresh clone.

## Durable run evidence

### Vast instance

An existing Vast instance was still running and idle at the last inspection:

- instance ID: `49087336`
- hardware: 8 × RTX 3090, 24 GB each
- price: approximately `$1.26/hour`
- image: PyTorch 2.5.1, CUDA 12.4, cuDNN 9 development image
- GPU state: idle, approximately 0% utilization
- controller, workers, and monitor: exited

Reconfirm this state immediately with the Vast CLI. Do not copy a web/Jupyter token into logs or this repository. Resolve connection information dynamically, for example with the Vast CLI’s SSH URL command.

This idle eight-GPU rental is actively billable. Either reuse it briefly for relevant diagnostics or shut it down before renting a one-GPU replacement. Do not leave both idling. The user has authorized a small replacement GPU if useful, not wasteful duplicate capacity.

### Remote output locations

The failed run’s durable root is:

`/workspace/outputs`

Important paths:

- controller log: `/workspace/controller-rearm-2026-08-30T041315Z.log`
- adaptive checkpoint: `/workspace/outputs/study/adaptive-run-checkpoint.json`
- adaptive receipt: `/workspace/outputs/study/adaptive-run-receipt.json`
- frozen study report: `/workspace/outputs/study/frozen/study-report.json`
- live history: `/workspace/outputs/monitoring/live-run-history.jsonl`
- fleet stop receipt: `/workspace/outputs/fleet-stop-receipt.json`
- production judge budget state: `/workspace/outputs/providers/production-judge-budget/controller-capacity-snapshot.json`

Local receipts and checkpoints are authoritative. W&B is observational only.

### Final run state

The last adaptive checkpoint reported:

| Field | Value |
|---|---:|
| Completed trials | 224 |
| Authorized through | 224 |
| Successful scored trials | 0 |
| Operational failures | 189 |
| Scientifically infeasible | 35 |
| Broad coverage complete | false |
| Phase | `finalization_reserved` |
| Stop reason | `evaluation_budget_reserve_reached` |
| Advisory trial count | 280 |
| Accounted infrastructure cost | about `$32.16` |
| Accounted evaluation cost | about `$3.68` |

Finalization correctly failed with:

`FinalistCheckpointError: study report is not selection-ready`

No finalist weights or unsafe exports were produced. The stopped phase is not directly compatible with the existing minimum-guarantee rearm path. Do not mutate the old journal to pretend it is resumable. Design an explicit, versioned rescore/restart path that preserves lineage.

### W&B

- project: `intelligent-liars`
- run ID: `2c8c9039985846a3b1f979422604756a`

Reconnect to this run only when continuing the same operational history. If a new judge contract or study identity is required, make that boundary explicit. W&B failure must never alter optimization behavior. Never upload API keys, sensitive prompts, or model weights.

## The important forensic fact

The production judge was not simply dead.

The judge budget ledger showed approximately:

- 1,036 attempted calls
- 1,022 completed calls
- 14 ambiguous failures
- actual judge cost around `$0.23`
- reserved-or-spent around `$0.58`
- no pending calls at final inspection

The judge cache contained approximately:

- 965 successful cached JSON results
- 124 failure receipts
- 3 terminal-failure aliases

Yet 189 of 224 trials were marked operational failures. This is failure amplification: one unresolved semantic item, schema failure, empty response, or globally opened circuit can discard a whole trial even when most of that trial’s judge work succeeded.

### Trial-level operational failure classes

The last forensic pass mapped the 189 operational trial failures approximately as follows:

- schema validation errors across old and current adapter hashes: the largest class
- `ProductionJudgeBudgetCircuitOpen`: a large batch-wide cascade class
- `JudgeContractError`: repeated across multiple trials
- empty response failures
- JSON decoding failures
- a smaller group of legacy receipt hashes not found in the current failure tree

There were 86 unique mapped failure receipt hashes reused across the 189 failed trials. Several individual failure receipts were reused by many trials. Preserve this sharing in the forensic model; it explains why trial counts greatly exceed distinct request failures.

Do not delete old receipts. Missing legacy hashes should be traced through checkpoint generations and staging directories rather than fabricated or silently discarded.

## Primary code seams

### Whole-trial failure amplification

File: `src/intelligent_liars/truth_editing_evaluator.py`

`RecipeEvaluator.evaluate` iterates through unresolved semantic records and calls the judge. A broad exception path returns a whole-trial `TrialAssessment("operational_failure", ...)`. `PaidJudgeCircuitOpen` can escape to the fleet layer, which also marks the whole trial operational.

The desired design is durable, per-record completion:

- retain every successful judge unit;
- persist exactly which units remain unresolved;
- retry only the missing units;
- aggregate a trial score only when all required units have a valid terminal state;
- never rebill a request whose outcome is already known;
- never lose N-1 successful units because the Nth unit failed;
- keep schema-invalid output fail closed.

### Response healing and terminal failure caching

File: `src/intelligent_liars/truth_editing_live_judge.py`

Inspect:

- explicit healing request construction;
- extraction of OpenRouter response content;
- normalization into the strict schema;
- `FileJudgeCache.terminal_failure`;
- `_infer_terminal_failure`;
- adapter-version compatibility;
- how failures retain enough safe diagnostic evidence to reproduce the mismatch.

Global circuit-open failures are transport state, not proof that a semantic request is permanently invalid. They must not become terminal semantic aliases.

### Budget circuit cascade

File: `src/intelligent_liars/truth_editing_production_judge_budget.py`

One ambiguous transport event currently opens a global circuit. The controller only acknowledges or reconciles it after the whole batch finishes, so later distinct requests in that batch fail immediately even though they were never sent.

The fixed behavior should pause or queue transport safely, reconcile the ambiguous request, and then continue independent missing work. Preserve append-only billing semantics and exact request identity. Never create duplicate paid calls merely to improve throughput.

Also inspect `scripts/run_truth_editing_cuda_fleet_controller.py`, especially the post-batch acknowledgement/reconciliation path.

### Replay and Optuna learning

File: `src/intelligent_liars/truth_editing_study.py`

Operational failures are already excluded from Optuna learning and queued FIFO with their exact original parameters. Preserve that. A replay counts as successful coverage only after it receives a complete score. Repeated deterministic failure receipts must not create an infinite replay loop; the underlying record-level cause must be repaired or surfaced explicitly.

### Finalization

The finalization refusal is downstream and correct. Do not patch it to accept an unscored study. Once a versioned rescore/restart produces selection-ready evidence, finalization should work through its normal gates.

## Three-hour execution plan

Use broad agent fanout, but assign non-overlapping ownership and keep one coordinator responsible for integrating the shared checkout.

### Parallel lane A — receipt forensics

- Build a deterministic classifier over all success caches, failure receipts, trial receipts, journal entries, and checkpoint generations.
- Produce counts by exact request identity, failure class, adapter version, bundle size, trial reuse, and whether the request was ever successfully answered later.
- Locate the legacy hashes missing from the current failure tree.
- Do not expose full prompts in ordinary logs.

### Parallel lane B — OpenRouter response reproducer

- Reconstruct representative stored/mock responses for every observed failure class.
- Exercise the current parser and response healer offline.
- Make a few cheap live requests through the **frozen** model/provider route.
- Capture status, headers needed for diagnosis, response shape, finish reason, latency, and validated JSON without printing credentials or sensitive prompts.
- Determine whether failures originate in content extraction, provider response format, truncation, healing prompts, schema strictness, transport timeout, or concurrency.

### Parallel lane C — resumable per-record evaluator

- Define a versioned durable record-completion receipt.
- Make trial evaluation resumable at individual semantic-record granularity.
- Aggregate only after all required units complete.
- Keep successful cached work reusable across crash/restart and exact-parameter replay.
- Preserve deterministic and scientific gates.

### Parallel lane D — budget and circuit concurrency

- Reproduce one ambiguous response while unrelated requests are queued.
- Redesign circuit behavior so it prevents ambiguous duplicate billing without poisoning unrelated work.
- Add reconciliation and crash-resume tests.
- Review lock ordering and multi-worker behavior adversarially.

### Parallel lane E — replay and study lineage

- Design a versioned rescore/restart mechanism for the preserved 224 trials.
- Do not edit old receipts or journal history.
- Preserve exact parameter combinations and distinguish old operational records from new scored attempts.
- Keep Optuna blind to operational failures.
- Do not begin concentration based on fake coverage.

### Parallel lane F — adversarial review

Independently attack:

- request identity collisions;
- duplicate billing after timeout or crash;
- terminal-negative-cache poisoning;
- incompatible adapter receipt reuse;
- one failed record erasing successful siblings;
- malformed JSON becoming accepted truth;
- scientific failures being relabeled operational or successful;
- W&B influencing authoritative state;
- unsafe resumption from `finalization_reserved`;
- race conditions across eight workers.

### Parallel lane G — access and deployment readiness

- Verify GitHub authentication and remote-main synchronization.
- Verify AWS role access using the renewable profile without browser login.
- Verify Vast CLI access and current instance state.
- Verify the local presence—not the value—of OpenRouter and W&B keys.
- Ensure the remote workload can receive credentials without printing or committing them.
- Confirm a fresh clone can run focused offline tests.

The coordinator should integrate only after tests and adversarial review. Agents must not revert one another’s work or touch unrelated untracked files.

## Required tests

Add focused regression tests for at least all of these cases:

1. One failed judge record among N does not erase N-1 successful records.
2. A resumed trial retries only missing records.
3. An ambiguous request cannot be blindly rebilled after crash.
4. One ambiguous transport result does not poison unrelated queued requests.
5. A successful cache result is reused across replay with compatible identity.
6. An incompatible judge or adapter identity is rejected fail closed.
7. Valid original JSON passes without healing.
8. Valid healed JSON passes strict validation.
9. Malformed or semantically incomplete healed JSON remains a failure.
10. Empty response, truncated response, fenced JSON, tool-like JSON, and provider-specific content shapes are handled deliberately.
11. A transport circuit failure never becomes a terminal semantic alias.
12. Exact-parameter replay eventually yields a complete score once missing records succeed.
13. Operational failures never enter Optuna’s learning set.
14. Scientifically infeasible trials remain scientifically infeasible after judge hardening.
15. Crash/restart preserves request identity, completed units, cost receipts, and W&B run identity.
16. Concurrent workers cannot double-spend the same request.
17. A fully operational test matrix achieves 100% scored trials.
18. Finalization remains blocked for an unscored study and succeeds only for a selection-ready versioned study.

Use stored/mock responses for the broad matrix. Use only a small paid live matrix to validate the real OpenRouter boundary.

The last known focused suite before this handoff passed:

```text
113 passed in 2.41s
```

It covered adaptive-run, phase-checkpoint, live-judge, and checkpoint-transfer tests. That result proves only that the existing tests passed; it does not validate production scoring.

## Definition of done

Do not declare success until all of the following are true:

- the historical failure classes have deterministic offline reproductions;
- the root failure-amplification paths are fixed, not merely retried more often;
- every successful judge unit survives sibling failure and process restart;
- ambiguous billing is reconciled without poisoning unrelated work;
- response healing is verified against representative stored and live response shapes;
- malformed outputs still fail closed;
- the paid micro-matrix returns complete valid judgments with the frozen provider and judge settings;
- a multi-trial integration run yields 100% **operationally scored** trials for all valid fixtures;
- genuine scientific infeasibility remains visible and excluded from the successful-scored count;
- operational failures remain excluded from Optuna learning and broad-coverage credit;
- the preserved 224-trial history is not rewritten or deleted;
- the restart/rescore lineage is explicit and versioned;
- focused and broader relevant tests pass;
- an adversarial reviewer signs off on billing, identity, replay, concurrency, and scientific integrity;
- the reviewed implementation is pushed to `main` and remote `main` is verified;
- a fresh clone passes the focused offline tests;
- no secret, prompt corpus, model weight, temporary output, or unrelated user artifact is committed;
- cloud resources are either deliberately in use or cleaned up, never forgotten idle.

“100% scoring” means no operationally valid trial is lost to infrastructure, transport, schema handling, cache, replay, or orchestration. It does not mean forcing scientifically infeasible interventions to pass.

## Access playbook

### GitHub

GitHub CLI authentication and `origin/main` access worked at the last check. Reverify without printing credentials. Never force-push. If newer remote work exists, fetch it and integrate carefully before pushing.

### AWS

Use the renewable IAM Roles Anywhere profile:

```text
AWS_PROFILE=intelligent-liars-run
```

It is restricted to the `intelligent-liars` S3 bucket. Do not run browser login unless the renewable role genuinely fails. Exact-object checks may work even when bucket listing is intentionally denied. Do not access sealed audit data.

The remote machine previously lacked the `aws` CLI; that is not evidence of invalid credentials. Use a safe SDK path or install the CLI if needed. Remote credential-process material is under `/workspace/.truth-editing-aws/`.

### OpenRouter and W&B

The local untracked `.env` contains `OPENROUTER_API_KEY` and `WANDB_API_KEY`. Verify only their presence and inject them without echoing values. Never commit `.env`. The remote repository did not persist the OpenRouter key; prior launches supplied it through a secure environment handoff.

### Vast

The Vast CLI is installed locally. Inspect instance `49087336` first. Do not expose its web token. If replacing it with one GPU, record instance ID, hourly rate, start time, purpose, and cleanup state in the live run history.

## Live history and reporting

Append concise entries to `docs/truth_editing_live_run_history.md` for each meaningful failure and fix. Include:

- timestamp;
- observed failure;
- scope and evidence;
- root cause or current hypothesis;
- fix applied;
- verification result;
- any cost or cloud mutation.

Do not flood it with repetitive polling. Preserve the remote JSONL history as audit evidence.

At the end, report separately:

- files changed;
- tests and results;
- architectural decisions;
- exact live OpenRouter evidence and cost, without sensitive payloads;
- GitHub/AWS/Vast/W&B readiness;
- remaining gaps;
- software readiness versus behavioral/scientific evidence;
- external mutations and spending;
- current resource cleanup state.

## First fifteen minutes

1. Read the mandatory documents and current Git state.
2. Reconfirm Vast instance `49087336`; stop unnecessary idle spend or deliberately reuse it.
3. Snapshot the durable remote checkpoint, judge ledger, cache counts, and failure classes without modifying them.
4. Spawn the non-overlapping forensic, parser/healing, circuit, evaluator, replay, access, and adversarial-review lanes.
5. Reproduce at least one schema failure and one circuit-cascade failure offline.
6. Start test-first fixes at the record-completion and circuit boundaries.

Do not spend the first hour restating the problem. The durable evidence already shows where to begin.
