# Qwen3-VL Capability Baseline and OSWorld Evaluation Plan

Status: planning only  
Created: 2026-07-22

This document defines the proposed evaluation lifecycle and its cost model. It
does not authorize infrastructure creation or benchmark execution.

## Objective

Establish an immutable capability baseline for the unmodified
`Qwen/Qwen3-VL-8B-Thinking` model, then evaluate selected layer-21
truth-direction interventions under the same harness.

The baseline consists of:

1. Full 5-shot MMLU over all 57 subjects and 14,042 test questions.
2. Official OSWorld-Verified over all 369 Ubuntu tasks, initially capped at 15
   steps per task.
3. Complete trajectories containing the initial screenshot, each subsequent
   screenshot, raw model response and reasoning, parsed action, execution
   result, reward, timestamps, retries, errors, task score, logs, and
   recording.

OSWorld's task-specific evaluator scores the final desktop state. KL divergence,
if measured, is a separate model-drift diagnostic.

## Facts established before implementation

- The official repository's `evaluation_examples/test_small.json` contains 39
  tasks. `test_all.json` contains 369 tasks, and `test_nogdrive.json` contains
  361 tasks.
- The eight Google Drive tasks require manual account and OAuth setup. The
  project's target remains 369 tasks; the 361-task manifest is acceptable only
  for debugging or as an explicitly labeled fallback.
- OSWorld's AWS provider uses a host-client architecture. The host runs the
  harness and launches one EC2 client for each concurrent desktop environment
  from an official OSWorld AMI.
- The current AWS provider creates `t3.xlarge` clients with 30 GB gp3 volumes,
  4,000 IOPS, 1,000 MB/s throughput, public IPv4 addresses, and a default
  three-hour termination TTL.
- OSWorld's published “within one hour” statement refers to AWS
  parallelization of desktop environments, including demonstrations of up to
  50 concurrent environments. It is not a guarantee for every model, step
  limit, endpoint, or account quota.
- The current Capy VM has effective privileged Docker but no `/dev/kvm`, so it
  cannot host the stock accelerated Docker/QEMU desktop provider.
- The official Qwen3-VL runner is useful reference code but is not ready for
  this experiment unchanged: its runner constructs `Qwen3VLAgent` without
  selecting the OpenAI backend, while the agent defaults to DashScope. The
  endpoint adapter, response capture, and action parser therefore require
  explicit validation.

The OSWorld repository inspected for this plan was
`b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf`. The execution manifest must pin a
reviewed commit rather than silently following `main`.

## Proposed architecture

### Control plane

Captain coordinates the experiment but does not choose computer actions. A
durable controller on the AWS host owns the immutable task queue, leases tasks
to workers, resumes incomplete work, and writes status events.

Capy Build agents may implement or diagnose components, but they are not the
durable monitors. Monitoring must continue if a Capy task VM is reclaimed.

### Model plane

One separately managed Vast.ai GPU serves the exact model revision through an
authenticated OpenAI-compatible endpoint. The server records its model
revision, tokenizer and processor revisions, package lock, generation
parameters, quantization and dtype, intervention configuration, and container
image digest.

Serving uses one BF16 model worker per GPU. Horizontal scaling means placing
multiple identical workers behind the AWS controller; it must not silently
change precision or introduce quantization.

The model server binds to loopback and the AWS controller reaches it through an
SSH local-forward. A non-loopback bind is an explicit unsafe override that
requires a separately secured transport.

The baseline and interventions must use the same serving stack. A layer-21
intervention may require a custom Transformers server rather than an
unmodified vLLM deployment, so endpoint throughput must be measured rather than
assumed.

### Desktop plane

An AWS host in `us-east-1` launches official OSWorld AMI clients in the same
VPC and subnet. Each worker performs this loop:

1. Lease one task from the frozen manifest.
2. Create or reset an official client.
3. Apply the task's official setup procedure.
4. Send the instruction, screenshot, and frozen history format to Qwen.
5. Store the raw response before parsing.
6. Parse exactly one Qwen-selected action and execute it.
7. Record the transition and repeat until `DONE`, `FAIL`, an error, or the step
   cap.
8. Run the official evaluator.
9. Upload the complete task bundle and checksum before acknowledging the task.
10. Terminate the client and release the lease.

Captain must never substitute its own computer-use decision for Qwen's action.

### Artifact plane

Large trajectories, screenshots, and recordings stay outside Git. Each run has
an append-only prefix derived from a run ID and frozen manifest hash. Every task
bundle is uploaded immediately, checksummed remotely, and only then marked
complete. A second manifest records missing, duplicate, retried, and superseded
attempts without overwriting any valid attempt.

The storage backend and retention price must be selected before execution.
Credentials are injected at runtime and are never stored in the model image,
repository, logs, or trajectory payloads.

## Monitoring and shutdown

Three durable processes are sufficient; adding one Capy agent per EC2 client
would add cost without improving control.

1. **Controller:** Tracks the frozen task grid, leases, attempts, terminal
   states, artifact checksums, and aggregate score. It distinguishes success,
   task failure, refusal, invalid action, model error, evaluator error, and
   infrastructure failure.
2. **GPU watchdog:** Checks endpoint authentication, model fingerprint, queue
   depth, request latency, error rate, GPU utilization and memory, disk usage,
   and Vast instance state. It rejects new leases when the endpoint is
   unhealthy.
3. **AWS watchdog:** Reconciles expected and actual EC2 clients, task
   heartbeats, TTL schedules, security groups, and artifact uploads. It
   terminates orphaned clients and finally the host after durable export.

Independent cloud-side kill switches are mandatory:

- Every AWS client gets a short termination TTL.
- The AWS host gets a separate maximum-lifetime termination schedule.
- The Vast instance gets a forced-stop deadline controlled outside the model
  server.
- New task leasing stops before any deadline so active tasks have time to
  export.

## Staged execution gates

No later stage begins merely because the preceding process exited. Its
artifacts and acceptance checks must pass.

1. **Offline harness tests:** Validate prompt construction, screenshot encoding,
   raw-response retention, parser behavior, evaluator invocation, task leasing,
   resume behavior, and immutable manifests without cloud resources.
2. **First paid production tranche:** Execute the frozen one-task 50-step manifest
   with the same `build_official_agent`, `build_official_aws_environment`, and
   `run_attempt` path used by every later task. It uses the pinned official task
   schema and evaluator, captures the complete trajectory and recording, and
   performs the same append-only upload and remote checksum verification as
   later tranches. A scored `task_failure` proves that the production harness
   ran and may pass this setup gate; model score is not the setup criterion.
   Refusal, invalid action, model, evaluator, infrastructure, capability, or
   setup failure stops immediately, and no later task may be leased. This task
   is exactly the first-task prefix of the frozen five-task manifest. Leasing
   any remaining task requires `require_first_tranche_pass`.
3. **Frozen five-task subset:** After the first-tranche gate passes, lease only
   the remaining four tasks from `pilot_five_50.template.json`. Validate retries,
   incremental checkpoints, evaluator outputs, and that no completed result is
   overwritten.
4. **Frozen no-Google-Drive baseline:** Run the exact 361-task, 50-step
   `test_nogdrive_361_50.template.json` manifest, initially with eight
   environments. Increase concurrency only if the model endpoint sustains it
   without materially changing per-step latency or failure rate.
5. **100-step promotion:** Materialize a separate frozen 100-step grid only when
   measured projected total remains strictly below $60. Never extend completed
   50-step trajectories in place.
6. **Intervention runs:** Begin only after the unmodified baseline is complete,
   validated, and immutable. Change only the intervention fields in the run
   manifest.

MMLU follows the same baseline-first rule and uses its own frozen dataset,
prompt, answer parser, and scoring manifest.

## Runtime and cost model

These are planning estimates for one OSWorld run, not approved spend. They use
on-demand AWS clients because interruption complicates task-level evidence.
All prices are USD and were checked on 2026-07-22.

### Unit assumptions

| Item | Planning rate |
|---|---:|
| AWS `t3.xlarge` client compute, us-east-1 | $0.1664/hour |
| Client 30 GB gp3 storage | $0.0033/hour |
| Client gp3 IOPS above baseline | $0.0068/hour |
| Client gp3 throughput above baseline | $0.0479/hour |
| Client public IPv4 | $0.0050/hour |
| **Total AWS client** | **$0.2295/hour** |
| Small-run host (`t3.medium`, storage and IPv4 included) | about $0.052/hour |
| Full-run host (`t3.large`, storage and IPv4 included) | about $0.094/hour |
| Vast L40S planning allowance | $0.60/hour |

The official guide recommends `t3.medium` for fewer than five environments and
`t3.large` for fewer than fifteen. It recommends a much larger host such as
`c4.8xlarge` above fifteen environments; that instance alone is about
$1.591/hour. The initial plan therefore uses four environments for Small and
eight for Full rather than buying 30 idle desktops before measuring the single
GPU endpoint.

Each estimate assumes:

- 90 seconds of client creation/reset/setup overhead per task.
- Every task consumes the full configured step cap. Early `DONE` or `FAIL`
  reduces cost.
- The low, base, and high cases use 10, 20, and 30 seconds per step,
  respectively, including model queueing, inference, action execution,
  screenshot capture, and evaluator overhead.
- Four concurrent clients for Small and eight for Full.
- The model endpoint sustains that concurrency. If requests serialize and
  clients wait, both AWS and Vast runtime increase.

### Estimated cost per single run

| Suite | Tasks | Step cap | Low | Base | High | High + 20% stop ceiling |
|---|---:|---:|---:|---:|---:|---:|
| OSWorld-Small | 39 | 15 | $1.02 | $1.66 | $2.30 | $2.76 |
| OSWorld-Small | 39 | 50 | $2.51 | $4.63 | $6.76 | $8.11 |
| OSWorld-Small | 39 | 100 | $4.63 | $8.89 | $13.14 | $15.77 |
| OSWorld Full | 369 | 15 | $7.78 | $12.64 | $17.50 | $21.00 |
| OSWorld Full | 369 | 50 | $19.12 | $35.33 | $51.53 | $61.84 |
| OSWorld Full | 369 | 100 | $35.33 | $67.74 | $100.15 | $120.18 |

The corresponding base-case wall-clock estimates are about 1.1, 3.0, and 5.7
hours for Small, and 5.0, 14.0, and 26.8 hours for Full. They deliberately do
not repeat OSWorld's one-hour marketing claim because one local 8B endpoint may
be the bottleneck.

Running the baseline plus one intervention approximately doubles the table
after the harness is validated. The 361-task no-Google-Drive grid is about 2.2%
smaller than the 369-task grid, which is not enough to change infrastructure
decisions.

### Exclusions and price gates

The table excludes:

- Capy VM/runtime charges.
- MMLU inference.
- Vast storage and bandwidth premiums attached to the selected marketplace
  offer.
- Proxy traffic, listed by OSWorld at approximately $1/GB.
- Durable object storage and retrieval.
- AWS data transfer beyond any account allowance.
- Failed-attempt retries and manual setup time.
- Taxes and account-specific credits.

Immediately before approval, revalidate measured Vast candidates by projected
total completed-run cost and wall-clock time, not hourly price. The selected
offer must be on the measured Pareto frontier; among frontier choices, select
lower total completed-run cost and then lower wall-clock time. A higher-hourly-
spend or multi-GPU choice requires approval-bound Pareto evidence and a non-empty
necessity justification. The approval binds the frontier's minimum measured
completed-run cost and the best wall-clock time at that minimum. One GPU remains
the default. Before any paid launch,
Captain must present that exact Vast offer, evidence hash, AWS instance counts
and TTLs, storage target, provider-specific maximum spend, and exact teardown
commands. No launch may occur until the Qwen endpoint URL/API-key and
`OSWORLD_CLIENT_PASSWORD` gates pass and the exact proposal is explicitly
approved.

### Authoritative authorization and teardown rules

The table above remains a planning input, not spend authorization. Cumulative
AWS spend has a hard ceiling keyed by the frozen run step cap: $75 for a 50-step
run and $120 for a 100-step run. A 100-step run may be promoted only when its
updated projected total is strictly below $60. The combined
baseline-plus-intervention envelope is $140. These gates apply even when a table
ceiling is higher.

Vast GPU spend has a separate cumulative $20 hard ceiling for the entire
evaluation across all tasks, tranches, and runs; it is never a per-task or
per-tranche allowance. AWS controller, client, storage, address, and transfer
spend is accounted and enforced separately. Provider samples remain cumulative
across every resource and tranche.

Cloud launch paths default to dry-run. Vast launches default to one GPU and
require approval of the measured completed-run cost, wall-clock projection,
Pareto evidence, concrete offer, and hourly rate. The Vast CLI path must be
resolved explicitly on Linux rather than inherited from a workstation default.
AWS execution uses the official OSWorld host/client provider; nested KVM on a
Capy VM is not an execution target.

Health or budget failures stop new task leases before teardown. Active workers
may export artifacts, and completed, surplus, stopped, or stale labeled
resources are terminated only after local and remote artifact SHA-256 values
match in the append-only artifact manifest. Teardown terminates instances
rather than leaving them stopped, deletes unattached billable volumes, records
resource termination IDs, and verifies that worker count shrinks to zero.
The controller polls leases and run tags, immediately removes checksum-verified
idle or surplus resources, and reconciles every provider to zero resources.

Every task incrementally checkpoints the frozen manifest, attempt ledger,
trajectory, screenshots, video, logs, result, metadata, and checksums to a unique
append-only S3 checkpoint prefix as work proceeds. S3 is authoritative. DVC,
Drive, Hugging Face, and Git LFS are optional mirrors only after S3 checksum
verification and are never required connectors. Pause, stop, or destroy requires
a remotely verified checksum set and resumable manifest state. A pause with
partially exported state is allowed only when the provider preserves all
required disk or volume state; otherwise the worker drains and exports first.
Stop or destroy requires a complete durable-state checkpoint, and unsynced state
is never destroyed.

The Vast lifecycle policy source inspected for these rules is private
`p3rciv3l/avi-skills` commit
`f453c250b3c5a684f42f6cf6abb97463dddda14c`, path
`.agents/skills/vast-gpu-experiments/`, including `vast_find_offer.py`,
`vast_run_workload.py`, and `vast_cleanup.py`. The private mirror also contains
`gmail`, `gmail-inbox-triage`, and `research`. Native Gmail or Bitwarden MCP
connectors are not assumed or required, and payment access is excluded.
Preflight reports only the configured environment-variable names and presence
status and never copies secret values.

Vast credentials use only these scopes: `misc` for offer search,
`instance_read` for inventory, status, logs, and volumes, and
`instance_write` for instance creation, management, and destruction. They
explicitly exclude `billing_write`, `user_write`, all `machine_*` scopes, and
payment access.

The AWS controller should use an attached IAM role rather than long-lived
keys. Access key, secret key, and optional session-token environment names are
supported only as bootstrap gates. Runtime permission is limited to the run's
VPC, tagged resources, artifact prefix, and termination role:

- `sts:GetCallerIdentity`;
- EC2 describe image/instance/tag/volume actions, `ec2:RunInstances`,
  `ec2:TerminateInstances`, `ec2:CreateTags`, and `ec2:DeleteVolume`, plus only
  additional image/reset actions demonstrated necessary by the pinned official
  OSWorld provider;
- `scheduler:CreateSchedule`, `scheduler:GetSchedule`, and
  `scheduler:DeleteSchedule`, with `iam:PassRole` limited to the termination
  role;
- append-only artifact-prefix access using `s3:ListBucket`, `s3:PutObject`,
  `s3:GetObject`, `s3:AbortMultipartUpload`, and
  `s3:ListMultipartUploadParts`; and
- optional read-only `ce:GetCostAndUsage`.

No Gmail or Bitwarden MCP integration is assumed or claimed.

## Why model size does not imply a one-hour run

OSWorld's AWS clients execute desktop state transitions; the model is served
elsewhere. A very large proprietary model can still support a short wall-clock
run if its provider offers enough parallel inference capacity. Conversely, an
8B model on one GPU can be slower overall if thirty desktop clients queue
behind one generation worker or if thinking responses are long.

Parameter count affects fit and raw inference cost, but endpoint latency,
reasoning-token length, batching, request concurrency, action validity, early
termination, and the step cap determine evaluation time. The one-task,
five-task, and 361-task gates must measure those quantities before a larger
quote is treated as reliable.

## Frozen run manifest

Every run must record at least:

- repository commit and dirty-state assertion;
- OSWorld commit, manifest hash, task IDs, and evaluator hashes;
- model, tokenizer, and processor revisions;
- server image digest, package lock, dtype, quantization, and attention backend;
- baseline or intervention identity, layer, direction hash, sign, scale, and
  application point;
- system prompt, action schema, parser version, observation type, screenshot
  transform, history policy, generation parameters, and step cap;
- AWS AMI IDs, region, client type, host type, concurrency, and security-group
  policy;
- Vast offer ID, GPU model, driver/runtime versions, and endpoint fingerprint;
- task attempts, retries, terminal-state taxonomy, timestamps, and artifact
  checksums.

Any change to these fields creates a new run ID. Valid results are never
silently replaced or combined across manifests.

For the offline controller schema, `task_grid.sha256` is the lowercase SHA-256
of the canonical compact JSON encoding of the ordered `task_grid.task_ids`
array (UTF-8, with no insignificant whitespace). Changing membership or order
therefore invalidates the frozen grid and run ID.

## Sources

- OSWorld repository and README:
  <https://github.com/xlang-ai/OSWorld>
- OSWorld AWS public evaluation guide:
  <https://timothyxxx.github.io/OSWorld/user_guides/run_public_evaluation.html>
- OSWorld AWS provider guide:
  <https://github.com/xlang-ai/OSWorld/blob/main/desktop_env/providers/aws/AWS_GUIDELINE.md>
- OSWorld-Verified infrastructure report:
  <https://xlang.ai/blog/osworld-verified>
- AWS EC2 on-demand pricing:
  <https://aws.amazon.com/ec2/pricing/on-demand/>
- AWS EBS pricing:
  <https://aws.amazon.com/ebs/pricing/>
- Vast marketplace pricing:
  <https://vast.ai/pricing>
