# Step 5 canary launch packet

Status: **prepared but deliberately blocked**

Execution: **disabled; no GPU, upload, or workload action is performed by this packet**

This is the controller handoff for the first single-GPU Step 5 prerequisite
canary. It proves that the exact runtime can load the frozen inputs and model,
measure host download speed, run the 13-coordinate arm's reachability and
intentional-overfit prerequisites, publish durable checkpoints, survive a
same-instance recovery, and return a fixed artifact inventory. It is not the
three-arm bounded screen and it does not claim the later seven scoring receipts.

## Frozen identities

- Step 5 plan SHA-256: `5282aaf0696098970de476e6812534e4c75c2434d7ec369495f5a311d6c09f99`
- Grouped probe qualification file: `6f409af64387a2a2ee1ca7f21e7fcdc7fbad1eeffa22672b8872277f45937ff2`
- Grouped probe qualification receipt: `f21781fdadab2eab6773d3e324d7500132e1f5f9e4bb38696c50837a07693b54`
- Regularizer probe file: `artifacts/probes/step5_grouped_ensemble_v1/probes/legacy-grouped-regularizer.json`
- Model revision: `92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b`
- Model content identity: `bbca6a8b09a56f0c538887b82b9594b0c0945c5fbcde54f39eda153a9f64eda8`
- PixMo content identity: `430de1b25babb4fcd462ed7cf0476bce9f8c4e2b8fc3872c843eea332c1e56bc`

The model cache is fixed at:

```text
s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/model-cache/v1/qwen--qwen3-vl-8b-thinking/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b/bbca6a8b09a56f0c538887b82b9594b0c0945c5fbcde54f39eda153a9f64eda8/
```

The portable image bundle is fixed at:

```text
s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/step5-assets/v1/pixmo/430de1b25babb4fcd462ed7cf0476bce9f8c4e2b8fc3872c843eea332c1e56bc/
```

Its tar SHA-256 is `284ec9d7d1f3e4833f8e71e6028e5f7d0ce039d2749742b220c488d2de2e6d55`.
The plan, corpora, and complete grouped probe bundle are in the single frozen
input object:

```text
s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/step5-inputs/v2/5282aaf0696098970de476e6812534e4c75c2434d7ec369495f5a311d6c09f99/09009258f4b2422aa12e547568536d9f5e15d14afc64e23ce8f86c33d1000439/step5_plan_5282aaf_inputs.clean.tar.gz
```

Its SHA-256 is `09009258f4b2422aa12e547568536d9f5e15d14afc64e23ce8f86c33d1000439`.
The completion sibling hashes to
`579664fd8e59f8d81f0c6d71b469d5f1011811100e981887df02af53e59f4882`.
The worker uses this clean S3 object. Tracked DVC files are local provenance
pointers, not worker payloads.

## What crosses onto the worker

The source manifest is a positive list of regular files with an exact SHA-256
for each file. It contains only the runner, lifecycle/diagnostic utilities, and
their required Python modules. The bundle verifier rejects symlinks, traversal,
secret-like paths, `.git`, credentials, and the sealed audit. Large corpus,
probe, model, and PixMo trees are absent because they are hydrated separately
from the immutable S3 objects above.

## Frozen canary outputs

The lifecycle wrapper accepts exactly the files in
`configs/tinylora_step5_canary_expected_artifacts_v1.json`, plus its own
`artifact_manifest.json` control file:

- result and prerequisite receipts;
- the 13-coordinate basis;
- a durable checkpoint-store archive;
- host, source-bundle, and input-hydration receipts;
- one canary summary and worker log.

No wildcard or extra file is accepted. The controller destroys the worker only
after the fetched files match this inventory and the S3 durable object passes
versioned HEAD, byte-size, checksum, and round-trip SHA-256 verification.
Otherwise the wrapper stops the same instance for recovery.

## Exact remote operation

The packet's `commands.remote` runs the prerequisite mode for
`tinylora_dim13` with 20 optimizer steps, one visible GPU, the exact plan,
regularizer probe, offline model cache, immutable runtime image digest, fixed
training/projection seeds, and an external checkpoint durability verifier. The
exact reviewed finalizer normalizes the fixed output inventory, publishes its
immutable durable object through a protected presigned URL, and writes the
lifecycle-compatible `artifact_manifest.json`. The credentialless checkpoint
bridge publishes each generation through the trusted controller.

The diagnostic command conservatively reports `software_failure` whenever it
can still run over SSH. It captures disk, GPU, Python/CUDA, offline-cache, and
checkpoint-pointer state and never requests replacement. The recovery command
requires a valid checkpoint pointer; the lifecycle wrapper then reruns the
identical remote command, whose runner validates the checkpoint identity before
resuming.

## Remaining substitutions before even a dry-run rental command is complete

1. `RUNTIME_IMAGE` and its matching `RUNTIME_IMAGE_DIGEST`, after the image is
   published by immutable digest.
2. `OFFER_ID`, `GPU_NAME`, and measured `GPU_VRAM_GIB`, selected from a fresh
   Vast search.
3. Exact `APPROVED_HOURLY_PRICE_USD`, `APPROVED_MAX_COST_USD`, and
   `APPROVED_AT` after the user sees and approves those concrete values.
4. `BUCKET_VERSIONING_STATUS=Enabled` and the SHA-256 of a fresh controller
   receipt proving that exact bucket state. It is currently disabled or
   unconfigured, so this remains a hard blocker.

The hydration, checkpoint durability, controller provisioning, and artifact
finalization commands are exact and hash-bound. The packet remains blocked only
on the immutable image, fresh offer/cost approval, and bucket-versioning proof.
Protected URL files must also exist as mode-0600 regular files at the frozen
controller paths before the complete validator will pass.

The validator reports `launch_ready: false` and exits 2 while any substitution
remains. `--allow-incomplete` changes only that reporting exit code for CI; it
does not execute anything. The packet contains no `--execute` or
`--confirmed-cost-approval` flag. Adding `--execute`, enabling execution,
changing any bound byte, or supplying a mutable image tag invalidates it.

Validate the inert packet with:

```bash
PYTHONPATH=src python scripts/validate_tinylora_step5_launch_packet.py \
  --packet configs/tinylora_step5_canary_launch_packet_v1.json \
  --allow-incomplete
```

After all substitutions are frozen, rerun without `--allow-incomplete`, inspect
the rendered argv, and separately request price-specific approval. Only a later
controller action may append both lifecycle execution flags. This packet never
does so itself.

## Boundary after the canary

If the prerequisite canary passes, each arm gets its own frozen
`prerequisite_receipt.json`, hash-bound to the plan, probe, code, arm, model, and
runtime image. The bounded screen then produces the separate base/candidate
paired and generation receipts, preservation, safety/XSTest, and probe receipts
required by the five-gate evaluator. Those are intentionally excluded from this
prerequisite canary inventory so a small systems qualification cannot masquerade
as behavioral evidence.
