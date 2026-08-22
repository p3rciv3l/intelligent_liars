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
- Grouped probe qualification receipt: `f21781fdadab2eab6773d3e324d7500132e1f5f9e4bb38696c50837a07693b54`
- Regularizer probe file: `artifacts/probes/step5_grouped_ensemble_v1/probes/legacy-grouped-regularizer.json`
- Model revision: `92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b`
- Model content identity: `bbca6a8b09a56f0c538887b82b9594b0c0945c5fbcde54f39eda153a9f64eda8`
- PixMo content identity: `6938f0269264cb47cb753ef421a511ca81973d4a9d6dee6f83f6f029220c5c97`

The model cache is fixed at:

```text
s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/model-cache/v1/qwen--qwen3-vl-8b-thinking/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b/bbca6a8b09a56f0c538887b82b9594b0c0945c5fbcde54f39eda153a9f64eda8/
```

The portable image bundle is fixed at:

```text
s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/step5-assets/v1/pixmo/6938f0269264cb47cb753ef421a511ca81973d4a9d6dee6f83f6f029220c5c97/
```

Its tar SHA-256 is `284ec9d7d1f3e4833f8e71e6028e5f7d0ce039d2749742b220c488d2de2e6d55`.
The plan, corpora, and complete grouped probe bundle are in the single frozen
input object:

```text
s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/step5-inputs/v1/5282aaf0696098970de476e6812534e4c75c2434d7ec369495f5a311d6c09f99/step5_plan_5282aaf_inputs.tar.gz
```

Its SHA-256 is `b1b40983e39463f2c1cc283441dc2f2d13fba1b1b4850728771ae55deafb322f`.
The completion sibling hashes to
`82af3c0259465dde974b5dafc998d65ee2f07e3f9f41f256f663c97865442a58`.
The worker uses this S3 object, not the currently pending DVC remote.

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
training/projection seeds, and an external checkpoint durability verifier. A
finalizer must then normalize the fixed output inventory, publish its durable
object, and write the lifecycle-compatible `artifact_manifest.json`.

The diagnostic command conservatively reports `software_failure` whenever it
can still run over SSH. It captures disk, GPU, Python/CUDA, offline-cache, and
checkpoint-pointer state and never requests replacement. The recovery command
requires the same checkpoint pointer and revalidates the inert packet; the
lifecycle wrapper then reruns the identical remote command, whose runner
validates the checkpoint identity before resuming.

## Remaining substitutions before even a dry-run rental command is complete

1. `RUNTIME_IMAGE` and its matching `RUNTIME_IMAGE_DIGEST`, after the image is
   published by immutable digest.
2. `OFFER_ID`, `GPU_NAME`, and measured `GPU_VRAM_GIB`, selected from a fresh
   Vast search.
3. Exact `APPROVED_HOURLY_PRICE_USD`, `APPROVED_MAX_COST_USD`, and
   `APPROVED_AT` after the user sees and approves those concrete values.
4. `CHECKPOINT_DURABILITY_COMMAND`, a reviewed command that uploads each
   generation and emits the runner's exact durability receipt.
5. `CANARY_ARTIFACT_FINALIZE_COMMAND`, a reviewed command that produces the
   frozen nine-file inventory, publishes it, and writes the lifecycle artifact
   manifest.

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
