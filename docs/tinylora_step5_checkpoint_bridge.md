# Step 5 credentialless checkpoint bridge

This bridge lets a rented GPU publish checkpoints without receiving AWS
credentials. Each checkpoint generation is packaged as a deterministic tar
archive. The worker announces its generation ID, manifest hash, archive hash,
byte size, random request nonce, and request hash. The trusted controller then:

1. derives a unique S3 key from the fixed run prefix, generation ID, and archive
   hash;
2. sends the worker a short-lived, generation-specific, create-only PUT URL;
3. independently checks S3 HEAD metadata and its immutable version ID;
4. downloads that exact version and verifies the archive, embedded manifest,
   and every declared file hash; and
5. signs an acknowledgement with a controller-only Ed25519 private key.

The worker has only the public key. The runner advances `latest.json` only after
the worker has validated the signed acknowledgement and emitted the exact v2
durability receipt. Missing, wrong-generation, wrong-manifest, stale, future,
unsigned, and self-attested acknowledgements fail closed. Query-bearing signed
URLs exist only in mode-0600 files, are deleted after use, and never appear in
the receipt, log output, or process arguments. A retry reuses the same request
and object key; S3 `If-None-Match: *` prevents overwrite. Local checkpoint
retention remains current plus previous verified generation, while durable S3
generations are retained (not automatically deleted) for recovery.

## One-time controller key

Run on the trusted controller, never on the rented worker:

```bash
umask 077
mkdir -p .local/step5-checkpoint-controller
openssl genpkey -algorithm ED25519 \
  -out .local/step5-checkpoint-controller/private.pem
openssl pkey -in .local/step5-checkpoint-controller/private.pem -pubout \
  -out .local/step5-checkpoint-controller/public.pem
```

The lifecycle controller copies the public key into the reviewed worker input
tree at `/workspace/inputs/controller-public.pem` before training starts. The
private key stays local and its mode must be exactly 0600.

## Exact worker command

This is the value of `CHECKPOINT_DURABILITY_COMMAND` inside the immutable
runtime image. It contains no credential and no URL:

```bash
python scripts/publish_step5_checkpoint_worker.py \
  --exchange-dir /workspace/checkpoint-bridge \
  --controller-public-key /workspace/inputs/controller-public.pem \
  --timeout-seconds 1200
```

`/workspace/checkpoint-bridge` is private to one worker/candidate process; it is
never shared between concurrent arms. Superseded exchange state is removed only
after the replacement generation is locally complete.

The surrounding screen command also includes:

```bash
--durability-controller-public-key /workspace/inputs/controller-public.pem
```

The screen runner supplies `STEP5_CHECKPOINT_GENERATION_DIR`,
`STEP5_CHECKPOINT_GENERATION_ID`, and `STEP5_CHECKPOINT_MANIFEST_SHA256` in the
child environment. The worker returns the signed request/ack envelope; the
screen runner verifies it again before accepting the v2 receipt. Nothing needs
to predict the generation IDs or cadence.

## Exact trusted-controller command

Start this locally after the lifecycle wrapper has pinned the worker SSH host
key and before starting/resuming training. It can be restarted after an
interruption; completed exact objects are independently reverified and receive
the same request-bound acknowledgement without another PUT.

```bash
uv run --with boto3 python scripts/run_step5_checkpoint_controller.py \
  --ssh-host "$STEP5_WORKER_HOST" \
  --ssh-port "$STEP5_WORKER_PORT" \
  --known-hosts "$STEP5_KNOWN_HOSTS" \
  --remote-exchange-dir /workspace/checkpoint-bridge \
  --bucket intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21 \
  --prefix step5/checkpoints/step5-canary-v1/tinylora_dim13 \
  --controller-private-key .local/step5-checkpoint-controller/private.pem \
  --controller-public-key .local/step5-checkpoint-controller/public.pem \
  --url-expiry-seconds 1200 \
  --idle-timeout-seconds 300
```

The controller inherits the signed-in operator's normal AWS credential chain.
Those credentials never cross SSH. Bucket versioning is required: an object
without a non-null S3 version ID is rejected even if its bytes hash correctly.

Automatic S3 deletion is deliberately absent. The bridge's immutable key
layout supports a later, separately approved retention sweep over exact,
acknowledged version IDs; this can never block resume or erase the only durable
checkpoint generation during a failure.
