# Model artifact registry

This repository uses `model-registry/v1` as the deterministic S3 namespace for
artifacts produced by truth-editing runs. The local contract is implemented in
[`src/intelligent_liars/model_registry.py`](src/intelligent_liars/model_registry.py),
with the machine-readable manifest schema at
[`schemas/model_registry_v1.schema.json`](schemas/model_registry_v1.schema.json).
The checked-in bucket and target-cache facts live in
[`configs/model_registry_v1.json`](configs/model_registry_v1.json).

## Prefix layout

Every published object is content-addressed and has one exact key. A registry
consumer must already know the key and immutable S3 `VersionId`; it must not
discover artifacts by listing a prefix.

```text
model-registry/v1/
  experiments/failed/<run-id>/<content-sha256>/<filename>
  experiments/successful/<run-id>/<content-sha256>/<filename>
  models/final/<model-slug>/<content-sha256>/<filename>
```

The three prefixes are deliberately distinct. A failed experiment is retained
for diagnosis, a successful experiment is retained as a reproducible result,
and a final model is a separately curated release artifact. The registry does
not infer scientific success from a storage category.

## Publication and verification contract

1. Hash the exact bytes locally and compute the destination with
   `artifact_key`.
2. Publish through the externally authorized uploader (the offline utilities in
   this repository do not upload anything).
3. Capture a non-empty, non-`null` S3 `VersionId` for the exact bucket/key.
4. Verify `HEAD` and then `GET` using that exact `VersionId`; compare byte count
   and SHA-256 to the local content identity.
5. Insert only the verified receipt into a registry and run `validate`.

`verify_s3_roundtrip` intentionally calls only exact-key `head_object` and
version-pinned `get_object`. Missing versions, changed versions, body errors,
size mismatches, hash mismatches, unknown fields, and unsupported schema
versions fail closed. A receipt with `verified: false` cannot enter a registry.

Offline checks:

```bash
PYTHONPATH=src .venv/bin/python scripts/model_registry.py config
PYTHONPATH=src .venv/bin/python scripts/model_registry.py validate path/to/registry.json
```

## Current target base cache

The target checkpoint is exactly
`Qwen/Qwen3-VL-8B-Thinking@92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b`.
The historical verified cache receipt records 14 files, 17,545,907,058 bytes,
and content SHA-256
`bbca6a8b09a56f0c538887b82b9594b0c0945c5fbcde54f39eda153a9f64eda8`.
Its canonical recorded prefix is:

```text
model-cache/v1/qwen--qwen3-vl-8b-thinking/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b/bbca6a8b09a56f0c538887b82b9594b0c0945c5fbcde54f39eda153a9f64eda8
```

Historical scheduler receipts also contain a `model-cache/v2/.../sha256meta`
prefix. That v2 object set has an internal manifest whose S3 paths still point
to `model-cache/v1`. This is an unresolved prefix/manifest mismatch, not a
verified second cache identity. The checked-in configuration therefore records
the mismatch explicitly and keeps v1 as the only current cache identity. Do
not promote v2 or silently rewrite the old objects; resolve it with an
independent, exact-key, version-pinned round-trip receipt before changing this
status.

No S3 write, upload, or deletion is part of this scaffold.
