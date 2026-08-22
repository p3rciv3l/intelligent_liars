# Step 5 multimodal asset bundle

The Step 5 vision-preservation rows are now portable without depending on the
checkout's ignored `data/` directory. The committed inventory
`configs/tinylora_step5_multimodal_assets_v1.json` covers every PixMo image
reference in both preservation corpora:

| Category | Rows | Unique images | Bytes |
|---|---:|---:|---:|
| charts | 50 | 50 | 6,752,915 |
| diagrams | 50 | 50 | 5,609,945 |
| other documents | 50 | 50 | 9,797,571 |
| tables | 50 | 50 | 6,464,031 |
| **Total** | **200** | **200** | **28,624,462** |

The manifest commitment is
`6938f0269264cb47cb753ef421a511ca81973d4a9d6dee6f83f6f029220c5c97`.
It also fixes one lexicographically selected smoke row per category.

## Guarantees

The builder fails closed unless every image reference is a traversal-free
relative path beneath the pinned PixMo snapshot directory. It rejects symlinks,
missing files, filename/hash disagreements, byte-hash disagreements, corrupt or
unsupported image decodes, duplicate image-row IDs, and non-vision categories.

The staged bundle has a content-addressed `images/<prefix>/<sha256>.jpg`
layout, rewritten copies of the full preservation corpora, a four-row smoke
corpus, and the manifest. Validation rejects missing or unexpected bundle
members and rechecks every byte hash, size, decode format, and dimension.
Archives use sorted members with fixed ownership, permissions, and timestamps,
so the same bundle produces the same uncompressed tar bytes.

## Build and validate

Run from the repository root with the project package available:

```bash
PYTHONPATH=src uv run python scripts/build_tinylora_step5_multimodal_bundle.py build \
  --output-dir /workspace/assets/tinylora-step5-pixmo \
  --archive /workspace/assets/tinylora-step5-pixmo.tar

PYTHONPATH=src uv run python scripts/build_tinylora_step5_multimodal_bundle.py validate \
  --bundle-root /workspace/assets/tinylora-step5-pixmo
```

Both output targets must be new paths; the tool never overwrites an existing
bundle or archive.

## Runner integration

Before handing multimodal rows to the Qwen processor, rebase their portable
bundle-relative image references to verified absolute paths:

```python
from intelligent_liars.step5_multimodal_assets import (
    rebase_image_references,
    validate_staged_bundle,
)

manifest = validate_staged_bundle(bundle_root)
rows = rebase_image_references(rows, bundle_root=bundle_root, manifest=manifest)
```

The command-line equivalent writes a new JSONL and refuses to overwrite it:

```bash
PYTHONPATH=src uv run python scripts/build_tinylora_step5_multimodal_bundle.py rebase \
  --bundle-root /workspace/assets/tinylora-step5-pixmo \
  --input /workspace/assets/tinylora-step5-pixmo/corpora/preservation_train.jsonl \
  --output /workspace/run/preservation_train.absolute.jsonl
```

This bundle is only a data/readiness artifact. Building it does not rent a GPU,
start training, or upload anything to S3.
