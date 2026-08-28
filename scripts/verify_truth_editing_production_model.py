#!/usr/bin/env python3
"""Re-hash the hydrated production model and publish its exact identity."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligent_liars.model_cache import verify_huggingface_cache_for_loading  # noqa: E402


def main() -> int:
    expected_model = os.environ.get("TRUTH_EDITING_EXPECTED_MODEL_SHA256", "")
    expected_manifest = os.environ.get(
        "TRUTH_EDITING_EXPECTED_SNAPSHOT_MANIFEST_SHA256", ""
    )
    cache = ROOT / "artifacts/truth-editing/model-cache/huggingface"
    manifest = ROOT / "artifacts/truth-editing/model-cache/snapshot-manifest.json"
    identity = verify_huggingface_cache_for_loading(
        cache_dir=cache,
        manifest_path=manifest,
        expected_model_sha256=expected_model,
        expected_manifest_sha256=expected_manifest,
    )
    unsigned = {
        "format": "truth_editing_production_model_verification_v1",
        "model_id": identity.model_id,
        "revision": identity.revision,
        "model_sha256": identity.model_sha256,
        "snapshot_manifest_sha256": identity.snapshot_manifest_sha256,
    }
    payload = {**unsigned, "self_sha256": hashlib.sha256(
        json.dumps(unsigned, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()}
    output = Path("/workspace/outputs/model/model-verification-receipt.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
