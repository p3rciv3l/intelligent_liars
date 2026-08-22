#!/usr/bin/env python3
"""Capture a controller-side, read-only S3 versioning receipt."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import boto3

    s3 = boto3.client("s3")
    status = s3.get_bucket_versioning(Bucket=args.bucket).get("Status")
    if status != "Enabled":
        raise SystemExit(f"bucket versioning is not Enabled: {status!r}")
    region = s3.get_bucket_location(Bucket=args.bucket).get("LocationConstraint")
    receipt = {
        "account_id": boto3.client("sts").get_caller_identity()["Account"],
        "bucket": args.bucket,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "format": "tinylora_step5_bucket_versioning_receipt_v1",
        "region": region or "us-east-1",
        "status": status,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=args.output.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
