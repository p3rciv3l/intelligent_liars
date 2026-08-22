#!/usr/bin/env python3
"""Create a protected, no-clobber presigned S3 PUT URL on the controller."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--url-file", type=Path, required=True)
    parser.add_argument("--expires-in", type=int, default=21600)
    return parser.parse_args()


def validate_destination(bucket: str, key: str, expires_in: int) -> None:
    if re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) is None:
        raise ValueError("bucket is not a valid S3 bucket name")
    pure = PurePosixPath(key)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != key:
        raise ValueError("key must be a safe relative S3 key")
    if not 60 <= expires_in <= 604800:
        raise ValueError("expires-in must be between 60 and 604800 seconds")


def write_private(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to replace an existing presigned URL file")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(value.strip() + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    validate_destination(args.bucket, args.key, args.expires_in)
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError("boto3 is required on the trusted controller") from error
    url = boto3.client("s3").generate_presigned_url(
        "put_object",
        Params={
            "Bucket": args.bucket,
            "Key": args.key,
            "IfNoneMatch": "*",
        },
        ExpiresIn=args.expires_in,
        HttpMethod="PUT",
    )
    write_private(args.url_file, url)
    print(
        json.dumps(
            {
                "format": "tinylora_step5_presigned_put_plan_v1",
                "durable_uri": f"s3://{args.bucket}/{args.key}",
                "expires_in": args.expires_in,
                "url_file": str(args.url_file),
                "if_none_match": "*",
                "checksum_verification": "controller_roundtrip_sha256",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
