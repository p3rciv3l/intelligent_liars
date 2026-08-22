#!/usr/bin/env python3
"""Create a hash-bound presigned S3 PUT and trusted controller receipt."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from intelligent_liars.step5_artifact_presigner import (
    build_receipt,
    canonical_bytes,
    sha256_bytes,
    write_private_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--approved-at", required=True)
    parser.add_argument("--url-file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expires-in", type=int, default=21600)
    return parser.parse_args()


def validate_destination(bucket: str, key: str, expires_in: int) -> None:
    if re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) is None:
        raise ValueError("bucket is not a valid S3 bucket name")
    pure = PurePosixPath(key)
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", key) is None
        or pure.is_absolute()
        or ".." in pure.parts
        or pure.as_posix() != key
    ):
        raise ValueError("key must be a safe relative S3 key")
    if not 60 <= expires_in <= 604800:
        raise ValueError("expires-in must be between 60 and 604800 seconds")


def write_private(path: Path, value: str) -> None:
    """Backward-compatible protected single-file writer."""
    write_private_outputs(((path, (value.strip() + "\n").encode()),))


def verified_client(region: str, account_id: str):
    try:
        import boto3
        from botocore.config import Config
    except ImportError as error:
        raise RuntimeError("boto3 and botocore are required on the controller") from error
    session = boto3.session.Session(region_name=region)
    identity = session.client("sts", region_name=region).get_caller_identity()
    if str(identity.get("Account")) != account_id:
        raise ValueError("active AWS account differs from frozen account ID")
    return session.client(
        "s3",
        region_name=region,
        endpoint_url=f"https://s3.{region}.amazonaws.com",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},
        ),
    )


def verify_bucket_region(s3, *, bucket: str, region: str) -> None:
    response = s3.get_bucket_location(Bucket=bucket)
    actual = response.get("LocationConstraint") or "us-east-1"
    if actual == "EU":
        actual = "eu-west-1"
    if actual != region:
        raise ValueError("S3 bucket region differs from frozen region")


def main() -> int:
    args = parse_args()
    validate_destination(args.bucket, args.key, args.expires_in)
    if args.url_file.absolute() == args.receipt.absolute():
        raise ValueError("URL and receipt outputs must be distinct")
    s3 = verified_client(args.region, args.account_id)
    verify_bucket_region(s3, bucket=args.bucket, region=args.region)
    generated_at = datetime.now(timezone.utc)
    url = s3.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": args.bucket,
            "Key": args.key,
            "IfNoneMatch": "*",
        },
        ExpiresIn=args.expires_in,
        HttpMethod="PUT",
    )
    receipt = build_receipt(
        url=url,
        bucket=args.bucket,
        key=args.key,
        region=args.region,
        account_id=args.account_id,
        approved_at=args.approved_at,
        generated_at=generated_at,
        expiry_seconds=args.expires_in,
    )
    receipt_bytes = canonical_bytes(receipt)
    write_private_outputs(
        (
            (args.url_file, (url.strip() + "\n").encode()),
            (args.receipt, receipt_bytes),
        )
    )
    print(
        json.dumps(
            {
                "format": "tinylora_step5_presigned_put_plan_v2",
                "account_id": args.account_id,
                "approved_at": receipt["approved_at"],
                "durable_uri": receipt["durable_uri"],
                "expires_at": receipt["expires_at"],
                "method": "PUT",
                "receipt": str(args.receipt),
                "receipt_sha256": sha256_bytes(receipt_bytes),
                "url_file": str(args.url_file),
                "url_sha256": receipt["url_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
