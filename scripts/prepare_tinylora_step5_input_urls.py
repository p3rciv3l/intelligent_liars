#!/usr/bin/env python3
"""Verify and presign frozen Step 5 inputs on the trusted controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from intelligent_liars.step5_input_url_controller import prepare_input_urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--manifest-bucket", required=True)
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--url-file", type=Path, required=True)
    parser.add_argument("--host-gate-url-file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expires-in", type=int, default=21600)
    return parser.parse_args()


def build_aws_clients(region: str):
    """Build regional SigV4 clients without making a cloud request."""

    import boto3
    from botocore.config import Config
    from botocore.session import get_session

    available_regions = set(
        get_session().get_available_regions(
            "s3", partition_name="aws", allow_non_regional=False
        )
    )
    if region not in available_regions:
        raise ValueError("region must be an SDK-known commercial AWS S3 region")

    def endpoint(service: str) -> str:
        hostname = f"{service}.{region}.amazonaws.com"
        value = f"https://{hostname}"
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.hostname != hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("region produced an unsafe AWS service endpoint")
        return value

    session = boto3.Session(region_name=region)
    s3_endpoint = endpoint("s3")
    sts_endpoint = endpoint("sts")
    s3 = session.client(
        "s3",
        endpoint_url=s3_endpoint,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},
        ),
    )
    sts = session.client(
        "sts",
        region_name=region,
        endpoint_url=sts_endpoint,
        config=Config(signature_version="v4"),
    )
    if s3.meta.endpoint_url != s3_endpoint or sts.meta.endpoint_url != sts_endpoint:
        raise ValueError("AWS SDK did not retain the frozen regional endpoints")
    return s3, sts


def main() -> int:
    args = parse_args()
    s3, sts = build_aws_clients(args.region)
    receipt = prepare_input_urls(
        json.loads(args.packet.read_text()),
        s3=s3,
        sts=sts,
        account_id=args.account_id,
        region=args.region,
        manifest_bucket=args.manifest_bucket,
        manifest_key=args.manifest_key,
        manifest_output=args.manifest_output.absolute(),
        url_file=args.url_file.absolute(),
        host_gate_url_file=args.host_gate_url_file.absolute(),
        receipt_path=args.receipt.absolute(),
        expiry_seconds=args.expires_in,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
