#!/usr/bin/env python3
"""Verify and presign frozen Step 5 inputs on the trusted controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def main() -> int:
    args = parse_args()
    import boto3

    session = boto3.Session(region_name=args.region)
    receipt = prepare_input_urls(
        json.loads(args.packet.read_text()),
        s3=session.client("s3"),
        sts=session.client("sts"),
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
