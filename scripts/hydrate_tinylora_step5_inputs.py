#!/usr/bin/env python3
"""Hydrate immutable Step 5 inputs from a credentialless HTTPS URL manifest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from intelligent_liars.step5_input_hydration import fetch_https, hydrate_all, https_origin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url-manifest-url-env",
        default="CANARY_INPUT_URL_MANIFEST_URL",
        help="Environment variable holding the signed HTTPS URL (never put in argv).",
    )
    parser.add_argument("--inputs-dir", type=Path, default=Path("/workspace/inputs"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/workspace/cache/huggingface"))
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    url = os.environ.get(args.url_manifest_url_env)
    if not url:
        raise ValueError(f"Missing URL environment variable: {args.url_manifest_url_env}")
    https_origin(url)
    with tempfile.TemporaryDirectory(prefix="step5-url-manifest-") as directory:
        path = Path(directory) / "urls.json"
        fetch_https(url, path)
        payload = json.loads(path.read_text())
    receipt = hydrate_all(
        payload,
        inputs_dir=args.inputs_dir.resolve(),
        cache_dir=args.cache_dir.resolve(),
        receipt_path=args.receipt.resolve(),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
