#!/usr/bin/env python3
"""Hydrate immutable Step 5 inputs from a credentialless HTTPS URL manifest."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path

from intelligent_liars.step5_input_hydration import (
    EXPECTED_IDENTITY_FIELDS,
    fetch_https,
    hydrate_all,
    https_origin,
)


def read_private_url(path: Path) -> str:
    absolute = path.absolute()
    parts = absolute.parts
    directory_fd = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY)
    file_fd: int | None = None
    try:
        for part in parts[1:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError("URL manifest URL file must be regular and mode 0600 or stricter")
        if metadata.st_size > 16384:
            raise ValueError("URL manifest URL file is unexpectedly large")
        with os.fdopen(file_fd, "r", encoding="utf-8", closefd=True) as stream:
            file_fd = None
            lines = stream.read(16385).splitlines()
    except OSError as error:
        raise ValueError("URL manifest URL file may not traverse symlinks") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)
    if len(lines) != 1 or not lines[0].strip():
        raise ValueError("URL manifest URL file must contain exactly one URL")
    return lines[0].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url-manifest-url-env",
        default="CANARY_INPUT_URL_MANIFEST_URL",
        help="Environment variable holding the signed HTTPS URL (never put in argv).",
    )
    parser.add_argument(
        "--url-manifest-url-file",
        type=Path,
        help="Mode-0600 file containing the signed HTTPS URL (preferred on workers).",
    )
    parser.add_argument("--inputs-dir", type=Path, default=Path("/workspace/inputs"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/workspace/cache/huggingface"))
    parser.add_argument("--receipt", type=Path, required=True)
    for field in EXPECTED_IDENTITY_FIELDS:
        parser.add_argument(f"--expected-{field.replace('_', '-')}", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.url_manifest_url_file is not None:
        url = read_private_url(args.url_manifest_url_file)
    else:
        url = os.environ.get(args.url_manifest_url_env, "").strip()
    if not url:
        raise ValueError("Missing hydration URL manifest URL")
    https_origin(url)
    with tempfile.TemporaryDirectory(prefix="step5-url-manifest-") as directory:
        path = Path(directory) / "urls.json"
        fetch_https(url, path)
        payload = json.loads(path.read_text())
    receipt = hydrate_all(
        payload,
        inputs_dir=args.inputs_dir.absolute(),
        cache_dir=args.cache_dir.absolute(),
        receipt_path=args.receipt.absolute(),
        expected_identities={
            field: getattr(args, f"expected_{field}")
            for field in EXPECTED_IDENTITY_FIELDS
        },
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
