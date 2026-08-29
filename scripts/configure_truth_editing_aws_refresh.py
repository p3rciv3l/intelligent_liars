#!/usr/bin/env python3
"""Install or refresh the private AWS credential-process used by a Vast run."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROFILE = "truth-editing-refresh"


def _credentials(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "Version",
        "AccessKeyId",
        "SecretAccessKey",
        "SessionToken",
        "Expiration",
    }:
        raise ValueError("AWS process credential fields differ")
    if value["Version"] != 1:
        raise ValueError("AWS process credential version differs")
    for field, minimum, maximum in (
        ("AccessKeyId", 16, 128),
        ("SecretAccessKey", 16, 256),
        ("SessionToken", 8, 16384),
    ):
        item = value[field]
        if (
            not isinstance(item, str)
            or not minimum <= len(item) <= maximum
            or item != item.strip()
            or any(character in item for character in ("\x00", "\r", "\n"))
        ):
            raise ValueError(f"AWS process credential {field} is malformed")
    expiration = value["Expiration"]
    if not isinstance(expiration, str):
        raise ValueError("AWS process credential Expiration is malformed")
    try:
        parsed = datetime.fromisoformat(expiration.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("AWS process credential Expiration is malformed") from error
    if parsed.tzinfo is None or parsed.astimezone(timezone.utc) <= datetime.now(timezone.utc):
        raise ValueError("AWS process credentials are already expired")
    return dict(value)


def _atomic_private(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink() or not stat.S_ISDIR(path.parent.stat().st_mode):
        raise ValueError("AWS refresh root must be a regular directory")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
        path.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _payload(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _process_path(root: Path) -> Path:
    return root / "process-credentials.json"


def _write_config(root: Path) -> None:
    command = shlex.join(
        (
            sys.executable,
            str(Path(__file__).resolve()),
            "emit",
            "--root",
            str(root.resolve()),
        )
    )
    config = (
        f"[profile {PROFILE}]\n"
        f"credential_process = {command}\n"
        "region = us-east-1\n"
    ).encode()
    _atomic_private(root / "config", config)
    _atomic_private(root / "credentials", b"")


def _from_environment() -> dict[str, Any]:
    return _credentials(
        {
            "Version": 1,
            "AccessKeyId": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "SecretAccessKey": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            "SessionToken": os.environ.get("AWS_SESSION_TOKEN", ""),
            "Expiration": os.environ.get("AWS_CREDENTIAL_EXPIRATION", ""),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "initialize", "replace-from-stdin", "emit")
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    path = _process_path(root)
    if args.mode == "prepare":
        _write_config(root)
        return 0
    if args.mode == "initialize":
        _atomic_private(path, _payload(_from_environment()))
        _write_config(root)
        return 0
    if args.mode == "replace-from-stdin":
        try:
            value = json.load(sys.stdin)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SystemExit("AWS process credential input is invalid JSON") from error
        _atomic_private(path, _payload(_credentials(value)))
        return 0
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise SystemExit("AWS process credential file is missing or unsafe")
    value = _credentials(json.loads(path.read_text(encoding="utf-8")))
    sys.stdout.buffer.write(_payload(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
