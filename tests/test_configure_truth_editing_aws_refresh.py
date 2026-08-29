from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "configure_truth_editing_aws_refresh.py"
)


def _credential(*, suffix: str, expires_in_seconds: int = 3600) -> dict[str, object]:
    return {
        "Version": 1,
        "AccessKeyId": f"AKIAEXAMPLE{suffix:0>8}",
        "SecretAccessKey": f"secret-value-{suffix}-that-is-long-enough",
        "SessionToken": f"session-token-{suffix}",
        "Expiration": (
            datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)
        ).isoformat(),
    }


def _run(
    root: Path,
    mode: str,
    *,
    environment: dict[str, str] | None = None,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), mode, "--root", str(root)],
        env=environment,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )


def test_initialize_creates_private_refresh_profile_and_emits_exact_identity(
    tmp_path: Path,
) -> None:
    credential = _credential(suffix="1")
    environment = {
        **os.environ,
        "AWS_ACCESS_KEY_ID": str(credential["AccessKeyId"]),
        "AWS_SECRET_ACCESS_KEY": str(credential["SecretAccessKey"]),
        "AWS_SESSION_TOKEN": str(credential["SessionToken"]),
        "AWS_CREDENTIAL_EXPIRATION": str(credential["Expiration"]),
    }
    root = tmp_path / "aws-refresh"

    initialized = _run(root, "initialize", environment=environment)
    emitted = _run(root, "emit")

    assert initialized.returncode == 0
    assert emitted.returncode == 0
    assert json.loads(emitted.stdout) == credential
    assert "[profile truth-editing-refresh]" in (root / "config").read_text()
    assert (root / "credentials").read_bytes() == b""
    for path in (root, root / "config", root / "credentials", root / "process-credentials.json"):
        permissions = stat.S_IMODE(path.stat().st_mode)
        assert permissions & 0o077 == 0


def test_replace_from_stdin_atomically_changes_emitted_identity(tmp_path: Path) -> None:
    root = tmp_path / "aws-refresh"
    first = _credential(suffix="1")
    second = _credential(suffix="2")
    environment = {
        **os.environ,
        "AWS_ACCESS_KEY_ID": str(first["AccessKeyId"]),
        "AWS_SECRET_ACCESS_KEY": str(first["SecretAccessKey"]),
        "AWS_SESSION_TOKEN": str(first["SessionToken"]),
        "AWS_CREDENTIAL_EXPIRATION": str(first["Expiration"]),
    }
    assert _run(root, "initialize", environment=environment).returncode == 0

    replaced = _run(root, "replace-from-stdin", stdin=json.dumps(second))
    emitted = _run(root, "emit")

    assert replaced.returncode == 0
    assert emitted.returncode == 0
    assert json.loads(emitted.stdout) == second
    assert list(root.glob(".process-credentials.json.*")) == []


def test_refresh_helper_rejects_malformed_and_expired_credentials(tmp_path: Path) -> None:
    root = tmp_path / "aws-refresh"
    malformed = _run(root, "replace-from-stdin", stdin='{"Version": 1}')
    expired = _run(
        root,
        "replace-from-stdin",
        stdin=json.dumps(_credential(suffix="3", expires_in_seconds=-1)),
    )

    assert malformed.returncode != 0
    assert expired.returncode != 0
    assert not (root / "process-credentials.json").exists()
