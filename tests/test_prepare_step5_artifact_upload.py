from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "prepare_tinylora_step5_artifact_upload.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_step5_artifact_upload", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_destination_validation_rejects_traversal_and_bad_expiry():
    with pytest.raises(ValueError, match="safe relative"):
        MODULE.validate_destination("valid-bucket", "../key", 3600)
    with pytest.raises(ValueError, match="between 60"):
        MODULE.validate_destination("valid-bucket", "runs/a.tar", 10)


def test_private_url_file_is_0600_and_no_clobber(tmp_path: Path):
    target = tmp_path / "signed-url"
    MODULE.write_private(target, "https://example.invalid/private")
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.read_text() == "https://example.invalid/private\n"
    with pytest.raises(FileExistsError, match="refusing to replace"):
        MODULE.write_private(target, "https://example.invalid/other")
