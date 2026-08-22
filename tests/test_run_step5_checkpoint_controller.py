from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_step5_checkpoint_controller.py"
SPEC = importlib.util.spec_from_file_location("run_step5_checkpoint_controller", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ready_file_is_atomic_private_and_no_clobber(tmp_path: Path):
    path = tmp_path / "ready.json"
    payload = {"format": "ready", "pid": 123}
    MODULE.write_ready_file(path, payload)
    assert json.loads(path.read_text()) == payload
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError, match="already exists"):
        MODULE.write_ready_file(path, payload)
