from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_vast_pilot_instance.py"
SPEC = importlib.util.spec_from_file_location("run_vast_pilot_instance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cleanup_stops_after_workload_when_artifacts_are_not_verified():
    assert MODULE.cleanup_action(workload_started=True, artifacts_verified=False) == "stop"


def test_cleanup_destroys_after_artifacts_are_verified():
    assert MODULE.cleanup_action(workload_started=True, artifacts_verified=True) == "destroy"


def test_cleanup_destroys_when_no_workload_started():
    assert MODULE.cleanup_action(workload_started=False, artifacts_verified=False) == "destroy"


def test_verified_artifact_hashes_requires_complete_nonempty_set(tmp_path: Path):
    result_dir = tmp_path / "rank_2"
    result_dir.mkdir()
    (result_dir / "result.json").write_text("{}")
    (result_dir / "pilot_state.pt").write_bytes(b"state")
    with pytest.raises(FileNotFoundError, match="Incomplete fetched artifact set"):
        MODULE.verified_artifact_hashes(tmp_path, 2)


def test_verified_artifact_hashes_returns_all_required_files(tmp_path: Path):
    result_dir = tmp_path / "rank_3"
    result_dir.mkdir()
    for name in ("result.json", "pilot_state.pt", "tinylora_rank3_basis.pt"):
        (result_dir / name).write_bytes(name.encode())
    hashes = MODULE.verified_artifact_hashes(tmp_path, 3)
    assert set(hashes) == {
        "rank_3/result.json",
        "rank_3/pilot_state.pt",
        "rank_3/tinylora_rank3_basis.pt",
    }


def test_artifact_destination_must_not_contain_a_previous_result(tmp_path: Path):
    (tmp_path / "rank_1").mkdir()
    with pytest.raises(FileExistsError, match="existing result"):
        MODULE.require_empty_artifact_destination(tmp_path, 1)
