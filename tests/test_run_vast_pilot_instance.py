from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import tarfile
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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact_inventory(rank: int) -> set[str]:
    return {
        f"rank_{rank}/result.json",
        f"rank_{rank}/pilot_state.pt",
        f"rank_{rank}/tinylora_rank{rank}_basis.pt",
    }


def test_verified_artifact_hashes_requires_complete_nonempty_set(tmp_path: Path):
    result_dir = tmp_path / "rank_2"
    result_dir.mkdir()
    (result_dir / "result.json").write_text("{}")
    (result_dir / "pilot_state.pt").write_bytes(b"state")
    with pytest.raises(FileNotFoundError, match="Incomplete fetched artifact set"):
        MODULE.verified_artifact_hashes(tmp_path, 2, _artifact_inventory(2))


def test_verified_artifact_hashes_returns_all_required_files(tmp_path: Path):
    result_dir = tmp_path / "rank_3"
    result_dir.mkdir()
    (result_dir / "result.json").write_bytes(b"{}")
    (result_dir / "pilot_state.pt").write_bytes(b"state")
    (result_dir / "tinylora_rank3_basis.pt").write_bytes(b"basis")
    hashes = MODULE.verified_artifact_hashes(tmp_path, 3, _artifact_inventory(3))
    assert set(hashes) == {
        "rank_3/result.json",
        "rank_3/pilot_state.pt",
        "rank_3/tinylora_rank3_basis.pt",
    }


def test_verified_artifact_hashes_rejects_unexpected_files(tmp_path: Path):
    result_dir = tmp_path / "rank_1"
    result_dir.mkdir()
    (result_dir / "result.json").write_bytes(b"{}")
    (result_dir / "pilot_state.pt").write_bytes(b"state")
    (result_dir / "tinylora_rank1_basis.pt").write_bytes(b"basis")
    (result_dir / "surprise.txt").write_bytes(b"not controller-approved")
    with pytest.raises(ValueError, match="inventory mismatch"):
        MODULE.verified_artifact_hashes(tmp_path, 1, _artifact_inventory(1))


def test_verified_artifact_hashes_rejects_symlink(tmp_path: Path):
    result_dir = tmp_path / "rank_1"
    result_dir.mkdir()
    (result_dir / "result.json").write_bytes(b"{}")
    (result_dir / "pilot_state.pt").write_bytes(b"state")
    outside = tmp_path / "outside"
    outside.write_bytes(b"basis")
    (result_dir / "tinylora_rank1_basis.pt").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        MODULE.verified_artifact_hashes(tmp_path, 1, _artifact_inventory(1))


def _write_archive(tmp_path: Path, members: dict[str, bytes]) -> tuple[Path, Path]:
    archive = tmp_path / "workload.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))
    manifest = tmp_path / "workload.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "tinylora_workload_archive_v1",
                "archive_sha256": MODULE.sha256_file(archive),
                "files": {name: _sha256(content) for name, content in members.items()},
            }
        )
    )
    return archive, manifest


def test_validate_workload_archive_accepts_exact_allowlist(tmp_path: Path):
    archive, manifest = _write_archive(tmp_path, {"scripts/run.py": b"print('ok')\n"})
    validated = MODULE.validate_workload_archive(archive, manifest)
    assert validated["files"] == {"scripts/run.py": _sha256(b"print('ok')\n")}


@pytest.mark.parametrize("name", [".env", ".git/config", "safe/../escape", "/absolute"])
def test_validate_workload_archive_rejects_sensitive_or_unsafe_paths(
    tmp_path: Path, name: str
):
    archive, manifest = _write_archive(tmp_path, {name: b"secret"})
    with pytest.raises(ValueError, match="unsafe|sensitive"):
        MODULE.validate_workload_archive(archive, manifest)


def test_validate_workload_archive_rejects_unlisted_member(tmp_path: Path):
    archive, manifest = _write_archive(
        tmp_path, {"scripts/run.py": b"ok", "unlisted.txt": b"extra"}
    )
    payload = json.loads(manifest.read_text())
    del payload["files"]["unlisted.txt"]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="allowlist mismatch"):
        MODULE.validate_workload_archive(archive, manifest)


def test_validate_workload_archive_rejects_symlink_member(tmp_path: Path):
    archive = tmp_path / "workload.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("scripts/run.py")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../.env"
        bundle.addfile(info)
    manifest = tmp_path / "workload.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "tinylora_workload_archive_v1",
                "archive_sha256": MODULE.sha256_file(archive),
                "files": {"scripts/run.py": _sha256(b"")},
            }
        )
    )
    with pytest.raises(ValueError, match="regular files"):
        MODULE.validate_workload_archive(archive, manifest)


def test_validate_workload_archive_rejects_directory_input(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="regular .tar.gz archive"):
        MODULE.validate_workload_archive(tmp_path, manifest)


def test_load_artifact_inventory_freezes_exact_safe_paths(tmp_path: Path):
    inventory = tmp_path / "artifacts.json"
    inventory.write_text(
        json.dumps(
            {
                "format": "tinylora_pilot_artifact_inventory_v1",
                "files": sorted(_artifact_inventory(2)),
            }
        )
    )
    assert MODULE.load_artifact_inventory(inventory, 2) == _artifact_inventory(2)


def test_load_artifact_inventory_rejects_paths_outside_rank(tmp_path: Path):
    inventory = tmp_path / "artifacts.json"
    inventory.write_text(
        json.dumps(
            {
                "format": "tinylora_pilot_artifact_inventory_v1",
                "files": ["rank_2/result.json", "../.env"],
            }
        )
    )
    with pytest.raises(ValueError, match="unsafe"):
        MODULE.load_artifact_inventory(inventory, 2)


def test_artifact_destination_must_not_contain_a_previous_result(tmp_path: Path):
    (tmp_path / "rank_1").mkdir()
    with pytest.raises(FileExistsError, match="existing result"):
        MODULE.require_empty_artifact_destination(tmp_path, 1)


def test_artifact_destination_must_not_be_a_symlink(tmp_path: Path):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual)
    with pytest.raises(ValueError, match="symlink"):
        MODULE.require_empty_artifact_destination(link, 1)
