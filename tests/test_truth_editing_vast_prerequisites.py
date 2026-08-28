from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from intelligent_liars.truth_editing_vast_prerequisites import (
    EphemeralWorkloadSecret,
    resolve_aws_workload_credentials,
    resolve_aws_cli_profile_credentials,
    FROZEN_BASE_IMAGE,
    FROZEN_MODEL_ID,
    FROZEN_MODEL_REVISION,
    FROZEN_RUNTIME_ID,
    JobConfig,
    Offer,
    VastPrerequisiteError,
    build_bundle,
    execute_lifecycle,
    lifecycle_plan,
    _remote_pack_command,
    _remote_input_bundle_verification_command,
    _secret_stdin_wrapper,
)
from intelligent_liars.truth_editing_vast_prerequisites import (
    _validate_created_instance,
)


def _raw_config(paths: list[str]) -> dict[str, object]:
    return {
        "format": "truth_editing_vast_prerequisite_job_v1",
        "image": FROZEN_BASE_IMAGE,
        "runtime_id": FROZEN_RUNTIME_ID,
        "model": {"repository": FROZEN_MODEL_ID, "revision": FROZEN_MODEL_REVISION},
        "resources": {
            "disk_gib": 120,
            "minimum_gpu_vram_gib": 24,
            "maximum_elapsed_seconds": 3600,
            "maximum_cost_usd": 4.0,
            "maximum_download_gib": 50.0,
            "maximum_upload_gib": 5.0,
        },
        "paths": {
            "remote_workdir": "/workspace/intelligent_liars",
            "remote_output_dir": "/workspace/outputs",
        },
        "commands": {
            "bootstrap": ["bash", "bootstrap.sh"],
            "workload": ["bash", "worker.sh"],
        },
        "expected_outputs": ["receipt.json"],
        "bundle_paths": paths,
    }


def _config(paths: list[str] | None = None) -> JobConfig:
    return JobConfig.from_mapping(_raw_config(paths or ["worker.sh"]))


def _offer(price: float = 0.7) -> Offer:
    return Offer.from_mapping(
        {
            "id": 123,
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "gpu_ram": 24576,
            "dph_base": price,
            "dph_total": price,
            "storage_cost": 0.0,
            "inet_down_cost": 0.0,
            "inet_up_cost": 0.0,
        }
    )


def test_created_instance_accepts_requested_disk_price_without_double_counting() -> None:
    approved = Offer.from_mapping(
        {
            "id": 33869598,
            "gpu_name": "RTX 3090",
            "num_gpus": 1,
            "gpu_ram": 24576,
            "dph_base": 0.12,
            "dph_total": 0.1213888888888889,
            "storage_cost": 0.2,
            "inet_down_cost": 0.00130208,
            "inet_up_cost": 0.00130208,
        }
    )
    raw_created = {
        "gpu_name": "RTX 3090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "dph_base": 0.12,
        "dph_total": 0.15333333333333332,
        "instance": {"diskHour": 0.03333333333333333},
        "inet_down_cost": 0.00130208,
        "inet_up_cost": 0.00130208,
    }
    raw = _raw_config(["worker.sh"])
    raw["resources"]["maximum_elapsed_seconds"] = 14400  # type: ignore[index]
    raw["resources"]["maximum_cost_usd"] = 0.7  # type: ignore[index]
    raw["resources"]["maximum_download_gib"] = 25.0  # type: ignore[index]
    raw["resources"]["maximum_upload_gib"] = 1.0  # type: ignore[index]
    config = JobConfig.from_mapping(raw)

    _validate_created_instance(raw_created, approved, config)


def test_preservation_offer_fixture_accounts_for_requested_disk_and_network() -> None:
    root = Path(__file__).parents[1]
    config = JobConfig.from_mapping(
        json.loads(
            (root / "configs/truth_editing_preservation_vast_job_v2_r4.json").read_text()
        )
    )
    offer = Offer.from_mapping(
        json.loads(
            (root / "artifacts/truth-editing/vast/offer-33869598.json").read_text()
        )
    )

    plan = lifecycle_plan(
        vastai="vastai",
        config=config,
        offer=offer,
        bundle=Path("bundle.tgz"),
        fetch_dir=Path("fetch"),
    )

    assert plan["offer"]["base_hourly_price_usd"] == pytest.approx(0.12)
    assert plan["offer"]["disk_hourly_price_usd"] == pytest.approx(1 / 30)
    assert plan["offer"]["hourly_price_usd"] == pytest.approx(0.15333333333333332)
    assert plan["offer"]["projected_max_cost_usd"] == pytest.approx(
        0.6471874133333333
    )


def _output_archive(files: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name, body in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    return payload.getvalue()


def _remote_archive_identity(payload: bytes) -> str:
    return json.dumps(
        {
            "format": "truth_editing_vast_output_archive_v1",
            "archive_sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    )


def test_frozen_identity_and_strict_schema_fail_closed() -> None:
    raw = _raw_config(["worker.sh"])
    raw["extra"] = True
    with pytest.raises(VastPrerequisiteError, match="fields or format"):
        JobConfig.from_mapping(raw)
    raw = _raw_config(["worker.sh"])
    raw["image"] = "latest"
    with pytest.raises(VastPrerequisiteError, match="digest-pinned"):
        JobConfig.from_mapping(raw)
    raw = _raw_config(["worker.sh"])
    raw["model"]["revision"] = "main"  # type: ignore[index]
    with pytest.raises(VastPrerequisiteError, match="revision changed"):
        JobConfig.from_mapping(raw)


def test_cost_is_strictly_below_fifteen_and_offer_is_bounded() -> None:
    raw = _raw_config(["worker.sh"])
    raw["resources"]["maximum_cost_usd"] = 15.0  # type: ignore[index]
    with pytest.raises(VastPrerequisiteError, match="strictly below"):
        JobConfig.from_mapping(raw)
    with pytest.raises(VastPrerequisiteError, match="approved bounded cost"):
        lifecycle_plan(
            vastai="vastai",
            config=_config(),
            offer=_offer(5.0),
            bundle=Path("bundle.tgz"),
            fetch_dir=Path("fetch"),
        )
    network_offer = Offer.from_mapping(
        {
            "id": 123,
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "gpu_ram": 24576,
            "dph_base": 0.1,
            "dph_total": 0.1,
            "storage_cost": 0.0,
            "inet_down_cost": 0.08,
            "inet_up_cost": 0.0,
        }
    )
    with pytest.raises(VastPrerequisiteError, match="approved bounded cost"):
        lifecycle_plan(
            vastai="vastai",
            config=_config(),
            offer=network_offer,
            bundle=Path("bundle.tgz"),
            fetch_dir=Path("fetch"),
        )


def test_bundle_contains_only_allowlisted_regular_files_and_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "worker.sh").write_text("echo safe\n")
    (repo / "unrelated.txt").write_text("do not ship")
    bundle = tmp_path / "bundle.tgz"
    manifest = build_bundle(repo, _config(), bundle)
    second = tmp_path / "bundle-2.tgz"
    second_manifest = build_bundle(repo, _config(), second)
    with tarfile.open(bundle) as archive:
        names = archive.getnames()
    assert names == ["worker.sh", "bundle-manifest.json"]
    assert manifest["files"][0]["path"] == "worker.sh"
    assert "unrelated.txt" not in names
    assert bundle.read_bytes() == second.read_bytes()
    assert manifest["archive_sha256"] == second_manifest["archive_sha256"]


def test_bundle_rejects_symlinks_traversal_and_secret_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("private")
    (repo / "link").symlink_to(outside)
    with pytest.raises(VastPrerequisiteError, match="not a regular"):
        build_bundle(repo, _config(["link"]), tmp_path / "bundle.tgz")
    with pytest.raises(VastPrerequisiteError, match="unsafe bundle"):
        JobConfig.from_mapping(_raw_config(["../outside"]))
    with pytest.raises(VastPrerequisiteError, match="secret-like"):
        JobConfig.from_mapping(_raw_config([".ssh/id_rsa"]))


def test_plan_has_timeout_fetch_and_destroy() -> None:
    plan = lifecycle_plan(
        vastai="vastai",
        config=_config(),
        offer=_offer(),
        bundle=Path("bundle.tgz"),
        fetch_dir=Path("fetch"),
    )
    assert "timeout --signal=TERM --kill-after=60 3600s" in plan["remote_command"]
    assert plan["destroy_command_template"] == [
        "vastai", "destroy", "instance", "<INSTANCE_ID>", "--raw"
    ]
    assert plan["offer"]["projected_max_cost_usd"] == pytest.approx(0.7)
    assert plan["remote_output_archive"] == "/workspace/truth-prerequisite-outputs.tar.gz"
    assert plan["fetch_command_template"][-2] == (
        "C.<INSTANCE_ID>:/workspace/truth-prerequisite-outputs.tar.gz"
    )


@pytest.mark.parametrize("missing_name", ["first.json", "second.json"])
def test_remote_pack_command_fails_closed_when_any_expected_output_is_missing(
    tmp_path: Path, missing_name: str
) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    for name in {"first.json", "second.json"} - {missing_name}:
        (output_dir / name).write_text("{}\n")
    archive_path = tmp_path / "outputs.tar.gz"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_stat = fake_bin / "stat"
    fake_stat.write_text("#!/bin/sh\nwc -c < \"$3\"\n")
    fake_stat.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            "-lc",
            "export COPYFILE_DISABLE=1 PATH="
            + shlex.quote(str(fake_bin))
            + ":$PATH; "
            + _remote_pack_command(
                output_dir=str(output_dir),
                archive_path=str(archive_path),
                expected_outputs=("first.json", "second.json"),
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 73
    assert not archive_path.exists()
    assert missing_name in result.stderr


def test_remote_pack_command_packs_complete_expected_output_inventory(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "first.json").write_text("{\"first\":true}\n")
    (output_dir / "second.json").write_text("{\"second\":true}\n")
    archive_path = tmp_path / "outputs.tar.gz"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_stat = fake_bin / "stat"
    fake_stat.write_text("#!/bin/sh\nwc -c < \"$3\"\n")
    fake_stat.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            "-lc",
            "export COPYFILE_DISABLE=1 PATH="
            + shlex.quote(str(fake_bin))
            + ":$PATH; "
            + _remote_pack_command(
                output_dir=str(output_dir),
                archive_path=str(archive_path),
                expected_outputs=("first.json", "second.json"),
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    identity = json.loads(result.stdout)
    assert identity["size_bytes"] == archive_path.stat().st_size
    with tarfile.open(archive_path, "r:gz") as archive:
        assert {name.removeprefix("./") for name in archive.getnames()} == {
            ".",
            "first.json",
            "second.json",
        }


def test_execute_fetches_one_verified_archive_and_atomically_publishes_outputs(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    payload = _output_archive({"receipt.json": b'{"ok":true}\n', "nested/data.bin": b"abc"})
    approved = {
        "id": 123,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "dph_base": 0.7,
        "dph_total": 0.7,
        "storage_cost": 0.0,
        "diskHour": 0.0,
        "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
    }

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        if command[1:3] == ["search", "offers"]:
            return SimpleNamespace(stdout=json.dumps([approved]), returncode=0)
        if command[1:3] == ["create", "instance"]:
            return SimpleNamespace(stdout='{"id": 1234}', returncode=0)
        if command[1:3] == ["show", "instance"]:
            return SimpleNamespace(stdout=json.dumps({**approved, "actual_status": "running"}), returncode=0)
        if command[1:] == ["ssh-url", "1234"]:
            return SimpleNamespace(stdout="ssh://root@example.test:22\n", returncode=0)
        if command[0] == "ssh" and "truth_editing_vast_output_archive_v1" in command[-1]:
            return SimpleNamespace(stdout=_remote_archive_identity(payload), returncode=0)
        if command[1:2] == ["copy"] and command[-2].endswith("truth-prerequisite-outputs.tar.gz"):
            Path(command[-1].removeprefix("local:")).write_bytes(payload)
        return SimpleNamespace(stdout="", returncode=0)

    config = _config()
    fetch_dir = tmp_path / "published"
    plan = lifecycle_plan(
        vastai="vastai", config=config, offer=_offer(), bundle=Path("b"), fetch_dir=fetch_dir
    )
    receipt = execute_lifecycle(
        plan=plan,
        config=config,
        metadata_path=tmp_path / "receipt.json",
        run_command=run,
        sleeper=lambda _seconds: None,
    )

    assert (fetch_dir / "receipt.json").read_bytes() == b'{"ok":true}\n'
    assert (fetch_dir / "nested/data.bin").read_bytes() == b"abc"
    fetches = [command for command in calls if command[1:2] == ["copy"] and "C.1234:" in command[-2]]
    assert len(fetches) == 1
    assert receipt["artifact_archive"] == {
        "format": "truth_editing_vast_output_archive_v1",
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "expected_outputs": ["receipt.json"],
        "published_directory": str(fetch_dir),
    }


def test_later_fetch_failure_receipt_keeps_bounded_redacted_success_diagnostics(
    tmp_path: Path,
) -> None:
    secret_value = "sk-or-success-secret-that-must-not-persist"
    payload = _output_archive({"other.json": b"{}\n"})
    approved = {
        "id": 123,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "dph_base": 0.7,
        "dph_total": 0.7,
        "storage_cost": 0.0,
        "diskHour": 0.0,
        "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
    }

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        if command[1:3] == ["search", "offers"]:
            return SimpleNamespace(stdout=json.dumps([approved]), returncode=0)
        if command[1:3] == ["create", "instance"]:
            return SimpleNamespace(stdout='{"id": 9753}', returncode=0)
        if command[1:3] == ["show", "instance"]:
            return SimpleNamespace(
                stdout=json.dumps({**approved, "actual_status": "running"}),
                returncode=0,
            )
        if command[1:] == ["ssh-url", "9753"]:
            return SimpleNamespace(stdout="ssh://root@example.test:22\n", returncode=0)
        if command[0] == "ssh" and "truth_editing_vast_output_archive_v1" in command[-1]:
            return SimpleNamespace(
                stdout=_remote_archive_identity(payload), stderr="", returncode=0
            )
        if command[0] == "ssh":
            return SimpleNamespace(
                stdout="X" * 5000 + secret_value,
                stderr="successful workload stderr",
                returncode=0,
            )
        if command[1:2] == ["copy"] and command[-2].endswith(
            "truth-prerequisite-outputs.tar.gz"
        ):
            Path(command[-1].removeprefix("local:")).write_bytes(payload)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    metadata_path = tmp_path / "receipt.json"
    config = _config()
    with pytest.raises(VastPrerequisiteError, match="fetched artifacts are incomplete"):
        execute_lifecycle(
            plan=lifecycle_plan(
                vastai="vastai",
                config=config,
                offer=_offer(),
                bundle=Path("b"),
                fetch_dir=tmp_path / "fetch",
            ),
            config=config,
            metadata_path=metadata_path,
            run_command=run,
            sleeper=lambda _seconds: None,
            workload_secret=EphemeralWorkloadSecret.openrouter(secret_value),
        )

    receipt_text = metadata_path.read_text()
    assert secret_value not in receipt_text
    workload_event = next(
        event
        for event in json.loads(receipt_text)["events"]
        if event["event"] == "workload_finished"
    )
    assert workload_event["exit_code"] == 0
    assert workload_event["stdout_truncated"] is True
    assert len(workload_event["stdout_tail"]) <= 4000
    assert workload_event["stdout_tail"].endswith("[REDACTED]")
    assert workload_event["stderr_tail"] == "successful workload stderr"


@pytest.mark.parametrize(
    "failure",
    ["oversize", "wrong_hash", "wrong_size", "traversal", "symlink", "missing"],
)
def test_execute_rejects_untrusted_archive_without_publishing(
    tmp_path: Path, failure: str
) -> None:
    name = "../escape" if failure == "traversal" else "receipt.json"
    if failure == "missing":
        name = "other.json"
    if failure == "symlink":
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            info = tarfile.TarInfo("receipt.json")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            archive.addfile(info)
        payload = stream.getvalue()
    else:
        payload = _output_archive({name: b"data"})
    advertised = {
        "format": "truth_editing_vast_output_archive_v1",
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    if failure == "oversize":
        advertised["size_bytes"] = 6 * 1024**3
    if failure == "wrong_hash":
        advertised["archive_sha256"] = "0" * 64
    if failure == "wrong_size":
        advertised["size_bytes"] = len(payload) + 1
    approved = {
        "id": 123,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "dph_base": 0.7,
        "dph_total": 0.7,
        "storage_cost": 0.0,
        "diskHour": 0.0,
        "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
    }

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        if command[1:3] == ["search", "offers"]:
            return SimpleNamespace(stdout=json.dumps([approved]), returncode=0)
        if command[1:3] == ["create", "instance"]:
            return SimpleNamespace(stdout='{"id": 4321}', returncode=0)
        if command[1:3] == ["show", "instance"]:
            return SimpleNamespace(stdout=json.dumps({**approved, "actual_status": "running"}), returncode=0)
        if command[1:] == ["ssh-url", "4321"]:
            return SimpleNamespace(stdout="ssh://root@example.test:22\n", returncode=0)
        if command[0] == "ssh" and "truth_editing_vast_output_archive_v1" in command[-1]:
            return SimpleNamespace(stdout=json.dumps(advertised), returncode=0)
        if command[1:2] == ["copy"] and command[-2].endswith("truth-prerequisite-outputs.tar.gz"):
            Path(command[-1].removeprefix("local:")).write_bytes(payload)
        return SimpleNamespace(stdout="", returncode=0)

    fetch_dir = tmp_path / "published"
    with pytest.raises(VastPrerequisiteError):
        execute_lifecycle(
            plan=lifecycle_plan(
                vastai="vastai",
                config=_config(),
                offer=_offer(),
                bundle=Path("b"),
                fetch_dir=fetch_dir,
            ),
            config=_config(),
            metadata_path=tmp_path / "receipt.json",
            run_command=run,
            sleeper=lambda _seconds: None,
        )
    assert not fetch_dir.exists()
    assert not (tmp_path / "escape").exists()
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["artifact_archive"] is None
    assert receipt["destroyed"] is True


def test_execute_destroys_after_success_and_writes_receipt(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    payload = _output_archive({"receipt.json": b"{}\n"})

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        if command[1:3] == ["search", "offers"]:
            return SimpleNamespace(stdout=json.dumps({"unused": True}), returncode=0)
        if command[1:3] == ["create", "instance"]:
            return SimpleNamespace(stdout='{"new_contract": 777}', returncode=0)
        if command[1:3] == ["show", "instance"]:
            return SimpleNamespace(
                stdout=json.dumps({**approved, "actual_status": "running"}),
                returncode=0,
            )
        if command[1:] == ["ssh-url", "777"]:
            return SimpleNamespace(stdout="ssh://root@example.test:2222\n", returncode=0)
        if command[0] == "ssh" and "truth_editing_vast_output_archive_v1" in command[-1]:
            return SimpleNamespace(stdout=_remote_archive_identity(payload), returncode=0)
        if command[1:2] == ["copy"] and command[-2].endswith("truth-prerequisite-outputs.tar.gz"):
            Path(command[-1].removeprefix("local:")).write_bytes(payload)
        return SimpleNamespace(stdout="", returncode=0)

    config = _config()
    plan = lifecycle_plan(
        vastai="vastai",
        config=config,
        offer=_offer(),
        bundle=Path("b"),
        fetch_dir=tmp_path / "f",
    )
    approved = {
        "id": 123, "gpu_name": "RTX 4090", "num_gpus": 1,
        "gpu_ram": 24576, "dph_base": 0.7, "dph_total": 0.7,
        "storage_cost": 0.0, "diskHour": 0.0, "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
    }
    def success_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[1:3] == ["search", "offers"]:
            calls.append(command)
            return SimpleNamespace(stdout=json.dumps([approved]), returncode=0)
        return run(command, **kwargs)
    receipt = execute_lifecycle(
        plan=plan, config=config, metadata_path=tmp_path / "receipt.json", run_command=success_run,
        sleeper=lambda _seconds: None,
    )
    assert receipt["destroyed"] is True
    assert calls[-1] == ["vastai", "destroy", "instance", "777", "--raw"]
    assert json.loads((tmp_path / "receipt.json").read_text())["instance_id"] == "777"


def test_execute_destroys_after_remote_failure(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        if command[1:3] == ["search", "offers"]:
            return SimpleNamespace(
                stdout=json.dumps([{"id": 123, "gpu_name": "RTX 4090", "num_gpus": 1, "gpu_ram": 24576, "dph_base": 0.7, "dph_total": 0.7, "storage_cost": 0.0, "inet_down_cost": 0.0, "inet_up_cost": 0.0}]),
                returncode=0,
            )
        if command[1:3] == ["create", "instance"]:
            return SimpleNamespace(stdout='{"id": 888}', returncode=0)
        if command[1:3] == ["show", "instance"]:
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "id": 123,
                        "gpu_name": "RTX 4090",
                        "num_gpus": 1,
                        "gpu_ram": 24576,
                        "dph_base": 0.7,
                        "dph_total": 0.7,
                        "diskHour": 0.0,
                        "inet_down_cost": 0.0,
                        "inet_up_cost": 0.0,
                        "actual_status": "running",
                    }
                ),
                returncode=0,
            )
        if command[1:] == ["ssh-url", "888"]:
            return SimpleNamespace(stdout="ssh://root@example.test:22\n", returncode=0)
        if command[0] == "ssh":
            return SimpleNamespace(stdout="bad", returncode=9)
        return SimpleNamespace(stdout="", returncode=0)

    config = _config()
    plan = lifecycle_plan(
        vastai="vastai",
        config=config,
        offer=_offer(),
        bundle=Path("b"),
        fetch_dir=tmp_path / "f",
    )
    with pytest.raises(VastPrerequisiteError, match="exited 9"):
        execute_lifecycle(
            plan=plan, config=config, metadata_path=tmp_path / "receipt.json", run_command=run,
            sleeper=lambda _seconds: None,
        )
    assert calls[-1] == ["vastai", "destroy", "instance", "888", "--raw"]
    assert json.loads((tmp_path / "receipt.json").read_text())["destroyed"] is True


@pytest.mark.parametrize(
    ("verification_exit", "diagnostic"),
    [
        (74, "uploaded bundle is missing"),
        (75, "uploaded bundle byte length differs"),
        (76, "uploaded bundle SHA-256 differs"),
    ],
)
def test_execute_rejects_scp_success_until_remote_bundle_identity_is_verified(
    tmp_path: Path,
    verification_exit: int,
    diagnostic: str,
) -> None:
    calls: list[list[str]] = []
    workload_started = False
    secret_value = "sk-or-upload-diagnostic-secret"
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"exact bundle bytes")
    identity = tmp_path / "id_ed25519"
    identity.write_text("test identity")
    approved = {
        "id": 123,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "dph_base": 0.7,
        "dph_total": 0.7,
        "storage_cost": 0.0,
        "diskHour": 0.0,
        "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
    }

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal workload_started
        calls.append(command)
        if command[1:3] == ["search", "offers"]:
            return SimpleNamespace(stdout=json.dumps([approved]), returncode=0)
        if command[1:3] == ["create", "instance"]:
            return SimpleNamespace(stdout='{"id": 4321}', returncode=0)
        if command[1:3] == ["show", "instance"]:
            return SimpleNamespace(
                stdout=json.dumps({**approved, "actual_status": "running"}),
                returncode=0,
            )
        if command[1:] == ["ssh-url", "4321"]:
            return SimpleNamespace(
                stdout="ssh://root@example.test:2222\n", returncode=0
            )
        if command[0] == "scp":
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if command[0] == "ssh" and "uploaded bundle" in command[-1]:
            return SimpleNamespace(
                stdout="",
                stderr="X" * 5000 + secret_value + " " + diagnostic,
                returncode=verification_exit,
            )
        if command[0] == "ssh" and command[-1] != "true":
            workload_started = True
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    metadata_path = tmp_path / "receipt.json"
    config = _config()
    with pytest.raises(VastPrerequisiteError, match="bundle verification failed"):
        execute_lifecycle(
            plan=lifecycle_plan(
                vastai="vastai",
                config=config,
                offer=_offer(),
                bundle=bundle,
                fetch_dir=tmp_path / "fetch",
                ssh_identity=identity,
            ),
            config=config,
            metadata_path=metadata_path,
            run_command=run,
            sleeper=lambda _seconds: None,
            workload_secret=EphemeralWorkloadSecret.openrouter(secret_value),
        )

    receipt_text = metadata_path.read_text()
    receipt = json.loads(receipt_text)
    assert secret_value not in receipt_text
    assert diagnostic in receipt["error"]
    assert len(receipt["error"]) < 4500
    assert "bundle_uploaded" not in [event["event"] for event in receipt["events"]]
    assert workload_started is False
    assert receipt["destroyed"] is True
    assert calls[-1] == ["vastai", "destroy", "instance", "4321", "--raw"]


def test_execute_records_scp_upload_only_after_exact_remote_identity_check(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"verified bundle bytes")
    expected_sha256 = hashlib.sha256(bundle.read_bytes()).hexdigest()
    identity = tmp_path / "id_ed25519"
    identity.write_text("test identity")
    output_payload = _output_archive({"receipt.json": b"{}\n"})
    approved = {
        "id": 123,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "dph_base": 0.7,
        "dph_total": 0.7,
        "storage_cost": 0.0,
        "diskHour": 0.0,
        "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
    }

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        if command[1:3] == ["search", "offers"]:
            return SimpleNamespace(stdout=json.dumps([approved]), returncode=0)
        if command[1:3] == ["create", "instance"]:
            return SimpleNamespace(stdout='{"id": 5432}', returncode=0)
        if command[1:3] == ["show", "instance"]:
            return SimpleNamespace(
                stdout=json.dumps({**approved, "actual_status": "running"}),
                returncode=0,
            )
        if command[1:] == ["ssh-url", "5432"]:
            return SimpleNamespace(
                stdout="ssh://root@example.test:2222\n", returncode=0
            )
        if command[0] == "ssh" and "uploaded bundle" in command[-1]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if command[0] == "ssh" and "truth_editing_vast_output_archive_v1" in command[-1]:
            return SimpleNamespace(
                stdout=_remote_archive_identity(output_payload),
                stderr="",
                returncode=0,
            )
        if command[0] == "scp" and command[-2].startswith("root@example.test:"):
            Path(command[-1]).write_bytes(output_payload)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    config = _config()
    receipt = execute_lifecycle(
        plan=lifecycle_plan(
            vastai="vastai",
            config=config,
            offer=_offer(),
            bundle=bundle,
            fetch_dir=tmp_path / "fetch",
            ssh_identity=identity,
        ),
        config=config,
        metadata_path=tmp_path / "receipt.json",
        run_command=run,
        sleeper=lambda _seconds: None,
    )

    verification_command = next(
        command[-1]
        for command in calls
        if command[0] == "ssh" and "uploaded bundle" in command[-1]
    )
    assert f'observed_bytes\" = {bundle.stat().st_size}' in verification_command
    assert expected_sha256 in verification_command
    upload_event = next(
        event for event in receipt["events"] if event["event"] == "bundle_uploaded"
    )
    assert upload_event == {
        "event": "bundle_uploaded",
        "size_bytes": bundle.stat().st_size,
        "archive_sha256": expected_sha256,
        "remote_identity_verified": True,
    }
    assert (tmp_path / "fetch" / "receipt.json").read_bytes() == b"{}\n"


def test_remote_input_bundle_identity_command_checks_real_bytes_and_hash(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle with spaces.tar.gz"
    bundle.write_bytes(b"known bundle")
    command = _remote_input_bundle_verification_command(
        archive_path=str(bundle),
        expected_size_bytes=bundle.stat().st_size,
        expected_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
    )

    verified = subprocess.run(
        ["bash", "-lc", command], check=False, capture_output=True, text=True
    )
    assert verified.returncode == 0, verified.stderr

    bundle.write_bytes(b"tampered bundle")
    rejected = subprocess.run(
        ["bash", "-lc", command], check=False, capture_output=True, text=True
    )
    assert rejected.returncode in {75, 76}
    assert "uploaded bundle" in rejected.stderr


def test_execute_injects_ephemeral_secret_only_over_stdin_and_redacts_failure(
    tmp_path: Path,
) -> None:
    secret_value = "sk-or-secret-that-must-never-be-persisted"
    observed_workload: dict[str, object] = {}

    approved = {
        "id": 123,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "dph_base": 0.7,
        "dph_total": 0.7,
        "storage_cost": 0.0,
        "diskHour": 0.0,
        "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
    }

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[1:3] == ["search", "offers"]:
            return SimpleNamespace(stdout=json.dumps([approved]), returncode=0)
        if command[1:3] == ["create", "instance"]:
            return SimpleNamespace(stdout='{"id": 2468}', returncode=0)
        if command[1:3] == ["show", "instance"]:
            return SimpleNamespace(
                stdout=json.dumps({**approved, "actual_status": "running"}),
                returncode=0,
            )
        if command[1:] == ["ssh-url", "2468"]:
            return SimpleNamespace(stdout="ssh://root@example.test:22\n", returncode=0)
        if command[0] == "ssh":
            observed_workload["command"] = command
            observed_workload["input"] = kwargs.get("input")
            return SimpleNamespace(
                stdout=f"provider rejected {secret_value}",
                stderr="",
                returncode=17,
            )
        return SimpleNamespace(stdout="", returncode=0)

    plan = lifecycle_plan(
        vastai="vastai",
        config=_config(),
        offer=_offer(),
        bundle=Path("b"),
        fetch_dir=tmp_path / "fetch",
    )
    with pytest.raises(VastPrerequisiteError) as captured:
        execute_lifecycle(
            plan=plan,
            config=_config(),
            metadata_path=tmp_path / "receipt.json",
            run_command=run,
            sleeper=lambda _seconds: None,
            workload_secret=EphemeralWorkloadSecret.openrouter(secret_value),
        )

    command = observed_workload["command"]
    assert isinstance(command, list)
    assert secret_value not in " ".join(command)
    assert "OPENROUTER_API_KEY" in command[-1]
    assert observed_workload["input"] == secret_value + "\n"
    assert secret_value not in str(captured.value)
    receipt_text = (tmp_path / "receipt.json").read_text()
    assert secret_value not in receipt_text
    assert "[REDACTED]" in receipt_text


def test_execute_injects_optional_wandb_secret_over_stdin_without_exposing_values(
    tmp_path: Path,
) -> None:
    openrouter_value = "sk-or-secret-that-must-never-be-persisted"
    wandb_value = "wandb-secret-that-must-never-be-persisted"
    aws_access_value = "AKIAEXAMPLE12345678"
    aws_secret_value = "aws-secret-that-must-never-be-persisted"
    observed_workload: dict[str, object] = {}
    approved = {
        "id": 123,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "dph_base": 0.7,
        "dph_total": 0.7,
        "storage_cost": 0.0,
        "diskHour": 0.0,
        "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
    }

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        if command[1:3] == ["search", "offers"]:
            return SimpleNamespace(stdout=json.dumps([approved]), returncode=0)
        if command[1:3] == ["create", "instance"]:
            return SimpleNamespace(stdout='{"id": 2468}', returncode=0)
        if command[1:3] == ["show", "instance"]:
            return SimpleNamespace(
                stdout=json.dumps({**approved, "actual_status": "running"}),
                returncode=0,
            )
        if command[1:] == ["ssh-url", "2468"]:
            return SimpleNamespace(stdout="ssh://root@example.test:22\n", returncode=0)
        if command[0] == "ssh":
            observed_workload["command"] = command
            observed_workload["input"] = kwargs.get("input")
            return SimpleNamespace(
                stdout=(
                    f"provider rejected {openrouter_value}, {wandb_value}, "
                    f"{aws_access_value}, and {aws_secret_value}"
                ),
                stderr="",
                returncode=17,
            )
        return SimpleNamespace(stdout="", returncode=0)

    secret = EphemeralWorkloadSecret.production(
        openrouter_value=openrouter_value,
        wandb_value=wandb_value,
        aws_credentials=resolve_aws_workload_credentials(
            environment={
                "AWS_ACCESS_KEY_ID": aws_access_value,
                "AWS_SECRET_ACCESS_KEY": aws_secret_value,
            },
            minimum_validity_seconds=3600,
        ),
    )
    plan = lifecycle_plan(
        vastai="vastai",
        config=_config(),
        offer=_offer(),
        bundle=Path("b"),
        fetch_dir=tmp_path / "fetch",
    )
    with pytest.raises(VastPrerequisiteError) as captured:
        execute_lifecycle(
            plan=plan,
            config=_config(),
            metadata_path=tmp_path / "receipt.json",
            run_command=run,
            sleeper=lambda _seconds: None,
            workload_secret=secret,
        )

    command = observed_workload["command"]
    assert isinstance(command, list)
    command_text = " ".join(command)
    assert "OPENROUTER_API_KEY" in command_text
    assert "WANDB_API_KEY" in command_text
    assert "AWS_ACCESS_KEY_ID" in command_text
    assert "AWS_SECRET_ACCESS_KEY" in command_text
    expected_secrets = (
        openrouter_value,
        wandb_value,
        aws_access_value,
        aws_secret_value,
    )
    assert observed_workload["input"] == (
        f"{openrouter_value}\n{wandb_value}\n{aws_access_value}\n"
        f"{aws_secret_value}\n\n"
    )
    for value in expected_secrets:
        assert value not in command_text
        assert value not in repr(secret)
        assert value not in str(captured.value)
    receipt_text = (tmp_path / "receipt.json").read_text()
    for value in expected_secrets:
        assert value not in receipt_text
    assert "[REDACTED]" in receipt_text


@pytest.mark.parametrize("wandb_value", [None, "", " has-spaces ", "bad\nvalue"])
def test_invalid_optional_wandb_secret_disables_monitoring_without_blocking(
    wandb_value: str | None,
) -> None:
    secret = EphemeralWorkloadSecret.production(
        openrouter_value="required-openrouter-secret",
        wandb_value=wandb_value,
    )

    assert secret.stdin_payload() == "required-openrouter-secret\n\n"
    assert secret.environment_names == ("OPENROUTER_API_KEY", "WANDB_API_KEY")


def test_production_secret_still_requires_openrouter() -> None:
    with pytest.raises(VastPrerequisiteError, match="OPENROUTER_API_KEY"):
        EphemeralWorkloadSecret.production(
            openrouter_value="",
            wandb_value="valid-wandb-key",
        )


def test_adaptive_secret_resolves_environment_aws_credentials_and_keeps_values_stdin_only() -> None:
    expiration = datetime.now(timezone.utc) + timedelta(hours=25)
    aws = resolve_aws_workload_credentials(
        environment={
            "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE12345678",
            "AWS_SECRET_ACCESS_KEY": "aws-secret-value-that-must-not-leak",
            "AWS_SESSION_TOKEN": "aws-session-value-that-must-not-leak",
            "AWS_CREDENTIAL_EXPIRATION": expiration.isoformat(),
        },
        minimum_validity_seconds=24 * 3600,
    )
    secret = EphemeralWorkloadSecret.production(
        openrouter_value="openrouter-secret-value",
        wandb_value="wandb-secret-value",
        aws_credentials=aws,
    )

    assert secret.environment_names == (
        "OPENROUTER_API_KEY",
        "WANDB_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    )
    assert secret.stdin_payload().splitlines() == [
        "openrouter-secret-value",
        "wandb-secret-value",
        "AKIAEXAMPLE12345678",
        "aws-secret-value-that-must-not-leak",
        "aws-session-value-that-must-not-leak",
    ]
    assert "aws-secret-value" not in repr(secret)
    assert "aws-session-value" not in repr(secret)


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"AWS_ACCESS_KEY_ID": "AKIAEXAMPLE12345678"},
        {
            "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE12345678",
            "AWS_SECRET_ACCESS_KEY": "aws-secret-value-that-must-not-leak",
            "AWS_SESSION_TOKEN": "temporary-token",
        },
    ],
)
def test_aws_credential_resolution_fails_closed_for_absent_incomplete_or_unbounded_temporary_credentials(
    environment: dict[str, str],
) -> None:
    with pytest.raises(VastPrerequisiteError, match="AWS credentials"):
        resolve_aws_workload_credentials(
            environment=environment,
            minimum_validity_seconds=3600,
        )


def test_aws_credential_resolution_rejects_credentials_expiring_during_paid_workload() -> None:
    with pytest.raises(VastPrerequisiteError, match="expire before"):
        resolve_aws_workload_credentials(
            environment={
                "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE12345678",
                "AWS_SECRET_ACCESS_KEY": "aws-secret-value-that-must-not-leak",
                "AWS_SESSION_TOKEN": "temporary-token",
                "AWS_CREDENTIAL_EXPIRATION": (
                    datetime.now(timezone.utc) + timedelta(minutes=30)
                ).isoformat(),
            },
            minimum_validity_seconds=3600,
        )


def test_aws_profile_resolution_reads_only_named_private_profile(tmp_path: Path) -> None:
    credentials = tmp_path / "credentials"
    credentials.write_text(
        "[production]\n"
        "aws_access_key_id = AKIAEXAMPLE12345678\n"
        "aws_secret_access_key = profile-secret-that-must-not-leak\n"
    )
    credentials.chmod(0o600)

    resolved = resolve_aws_workload_credentials(
        environment={},
        profile_name="production",
        credentials_file=credentials,
        minimum_validity_seconds=24 * 3600,
    )

    assert resolved.source == "shared_credentials_profile"
    assert "profile-secret" not in repr(resolved)


def test_aws_cli_profile_resolution_supports_sso_without_leaking_process_output() -> None:
    secret = "cli-secret-that-must-not-leak"
    token = "cli-session-token-that-must-not-leak"
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        assert kwargs["capture_output"] is True
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "Version": 1,
                    "AccessKeyId": "AKIAEXAMPLE12345678",
                    "SecretAccessKey": secret,
                    "SessionToken": token,
                    "Expiration": (
                        datetime.now(timezone.utc) + timedelta(hours=25)
                    ).isoformat(),
                }
            ),
            stderr="",
        )

    resolved = resolve_aws_cli_profile_credentials(
        profile_name="step5-week",
        minimum_validity_seconds=24 * 3600,
        aws_cli="aws",
        run_command=run,
    )

    assert commands == [[
        "aws", "configure", "export-credentials", "--profile", "step5-week",
        "--format", "process",
    ]]
    assert secret not in repr(resolved)
    assert token not in repr(resolved)


def test_aws_stdin_wrapper_exports_credentials_only_inside_remote_shell() -> None:
    names = (
        "OPENROUTER_API_KEY",
        "WANDB_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    )
    wrapped = _secret_stdin_wrapper(
        'test -n "$AWS_SECRET_ACCESS_KEY" && printf ok',
        names,
    )
    payload = (
        "openrouter-secret\nwandb-secret\nAKIAEXAMPLE12345678\n"
        "aws-secret-that-must-not-leak\n\n"
    )

    completed = subprocess.run(
        ["bash", "-lc", wrapped],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "ok"
    assert "aws-secret-that-must-not-leak" not in wrapped


@pytest.mark.parametrize(
    ("wandb_line", "expected_wandb_state"),
    [("wandb-secret-value", "set"), ("", "unset")],
)
def test_secret_wrapper_executes_entire_compound_command_in_real_bash(
    tmp_path: Path,
    wandb_line: str,
    expected_wandb_state: str,
) -> None:
    first = tmp_path / "first"
    after = tmp_path / "after"
    wandb_state = tmp_path / "wandb-state"
    remote = (
        f"mkdir -p {shlex.quote(str(first))} && "
        f"printf after > {shlex.quote(str(after))} && "
        'if test -n "${WANDB_API_KEY:-}"; then '
        f"printf set > {shlex.quote(str(wandb_state))}; else "
        f"printf unset > {shlex.quote(str(wandb_state))}; fi"
    )
    wrapped = _secret_stdin_wrapper(
        remote, ("OPENROUTER_API_KEY", "WANDB_API_KEY")
    )

    result = subprocess.run(
        ["bash", "-lc", wrapped],
        input=f"openrouter-secret-value\n{wandb_line}\n",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert first.is_dir()
    assert after.read_text() == "after"
    assert wandb_state.read_text() == expected_wandb_state
    assert "openrouter-secret-value" not in wrapped
    assert "wandb-secret-value" not in wrapped


def test_secret_wrapper_propagates_compound_command_exit_status(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    remote = f"printf ran > {shlex.quote(str(marker))} && exit 23"
    wrapped = _secret_stdin_wrapper(remote, ("OPENROUTER_API_KEY",))

    result = subprocess.run(
        ["bash", "-lc", wrapped],
        input="openrouter-secret-value\n",
        capture_output=True,
        text=True,
        check=False,
    )

    assert marker.read_text() == "ran"
    assert result.returncode == 23
    assert "openrouter-secret-value" not in wrapped


def test_execute_revalidates_offer_before_create(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        assert command[1:3] == ["search", "offers"]
        # The Vast CLI's server-side ``id=...`` filter can return an empty
        # inventory for a currently rentable ask.  Query the narrowly eligible
        # GPU inventory and require the controller to match the approved ID
        # and price locally before creation.
        assert "gpu_name=RTX_4090" in command[3]
        assert "rentable=true" in command[3]
        return SimpleNamespace(
            stdout=json.dumps(
                [
                    {
                        "id": 123,
                        "gpu_name": "RTX 4090",
                        "num_gpus": 1,
                        "gpu_ram": 24576,
                        "dph_base": 0.71,
                        "dph_total": 0.71,
                        "storage_cost": 0.0,
                        "inet_down_cost": 0.0,
                        "inet_up_cost": 0.0,
                    }
                ]
            ),
            returncode=0,
        )

    config = _config()
    plan = lifecycle_plan(
        vastai="vastai",
        config=config,
        offer=_offer(),
        bundle=Path("b"),
        fetch_dir=tmp_path / "f",
    )
    with pytest.raises(VastPrerequisiteError, match="price changed"):
        execute_lifecycle(
            plan=plan,
            config=config,
            metadata_path=tmp_path / "receipt.json",
            run_command=run,
        )
    assert not any(command[1:3] == ["create", "instance"] for command in calls)


def test_readiness_poll_and_transient_copy_are_bounded_and_recorded(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    show_count = 0
    copy_count = 0
    payload = _output_archive({"receipt.json": b"{}\n"})
    approved = {
        "id": 123,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "dph_base": 0.7,
        "dph_total": 0.7,
        "storage_cost": 0.0,
        "diskHour": 0.0,
        "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
    }

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal show_count, copy_count
        calls.append(command)
        if command[1:3] == ["search", "offers"]:
            return SimpleNamespace(stdout=json.dumps([approved]), returncode=0)
        if command[1:3] == ["create", "instance"]:
            return SimpleNamespace(stdout='{"id": 999}', returncode=0)
        if command[1:3] == ["show", "instance"]:
            show_count += 1
            state = "loading" if show_count == 1 else "running"
            return SimpleNamespace(
                stdout=json.dumps({**approved, "actual_status": state}), returncode=0
            )
        if command[1:] == ["ssh-url", "999"]:
            return SimpleNamespace(stdout="ssh://root@example.test:22\n", returncode=0)
        if command[1:2] == ["copy"] and "local:b" in command:
            copy_count += 1
            if copy_count == 1:
                raise subprocess.CalledProcessError(1, command)
        if command[0] == "ssh" and "truth_editing_vast_output_archive_v1" in command[-1]:
            return SimpleNamespace(stdout=_remote_archive_identity(payload), returncode=0)
        if command[1:2] == ["copy"] and command[-2].endswith("truth-prerequisite-outputs.tar.gz"):
            Path(command[-1].removeprefix("local:")).write_bytes(payload)
        return SimpleNamespace(stdout="", returncode=0)

    config = _config()
    plan = lifecycle_plan(
        vastai="vastai", config=config, offer=_offer(), bundle=Path("b"), fetch_dir=tmp_path / "fetch"
    )
    receipt = execute_lifecycle(
        plan=plan,
        config=config,
        metadata_path=tmp_path / "receipt.json",
        run_command=run,
        sleeper=lambda _seconds: None,
    )
    assert show_count == 2
    assert copy_count == 2
    assert "instance_ready" in [event["event"] for event in receipt["events"]]


def test_cli_requires_double_execute_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path(__file__).parents[1] / "scripts/run_truth_editing_vast_prerequisites.py"
    spec = importlib.util.spec_from_file_location("vast_prerequisite_cli", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                config=tmp_path / "config.json",
                repo=tmp_path,
                offer_json=tmp_path / "offer.json",
                bundle=tmp_path / "bundle.tgz",
                fetch_dir=tmp_path / "fetch",
                    metadata=tmp_path / "receipt.json",
                    ssh_identity=None,
                    vastai="vastai",
                execute=True,
                confirmed_cost_approval=False,
            )
        ),
    )
    (tmp_path / "worker.sh").write_text("true\n")
    (tmp_path / "config.json").write_text(json.dumps(_raw_config(["worker.sh"])))
    (tmp_path / "offer.json").write_text(
        json.dumps(
            {
                "id": 123,
                "gpu_name": "RTX 4090",
                "num_gpus": 1,
                "gpu_ram": 24576,
                "dph_base": 0.7,
                "dph_total": 0.7,
                "storage_cost": 0.0,
                "inet_down_cost": 0.0,
                "inet_up_cost": 0.0,
            }
        )
    )
    with pytest.raises(SystemExit, match="confirmed-cost-approval"):
        module.main()


def test_repository_config_parses_and_builds_narrow_bundle(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    config = JobConfig.from_mapping(
        json.loads((root / "configs/truth_editing_vast_prerequisites_v1.json").read_text())
    )
    missing = [path for path in config.bundle_paths if not (root / path).is_file()]
    assert missing == []
    assert any(path.endswith("truth_editing_refusal_extraction.py") for path in config.bundle_paths)
    assert any(path.endswith("run_truth_editing_prerequisite_inference.py") for path in config.bundle_paths)
    manifest = build_bundle(root, config, tmp_path / "prerequisites.tgz")
    assert manifest["files"]
    assert sum(item["bytes"] for item in manifest["files"]) < 128 * 1024**2
