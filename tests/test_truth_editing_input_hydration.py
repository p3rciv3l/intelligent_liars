from __future__ import annotations

import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_input_hydration import (
    HydrationError,
    MAX_MEMBERS,
    build_production_input_bundle,
    build_production_input_bundle_from_entries,
    entries_from_vast_job_config,
    hydrate_production_inputs,
)


MANIFEST_FORMAT = "truth_editing_production_input_manifest_v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bundle(members: dict[str, bytes]) -> bytes:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed:
        compressed.write(tar_buffer.getvalue())
    return output.getvalue()


def _write_manifest(
    tmp_path: Path,
    *,
    archive_bytes: bytes,
    members: dict[str, tuple[str, bytes]],
    uri: str | None = None,
) -> Path:
    archive_path = tmp_path / "inputs.tar.gz"
    archive_path.write_bytes(archive_bytes)
    manifest = {
        "format": MANIFEST_FORMAT,
        "archive": {
            "uri": uri or str(archive_path),
            "sha256": _sha256(archive_bytes),
            "size_bytes": len(archive_bytes),
            "compression": "gzip",
            "archive_format": "ustar",
        },
        "members": [
            {
                "archive_path": archive_name,
                "destination_path": destination,
                "sha256": _sha256(payload),
                "size_bytes": len(payload),
            }
            for archive_name, (destination, payload) in members.items()
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_hydrates_exact_bundle_and_idempotently_verifies_existing_files(
    tmp_path: Path,
) -> None:
    payloads = {
        "bundle/directions.json": b'{"directions":[]}\n',
        "bundle/panel.jsonl": b'{"question":"q"}\n',
    }
    archive = _bundle(payloads)
    manifest_path = _write_manifest(
        tmp_path,
        archive_bytes=archive,
        members={
            name: (f"production-inputs/{Path(name).name}", payload)
            for name, payload in payloads.items()
        },
    )
    repo = tmp_path / "repo"
    repo.mkdir()

    first = hydrate_production_inputs(manifest_path, repo_root=repo)
    second = hydrate_production_inputs(manifest_path, repo_root=repo)

    assert (repo / "production-inputs/directions.json").read_bytes() == payloads[
        "bundle/directions.json"
    ]
    assert (repo / "production-inputs/panel.jsonl").read_bytes() == payloads[
        "bundle/panel.jsonl"
    ]
    assert first["format"] == "truth_editing_production_input_hydration_receipt_v1"
    assert first["status"] == "hydrated"
    assert second["status"] == "verified_existing"
    assert second["manifest_sha256"] == first["manifest_sha256"]


def test_build_is_deterministic_and_round_trips_from_json_allowlist(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    (source_root / "inputs").mkdir(parents=True)
    (source_root / "inputs/z.json").write_text('{"z":1}\n')
    (source_root / "inputs/a.txt").write_text("alpha\n")
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "format": "truth_editing_production_input_build_allowlist_v1",
                "files": [
                    {
                        "source_path": "inputs/z.json",
                        "archive_path": "payload/z.json",
                        "destination_path": "data/z.json",
                    },
                    {
                        "source_path": "inputs/a.txt",
                        "archive_path": "payload/a.txt",
                        "destination_path": "data/a.txt",
                    },
                ],
            }
        )
    )
    first_archive = tmp_path / "first.tar.gz"
    first_manifest = tmp_path / "first.json"
    second_archive = tmp_path / "second.tar.gz"
    second_manifest = tmp_path / "second.json"

    build_production_input_bundle(
        allowlist,
        source_root=source_root,
        archive_path=first_archive,
        manifest_path=first_manifest,
        archive_uri="bundle.tar.gz",
    )
    build_production_input_bundle(
        allowlist,
        source_root=source_root,
        archive_path=second_archive,
        manifest_path=second_manifest,
        archive_uri="bundle.tar.gz",
    )

    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    hydrated = tmp_path / "hydrated"
    hydrated.mkdir()
    # The URI is relative to the manifest, so copy the deterministic bundle to it.
    (tmp_path / "bundle.tar.gz").write_bytes(first_archive.read_bytes())
    hydrate_production_inputs(first_manifest, repo_root=hydrated)
    assert (hydrated / "data/a.txt").read_text() == "alpha\n"
    assert (hydrated / "data/z.json").read_text() == '{"z":1}\n'


def test_newline_allowlist_uses_same_repo_relative_path_for_all_roles(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested/file.txt").write_text("content\n")
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("nested/file.txt\n")

    manifest = build_production_input_bundle(
        allowlist,
        source_root=source,
        archive_path=tmp_path / "bundle.tar.gz",
        manifest_path=tmp_path / "manifest.json",
    )

    assert manifest["members"][0]["archive_path"] == "nested/file.txt"
    assert manifest["members"][0]["destination_path"] == "nested/file.txt"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value["archive"].update(compression="zip"),
        lambda value: value["members"][0].update(destination_path="../escaped"),
        lambda value: value["members"][0].update(sha256="A" * 64),
    ],
)
def test_manifest_is_strict_and_fail_closed(tmp_path: Path, mutation) -> None:
    payload = b"payload"
    archive = _bundle({"member.txt": payload})
    path = _write_manifest(
        tmp_path,
        archive_bytes=archive,
        members={"member.txt": ("inputs/member.txt", payload)},
    )
    value = json.loads(path.read_text())
    mutation(value)
    path.write_text(json.dumps(value))
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(HydrationError):
        hydrate_production_inputs(path, repo_root=repo)

    assert list(repo.iterdir()) == []


def test_rejects_extra_archive_member_before_install(tmp_path: Path) -> None:
    expected = b"expected"
    archive = _bundle({"member.txt": expected, "extra.txt": b"extra"})
    path = _write_manifest(
        tmp_path,
        archive_bytes=archive,
        members={"member.txt": ("inputs/member.txt", expected)},
    )
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(HydrationError, match="too many members|allowlist"):
        hydrate_production_inputs(path, repo_root=repo)

    assert not (repo / "inputs/member.txt").exists()


def test_rejects_archive_or_existing_destination_identity_mismatch(tmp_path: Path) -> None:
    payload = b"expected"
    archive = _bundle({"member.txt": payload})
    path = _write_manifest(
        tmp_path,
        archive_bytes=archive,
        members={"member.txt": ("inputs/member.txt", payload)},
    )
    repo = tmp_path / "repo"
    (repo / "inputs").mkdir(parents=True)
    (repo / "inputs/member.txt").write_bytes(b"wrong")

    with pytest.raises(HydrationError, match="existing destination"):
        hydrate_production_inputs(path, repo_root=repo)

    assert (repo / "inputs/member.txt").read_bytes() == b"wrong"


def test_s3_uri_uses_exact_bucket_and_key(tmp_path: Path) -> None:
    payload = b"expected"
    archive = _bundle({"member.txt": payload})
    path = _write_manifest(
        tmp_path,
        archive_bytes=archive,
        members={"member.txt": ("inputs/member.txt", payload)},
        uri="s3://production-bucket/prefix/bundle.tar.gz",
    )

    class Client:
        calls: list[tuple[str, str]] = []

        def get_object(self, *, Bucket: str, Key: str):
            self.calls.append((Bucket, Key))
            return {"ContentLength": len(archive), "Body": io.BytesIO(archive)}

    client = Client()
    repo = tmp_path / "repo"
    repo.mkdir()
    hydrate_production_inputs(path, repo_root=repo, s3_client=client)

    assert client.calls == [("production-bucket", "prefix/bundle.tar.gz")]


def test_rejects_link_and_truncated_ustar_before_install(tmp_path: Path) -> None:
    for case in ("link", "truncated"):
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            info = tarfile.TarInfo("member.txt")
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            if case == "link":
                info.type = tarfile.SYMTYPE
                info.linkname = "target"
                archive.addfile(info)
                expected = b""
            else:
                expected = b"payload"
                info.size = len(expected)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(expected))
        tar_bytes = tar_buffer.getvalue()
        if case == "truncated":
            tar_bytes = tar_bytes[:1536]
        output = io.BytesIO()
        with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as stream:
            stream.write(tar_bytes)
        bundle = output.getvalue()
        case_dir = tmp_path / case
        case_dir.mkdir()
        manifest = _write_manifest(
            case_dir,
            archive_bytes=bundle,
            members={"member.txt": ("inputs/member.txt", expected)},
        )
        repo = case_dir / "repo"
        repo.mkdir()

        with pytest.raises(HydrationError):
            hydrate_production_inputs(manifest, repo_root=repo)

        assert not (repo / "inputs/member.txt").exists()


def test_build_rejects_member_count_overflow(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    entries = [
        {
            "source_path": f"file-{index}",
            "archive_path": f"file-{index}",
            "destination_path": f"file-{index}",
        }
        for index in range(MAX_MEMBERS + 1)
    ]

    with pytest.raises(HydrationError, match="member-count"):
        build_production_input_bundle_from_entries(
            entries,
            source_root=root,
            archive_path=tmp_path / "bundle.tar.gz",
            manifest_path=tmp_path / "manifest.json",
        )


def test_vast_job_ignored_only_derives_exact_git_ignored_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "artifacts").mkdir(parents=True)
    (repo / "tracked").mkdir()
    (repo / "artifacts/private.bin").write_bytes(b"private")
    (repo / "tracked/config.json").write_text("{}\n")
    (repo / ".gitignore").write_text("artifacts/\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    config = repo / "job.json"
    config.write_text(
        json.dumps(
            {
                "base_job": {
                    "bundle_paths": ["tracked/config.json", "artifacts/private.bin"]
                }
            }
        )
    )

    entries = entries_from_vast_job_config(config, repo_root=repo, ignored_only=True)

    assert [entry["source_path"] for entry in entries] == ["artifacts/private.bin"]
    assert entries[0]["destination_path"] == "artifacts/private.bin"
    assert entries[0]["archive_path"].startswith("members/")


def test_rejects_destination_prefix_collision_before_fetch(tmp_path: Path) -> None:
    payloads = {"one": b"one", "two": b"two"}
    archive = _bundle(payloads)
    manifest = _write_manifest(
        tmp_path,
        archive_bytes=archive,
        members={"one": ("inputs/node", b"one"), "two": ("inputs/node/child", b"two")},
    )
    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(HydrationError, match="ancestor collision"):
        hydrate_production_inputs(manifest, repo_root=repo)

    assert list(repo.iterdir()) == []


def test_archive_and_member_corruption_leave_no_installed_files(tmp_path: Path) -> None:
    payload = b"expected"
    for case in ("archive", "member"):
        case_dir = tmp_path / case
        case_dir.mkdir()
        archive = _bundle({"member.txt": payload})
        manifest = _write_manifest(
            case_dir,
            archive_bytes=archive,
            members={"member.txt": ("inputs/member.txt", payload)},
        )
        if case == "archive":
            (case_dir / "inputs.tar.gz").write_bytes(archive + b"unexpected")
        else:
            value = json.loads(manifest.read_text())
            value["members"][0]["sha256"] = _sha256(b"different")
            manifest.write_text(json.dumps(value))
        repo = case_dir / "repo"
        repo.mkdir()

        with pytest.raises(HydrationError):
            hydrate_production_inputs(manifest, repo_root=repo)

        assert not (repo / "inputs/member.txt").exists()


def test_install_failure_rolls_back_all_new_files(tmp_path: Path, monkeypatch) -> None:
    payloads = {"one": b"one", "two": b"two"}
    archive = _bundle(payloads)
    manifest = _write_manifest(
        tmp_path,
        archive_bytes=archive,
        members={"one": ("inputs/one", b"one"), "two": ("inputs/two", b"two")},
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    real_link = __import__("os").link
    installs = 0

    def flaky_link(src, dst, *args, **kwargs):
        nonlocal installs
        if kwargs.get("dst_dir_fd") is not None:
            installs += 1
            if installs == 2:
                raise OSError("injected install failure")
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr("intelligent_liars.truth_editing_input_hydration.os.link", flaky_link)

    with pytest.raises(HydrationError, match="atomic destination"):
        hydrate_production_inputs(manifest, repo_root=repo)

    assert not (repo / "inputs/one").exists()
    assert not (repo / "inputs/two").exists()


def test_real_scale_1525_member_build_hydrate_and_rerun(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    entries = []
    for index in range(1525):
        name = f"inputs/{index:04d}.txt"
        path = source / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"{index}\n")
        entries.append(
            {
                "source_path": name,
                "archive_path": f"members/{index:04d}",
                "destination_path": name,
            }
        )
    archive = tmp_path / "bundle.tar.gz"
    manifest = tmp_path / "manifest.json"
    build_production_input_bundle_from_entries(
        entries,
        source_root=source,
        archive_path=archive,
        manifest_path=manifest,
    )
    hydrated = tmp_path / "hydrated"
    hydrated.mkdir()

    assert hydrate_production_inputs(manifest, repo_root=hydrated)["status"] == "hydrated"
    assert (
        hydrate_production_inputs(manifest, repo_root=hydrated)["status"]
        == "verified_existing"
    )
    assert len(list((hydrated / "inputs").iterdir())) == 1525


def test_cli_receipt_is_byte_identical_on_rerun(tmp_path: Path) -> None:
    payload = b"payload"
    archive = _bundle({"member.txt": payload})
    manifest = _write_manifest(
        tmp_path,
        archive_bytes=archive,
        members={"member.txt": ("inputs/member.txt", payload)},
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    receipt = tmp_path / "receipt.json"
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "scripts/hydrate_truth_editing_production_inputs.py"),
        "hydrate",
        "--manifest",
        str(manifest),
        "--repo-root",
        str(repo),
        "--receipt",
        str(receipt),
    ]

    subprocess.run(command, check=True, capture_output=True)
    first = receipt.read_bytes()
    subprocess.run(command, check=True, capture_output=True)

    assert receipt.read_bytes() == first
    assert json.loads(first)["status"] == "verified"
