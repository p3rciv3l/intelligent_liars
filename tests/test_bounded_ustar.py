from __future__ import annotations

import hashlib
import io
import os
import tarfile
from pathlib import Path

import pytest

from intelligent_liars.bounded_ustar import (
    BLOCK_BYTES,
    MAX_IN_MEMORY_MEMBER_BYTES,
    BoundedUstarError,
    MemberRule,
    UstarPolicy,
    open_bounded_ustar,
)


MEMBERS = {
    "manifest.json": b'{"files":[]}',
    "state.json": b'{"step":20}',
    "training_state.pt": b"tensor-state",
}


def _policy(**overrides: object) -> UstarPolicy:
    rules = {
        name: MemberRule(max_bytes=1024 * 1024, modes=frozenset({0o600}))
        for name in MEMBERS
    }
    values = {
        "members": rules,
        "max_archive_bytes": 8 * 1024 * 1024,
        "max_total_payload_bytes": 3 * 1024 * 1024,
    }
    values.update(overrides)
    return UstarPolicy(**values)  # type: ignore[arg-type]


def _archive(path: Path, *, members: dict[str, bytes] | None = None) -> None:
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for name, value in sorted((members or MEMBERS).items()):
            info = tarfile.TarInfo(name)
            info.size = len(value)
            info.mode = 0o600
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(value))


def _first_header(path: Path) -> bytearray:
    return bytearray(path.read_bytes())


def _checksum(payload: bytearray, offset: int = 0) -> None:
    payload[offset + 148 : offset + 156] = b"        "
    checksum = sum(payload[offset : offset + BLOCK_BYTES])
    payload[offset + 148 : offset + 156] = f"{checksum:06o}\0 ".encode()


def _first_zero_block(payload: bytes) -> int:
    for offset in range(0, len(payload), BLOCK_BYTES):
        if payload[offset : offset + BLOCK_BYTES] == b"\0" * BLOCK_BYTES:
            return offset
    raise AssertionError("archive has no terminator")


def test_valid_archive_is_exactly_inventoried_and_streamed(tmp_path: Path):
    path = tmp_path / "valid.tar"
    _archive(path)

    with open_bounded_ustar(path, policy=_policy()) as archive:
        assert set(archive.members) == set(MEMBERS)
        assert archive.read_member("manifest.json") == MEMBERS["manifest.json"]
        output = io.BytesIO()
        digest = archive.copy_and_hash("training_state.pt", output)
        assert output.getvalue() == MEMBERS["training_state.pt"]
        assert digest == hashlib.sha256(MEMBERS["training_state.pt"]).hexdigest()
        assert archive.archive_sha256() == hashlib.sha256(path.read_bytes()).hexdigest()


def test_policy_name_limit_matches_required_nul_terminated_header():
    with pytest.raises(ValueError, match="member name"):
        UstarPolicy(
            members={"a" * 100: MemberRule(max_bytes=1)},
            max_archive_bytes=1024,
            max_total_payload_bytes=1,
        )


@pytest.mark.parametrize(
    "typeflag",
    [b"\0", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"x", b"g", b"L", b"K", b"S"],
)
def test_rejects_null_pax_gnu_sparse_links_directories_and_special_types(
    tmp_path: Path, typeflag: bytes
):
    valid = tmp_path / "valid.tar"
    malicious = tmp_path / f"type-{typeflag.hex()}.tar"
    _archive(valid)
    payload = _first_header(valid)
    payload[156:157] = typeflag
    _checksum(payload)
    malicious.write_bytes(payload)

    with pytest.raises(BoundedUstarError, match="special member rejected"):
        open_bounded_ustar(malicious, policy=_policy())


def test_rejects_actual_pax_and_gnu_long_name_archives(tmp_path: Path):
    pax = tmp_path / "pax.tar"
    with tarfile.open(pax, "w", format=tarfile.PAX_FORMAT, pax_headers={"comment": "x"}) as archive:
        info = tarfile.TarInfo("manifest.json")
        info.size = 2
        archive.addfile(info, io.BytesIO(b"{}"))
    gnu = tmp_path / "gnu.tar"
    with tarfile.open(gnu, "w", format=tarfile.GNU_FORMAT) as archive:
        info = tarfile.TarInfo("a" * 180)
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))

    for path in (pax, gnu):
        with pytest.raises(BoundedUstarError):
            open_bounded_ustar(path, policy=_policy())


def test_rejects_base256_size_reserved_bytes_and_nonzero_padding(tmp_path: Path):
    valid = tmp_path / "valid.tar"
    _archive(valid)
    base256 = _first_header(valid)
    base256[124] = 0x80
    _checksum(base256)
    (tmp_path / "base256.tar").write_bytes(base256)
    with pytest.raises(BoundedUstarError, match="base-256"):
        open_bounded_ustar(tmp_path / "base256.tar", policy=_policy())

    reserved = _first_header(valid)
    reserved[500] = 1
    _checksum(reserved)
    (tmp_path / "reserved.tar").write_bytes(reserved)
    with pytest.raises(BoundedUstarError, match="reserved"):
        open_bounded_ustar(tmp_path / "reserved.tar", policy=_policy())

    padding = _first_header(valid)
    first_size = int(bytes(padding[124:136]).rstrip(b"\0 "), 8)
    padding[BLOCK_BYTES + first_size] = 1
    (tmp_path / "padding.tar").write_bytes(padding)
    with pytest.raises(BoundedUstarError, match="nonzero padding"):
        open_bounded_ustar(tmp_path / "padding.tar", policy=_policy())


def test_rejects_non_block_trailer_one_block_terminator_and_hidden_trailing_data(
    tmp_path: Path,
):
    valid = tmp_path / "valid.tar"
    _archive(valid)
    raw = valid.read_bytes()
    (tmp_path / "partial.tar").write_bytes(raw + b"\0")
    with pytest.raises(BoundedUstarError, match="archive size"):
        open_bounded_ustar(tmp_path / "partial.tar", policy=_policy())

    terminator = bytearray(raw)
    first_zero = _first_zero_block(raw)
    terminator[first_zero + BLOCK_BYTES] = 1
    (tmp_path / "one-zero.tar").write_bytes(terminator)
    with pytest.raises(BoundedUstarError, match="only one zero"):
        open_bounded_ustar(tmp_path / "one-zero.tar", policy=_policy())

    trailing = bytearray(raw)
    trailing[first_zero + 2 * BLOCK_BYTES] = 1
    (tmp_path / "trailing.tar").write_bytes(trailing)
    with pytest.raises(BoundedUstarError, match="trailing"):
        open_bounded_ustar(tmp_path / "trailing.tar", policy=_policy())


def test_exact_allowlist_count_member_and_aggregate_caps_precede_payload_reads(
    tmp_path: Path,
):
    extra = {**MEMBERS, "extra.bin": b"x"}
    extra_path = tmp_path / "extra.tar"
    _archive(extra_path, members=extra)
    with pytest.raises(BoundedUstarError, match="not allowlisted|too many members"):
        open_bounded_ustar(extra_path, policy=_policy())

    path = tmp_path / "valid.tar"
    _archive(path)
    small_member = _policy(
        members={
            name: MemberRule(max_bytes=1, modes=frozenset({0o600}))
            for name in MEMBERS
        }
    )
    with pytest.raises(BoundedUstarError, match="predeclared cap"):
        open_bounded_ustar(path, policy=small_member)

    aggregate = _policy(max_total_payload_bytes=sum(map(len, MEMBERS.values())) - 1)
    with pytest.raises(BoundedUstarError, match="aggregate"):
        open_bounded_ustar(path, policy=aggregate)


def test_sparse_oversized_file_and_in_memory_read_are_rejected_without_large_allocation(
    tmp_path: Path,
):
    oversized = tmp_path / "oversized.tar"
    with oversized.open("wb") as handle:
        handle.truncate(8 * 1024 * 1024 + BLOCK_BYTES)
    with pytest.raises(BoundedUstarError, match="archive size"):
        open_bounded_ustar(oversized, policy=_policy())

    path = tmp_path / "valid.tar"
    _archive(path)
    with open_bounded_ustar(path, policy=_policy()) as archive:
        with pytest.raises(BoundedUstarError, match="in-memory read cap"):
            archive.read_member("manifest.json", cap=MAX_IN_MEMORY_MEMBER_BYTES + 1)
        with pytest.raises(BoundedUstarError, match="in-memory read cap"):
            archive.read_member("manifest.json", cap=True)


def test_copy_handles_partial_writes_and_rejects_zero_progress(tmp_path: Path):
    path = tmp_path / "valid.tar"
    _archive(path)

    class PartialSink:
        def __init__(self, limit: int):
            self.limit = limit
            self.value = bytearray()

        def write(self, value: bytes) -> int:
            written = min(self.limit, len(value))
            self.value.extend(value[:written])
            return written

    with open_bounded_ustar(path, policy=_policy()) as archive:
        partial = PartialSink(2)
        archive.copy_and_hash("training_state.pt", partial)  # type: ignore[arg-type]
        assert bytes(partial.value) == MEMBERS["training_state.pt"]
        with pytest.raises(BoundedUstarError, match="short write"):
            archive.copy_and_hash(
                "training_state.pt", PartialSink(0)  # type: ignore[arg-type]
            )


def test_platform_without_nofollow_fails_closed(tmp_path: Path, monkeypatch):
    path = tmp_path / "valid.tar"
    _archive(path)
    monkeypatch.delattr(os, "O_NOFOLLOW")
    with pytest.raises(BoundedUstarError, match="O_NOFOLLOW"):
        open_bounded_ustar(path, policy=_policy())


def test_open_descriptor_is_pinned_against_path_replacement_and_symlinks(tmp_path: Path):
    path = tmp_path / "archive.tar"
    replacement = tmp_path / "replacement.tar"
    _archive(path)
    archive = open_bounded_ustar(path, policy=_policy())
    _archive(replacement)
    replacement.replace(path)
    try:
        with pytest.raises(BoundedUstarError, match="changed"):
            archive.archive_sha256()
    finally:
        with pytest.raises(BoundedUstarError, match="changed"):
            archive.close()

    target = tmp_path / "target.tar"
    _archive(target)
    link = tmp_path / "link.tar"
    link.symlink_to(target)
    with pytest.raises(BoundedUstarError, match="safely open"):
        open_bounded_ustar(link, policy=_policy())


def test_in_place_mutation_is_detected_on_pinned_descriptor(tmp_path: Path):
    path = tmp_path / "archive.tar"
    _archive(path)
    archive = open_bounded_ustar(path, policy=_policy())
    original = path.stat()
    with path.open("r+b") as handle:
        handle.seek(BLOCK_BYTES)
        handle.write(b"X")
        handle.flush()
        os.fsync(handle.fileno())
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(BoundedUstarError, match="changed"):
        archive.archive_sha256()
    with pytest.raises(BoundedUstarError, match="changed"):
        archive.close()
