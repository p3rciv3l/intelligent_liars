"""Strict, allocation-bounded reader for controller-trusted USTAR archives.

This intentionally implements only the small USTAR subset emitted by the Step 5
workers.  It does not invoke :mod:`tarfile` while parsing untrusted bytes, so PAX,
GNU long-name, sparse, link, device, directory, and base-256 extensions never
reach a permissive compatibility parser.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import BinaryIO

BLOCK_BYTES = 512
MAX_POLICY_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_POLICY_MEMBERS = 256
MAX_IN_MEMORY_MEMBER_BYTES = 32 * 1024 * 1024
STREAM_CHUNK_BYTES = 1024 * 1024


class BoundedUstarError(ValueError):
    """An archive is outside the controller's exact bounded USTAR contract."""


@dataclass(frozen=True)
class MemberRule:
    """Predeclared limits for one required regular-file member."""

    max_bytes: int
    min_bytes: int = 1
    modes: frozenset[int] = frozenset({0o600, 0o644})

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_bytes, bool)
            or isinstance(self.max_bytes, bool)
            or not isinstance(self.min_bytes, int)
            or not isinstance(self.max_bytes, int)
            or self.min_bytes < 0
            or self.max_bytes < self.min_bytes
            or self.max_bytes > MAX_POLICY_ARCHIVE_BYTES
            or not self.modes
            or any(
                isinstance(mode, bool)
                or not isinstance(mode, int)
                or mode not in {0o600, 0o644}
                for mode in self.modes
            )
        ):
            raise ValueError("invalid bounded USTAR member rule")


@dataclass(frozen=True)
class UstarPolicy:
    """Exact member allowlist plus archive and aggregate payload caps."""

    members: Mapping[str, MemberRule]
    max_archive_bytes: int
    max_total_payload_bytes: int

    def __post_init__(self) -> None:
        normalized = dict(self.members)
        if not normalized or len(normalized) > MAX_POLICY_MEMBERS:
            raise ValueError("bounded USTAR policy has an invalid member count")
        for name, rule in normalized.items():
            _validate_member_name(name)
            if not isinstance(rule, MemberRule):
                raise TypeError("bounded USTAR policy member rule is invalid")
        for value in (self.max_archive_bytes, self.max_total_payload_bytes):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                or value > MAX_POLICY_ARCHIVE_BYTES
            ):
                raise ValueError("bounded USTAR policy byte cap is invalid")
        if sum(rule.min_bytes for rule in normalized.values()) > self.max_total_payload_bytes:
            raise ValueError("bounded USTAR minimum payload exceeds aggregate cap")
        object.__setattr__(self, "members", MappingProxyType(normalized))


@dataclass(frozen=True)
class UstarMember:
    """A validated regular-file extent on the pinned archive descriptor."""

    name: str
    offset: int
    size_bytes: int
    mode: int


class BoundedUstarArchive:
    """Pinned descriptor plus exact validated member extents."""

    def __init__(
        self,
        descriptor: int,
        *,
        size_bytes: int,
        identity: tuple[int, int, int, int, int],
        members: Mapping[str, UstarMember],
    ) -> None:
        self._descriptor = descriptor
        self.size_bytes = size_bytes
        self._identity = identity
        self.members = MappingProxyType(dict(members))

    def __enter__(self) -> BoundedUstarArchive:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._descriptor >= 0:
            try:
                self._assert_unchanged()
            finally:
                os.close(self._descriptor)
                self._descriptor = -1

    def read_member(self, name: str, *, cap: int = MAX_IN_MEMORY_MEMBER_BYTES) -> bytes:
        """Read a small member only after both policy and caller caps are known."""
        member = self._member(name)
        if (
            isinstance(cap, bool)
            or not isinstance(cap, int)
            or cap < 0
            or cap > MAX_IN_MEMORY_MEMBER_BYTES
            or member.size_bytes > cap
        ):
            raise BoundedUstarError("USTAR member exceeds the in-memory read cap")
        value = _pread_exact(self._descriptor, member.offset, member.size_bytes)
        self._assert_unchanged()
        return value

    def copy_and_hash(self, name: str, destination: BinaryIO | None = None) -> str:
        """Stream one member in fixed chunks, optionally copying it to a trusted file."""
        member = self._member(name)
        digest = hashlib.sha256()
        remaining = member.size_bytes
        offset = member.offset
        while remaining:
            chunk = os.pread(self._descriptor, min(STREAM_CHUNK_BYTES, remaining), offset)
            if not chunk:
                raise BoundedUstarError("USTAR member changed or truncated during read")
            digest.update(chunk)
            if destination is not None:
                pending = memoryview(chunk)
                while pending:
                    written = destination.write(pending)
                    if (
                        isinstance(written, bool)
                        or not isinstance(written, int)
                        or written <= 0
                        or written > len(pending)
                    ):
                        raise BoundedUstarError("USTAR destination made a short write")
                    pending = pending[written:]
            offset += len(chunk)
            remaining -= len(chunk)
        self._assert_unchanged()
        return digest.hexdigest()

    def archive_sha256(self) -> str:
        """Hash exactly the scanned bytes on the same pinned descriptor."""
        digest = hashlib.sha256()
        offset = 0
        remaining = self.size_bytes
        while remaining:
            chunk = os.pread(self._descriptor, min(STREAM_CHUNK_BYTES, remaining), offset)
            if not chunk:
                raise BoundedUstarError("USTAR archive changed or truncated during hash")
            digest.update(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        self._assert_unchanged()
        return digest.hexdigest()

    def _member(self, name: str) -> UstarMember:
        try:
            return self.members[name]
        except KeyError as error:
            raise BoundedUstarError(f"USTAR member is not allowlisted: {name}") from error

    def _assert_unchanged(self) -> None:
        if self._descriptor < 0:
            raise BoundedUstarError("USTAR archive descriptor is closed")
        if _stat_identity(os.fstat(self._descriptor)) != self._identity:
            raise BoundedUstarError("USTAR archive changed during verification")


def open_bounded_ustar(path: Path, *, policy: UstarPolicy) -> BoundedUstarArchive:
    """Open once with ``O_NOFOLLOW``, scan headers, and return pinned extents."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise BoundedUstarError("platform lacks required O_NOFOLLOW support")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BoundedUstarError("cannot safely open USTAR archive") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise BoundedUstarError("USTAR archive is not a regular file")
        size = metadata.st_size
        if (
            size <= 0
            or size > policy.max_archive_bytes
            or size % BLOCK_BYTES != 0
        ):
            raise BoundedUstarError("USTAR archive size is outside its exact cap")
        identity = _stat_identity(metadata)
        members = _scan_headers(descriptor, archive_size=size, policy=policy)
        if _stat_identity(os.fstat(descriptor)) != identity:
            raise BoundedUstarError("USTAR archive changed during header scan")
        return BoundedUstarArchive(
            descriptor,
            size_bytes=size,
            identity=identity,
            members=members,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _scan_headers(
    descriptor: int, *, archive_size: int, policy: UstarPolicy
) -> dict[str, UstarMember]:
    members: dict[str, UstarMember] = {}
    total_payload = 0
    offset = 0
    found_terminator = False
    while offset < archive_size:
        header = _pread_exact(descriptor, offset, BLOCK_BYTES)
        if header == b"\0" * BLOCK_BYTES:
            second = _pread_exact(descriptor, offset + BLOCK_BYTES, BLOCK_BYTES)
            if second != b"\0" * BLOCK_BYTES:
                raise BoundedUstarError("USTAR archive has only one zero terminator block")
            offset += 2 * BLOCK_BYTES
            found_terminator = True
            break
        if len(members) >= len(policy.members):
            raise BoundedUstarError("USTAR archive has too many members")
        if header[257:263] != b"ustar\0" or header[263:265] != b"00":
            raise BoundedUstarError("archive is not canonical USTAR")
        if header[156:157] != b"0":
            raise BoundedUstarError(
                "USTAR PAX, GNU, sparse, link, directory, or special member rejected"
            )
        if any(header[157:257]) or any(header[345:512]):
            raise BoundedUstarError("USTAR extension or reserved header bytes rejected")
        stored_checksum = _octal(header[148:156], label="checksum")
        calculated_checksum = sum(header[:148]) + (8 * ord(" ")) + sum(header[156:])
        if stored_checksum != calculated_checksum:
            raise BoundedUstarError("USTAR header checksum mismatch")
        name = _text(header[0:100], label="name")
        _validate_member_name(name)
        if name in members or name not in policy.members:
            raise BoundedUstarError("USTAR member is duplicate or not allowlisted")
        rule = policy.members[name]
        mode = _octal(header[100:108], label="mode")
        size = _octal(header[124:136], label="size")
        if (
            mode not in rule.modes
            or _octal(header[108:116], label="uid") != 0
            or _octal(header[116:124], label="gid") != 0
            or _octal(header[136:148], label="mtime") != 0
            or _octal(header[329:337], label="devmajor") != 0
            or _octal(header[337:345], label="devminor") != 0
            or any(header[265:329])
        ):
            raise BoundedUstarError("USTAR member metadata is not canonical")
        if not rule.min_bytes <= size <= rule.max_bytes:
            raise BoundedUstarError("USTAR member size exceeds its predeclared cap")
        total_payload += size
        if total_payload > policy.max_total_payload_bytes:
            raise BoundedUstarError("USTAR aggregate payload exceeds its cap")
        data_offset = offset + BLOCK_BYTES
        padded_size = ((size + BLOCK_BYTES - 1) // BLOCK_BYTES) * BLOCK_BYTES
        next_offset = data_offset + padded_size
        if next_offset > archive_size:
            raise BoundedUstarError("USTAR member payload is truncated")
        padding_size = padded_size - size
        if padding_size and any(_pread_exact(descriptor, data_offset + size, padding_size)):
            raise BoundedUstarError("USTAR member has nonzero padding")
        members[name] = UstarMember(name, data_offset, size, mode)
        offset = next_offset
    if not found_terminator:
        raise BoundedUstarError("USTAR archive lacks its two-block terminator")
    while offset < archive_size:
        if _pread_exact(descriptor, offset, BLOCK_BYTES) != b"\0" * BLOCK_BYTES:
            raise BoundedUstarError("USTAR archive has nonzero trailing blocks")
        offset += BLOCK_BYTES
    if set(members) != set(policy.members):
        raise BoundedUstarError("USTAR archive inventory differs from exact allowlist")
    return members


def _validate_member_name(name: object) -> str:
    if not isinstance(name, str) or not name or len(name.encode("utf-8")) > 99:
        raise ValueError("bounded USTAR member name is invalid")
    pure = PurePosixPath(name)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or name != pure.as_posix()
        or "\x00" in name
        or name.startswith("./")
    ):
        raise ValueError("bounded USTAR member name is unsafe")
    return name


def _text(raw: bytes, *, label: str) -> str:
    value, separator, padding = raw.partition(b"\0")
    if not separator or any(padding):
        raise BoundedUstarError(f"USTAR {label} lacks canonical NUL padding")
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BoundedUstarError(f"USTAR {label} is not UTF-8") from error


def _octal(raw: bytes, *, label: str) -> int:
    if raw and raw[0] & 0x80:
        raise BoundedUstarError(f"USTAR {label} uses forbidden base-256 encoding")
    value = raw.rstrip(b"\0 ").lstrip(b" ")
    if not value:
        return 0
    if any(character not in b"01234567" for character in value):
        raise BoundedUstarError(f"USTAR {label} is not strict octal")
    return int(value, 8)


def _pread_exact(descriptor: int, offset: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.pread(descriptor, min(STREAM_CHUNK_BYTES, remaining), offset)
        if not chunk:
            raise BoundedUstarError("USTAR archive changed or truncated during read")
        chunks.append(chunk)
        offset += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
