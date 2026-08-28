"""Fail-closed hydration of one immutable production-input USTAR bundle."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.parse
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from intelligent_liars.bounded_ustar import BLOCK_BYTES, MAX_POLICY_ARCHIVE_BYTES


MANIFEST_FORMAT = "truth_editing_production_input_manifest_v1"
RECEIPT_FORMAT = "truth_editing_production_input_hydration_receipt_v1"
BUILD_ALLOWLIST_FORMAT = "truth_editing_production_input_build_allowlist_v1"
MAX_MEMBERS = 4096
MAX_TOTAL_PAYLOAD_BYTES = MAX_POLICY_ARCHIVE_BYTES - (
    (2 * MAX_MEMBERS + 20) * BLOCK_BYTES
)
_SHA256 = frozenset("0123456789abcdef")


class HydrationError(ValueError):
    """The manifest, bundle, or destination violates the frozen contract."""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HydrationError(f"manifest contains duplicate key: {key}")
        result[key] = value
    return result


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise HydrationError(f"{label} fields do not match the versioned contract")
    return value


def _size(value: Any, label: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise HydrationError(f"{label} must be an integer from 0 through {maximum}")
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise HydrationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise HydrationError(f"{label} must be a POSIX repo-relative file path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise HydrationError(f"{label} must be a safe repo-relative file path")
    return path


def _load_manifest(path: Path) -> tuple[Mapping[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise HydrationError("manifest must be a regular non-symlink file")
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_object_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HydrationError("manifest is not valid JSON") from error
    manifest = _exact_keys(value, {"format", "archive", "members"}, "manifest")
    if manifest["format"] != MANIFEST_FORMAT:
        raise HydrationError("manifest format is unsupported")
    return manifest, hashlib.sha256(raw).hexdigest()


def _write_new_or_verify(path: Path, payload: bytes) -> None:
    absolute = path.absolute().resolve(strict=False)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    directory = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:-1]:
            try:
                os.mkdir(part, 0o755, dir_fd=directory)
            except FileExistsError:
                pass
            next_directory = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = next_directory

        def verify_existing() -> bool:
            try:
                descriptor = os.open(
                    absolute.name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory,
                )
            except FileNotFoundError:
                return False
            except OSError as error:
                raise HydrationError(
                    f"refusing unsafe existing output: {absolute}"
                ) from error
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise HydrationError(f"output is not a regular file: {absolute}")
                existing = bytearray()
                while chunk := os.read(descriptor, 1024 * 1024):
                    existing.extend(chunk)
                    if len(existing) > len(payload):
                        break
                if bytes(existing) != payload:
                    raise HydrationError(
                        f"refusing to overwrite non-identical output: {absolute}"
                    )
                return True
            finally:
                os.close(descriptor)

        if verify_existing():
            return
        temporary_name = f".{absolute.name}.{os.getpid()}.part"
        try:
            os.unlink(temporary_name, dir_fd=directory)
        except FileNotFoundError:
            pass
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
            dir_fd=directory,
        )
        try:
            pending = memoryview(payload)
            while pending:
                count = os.write(descriptor, pending)
                if count <= 0:
                    raise HydrationError("receipt output made a short write")
                pending = pending[count:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary_name,
                absolute.name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        except FileExistsError:
            verify_existing()
        os.fsync(directory)
    finally:
        try:
            os.unlink(f".{absolute.name}.{os.getpid()}.part", dir_fd=directory)
        except FileNotFoundError:
            pass
        os.close(directory)


def _publish_build_pair(
    archive_stage: Path,
    archive_path: Path,
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    manifest_path: Path,
    manifest_bytes: bytes,
) -> None:
    if archive_path == manifest_path:
        raise HydrationError("archive and manifest outputs must differ")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    archive_exists = archive_path.exists() or archive_path.is_symlink()
    manifest_exists = manifest_path.exists() or manifest_path.is_symlink()
    if archive_exists and (
        archive_path.is_symlink()
        or not archive_path.is_file()
        or archive_path.stat().st_size != archive_size_bytes
        or _hash_file(archive_path) != archive_sha256
    ):
        raise HydrationError(f"refusing to overwrite non-identical output: {archive_path}")
    if manifest_exists and (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.read_bytes() != manifest_bytes
    ):
        raise HydrationError(f"refusing to overwrite non-identical output: {manifest_path}")
    manifest_stage = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.part")
    manifest_stage.unlink(missing_ok=True)
    installed_archive = False
    archive_identity: tuple[int, int] | None = None
    try:
        manifest_stage.write_bytes(manifest_bytes)
        if not archive_exists:
            os.link(archive_stage, archive_path, follow_symlinks=False)
            metadata = archive_path.stat(follow_symlinks=False)
            archive_identity = (metadata.st_dev, metadata.st_ino)
            installed_archive = True
        if not manifest_exists:
            os.link(manifest_stage, manifest_path, follow_symlinks=False)
    except BaseException:
        if installed_archive and archive_identity is not None:
            try:
                current = archive_path.stat(follow_symlinks=False)
                if (current.st_dev, current.st_ino) == archive_identity:
                    archive_path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        manifest_stage.unlink(missing_ok=True)


def _load_build_allowlist(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise HydrationError("build allowlist must be a regular non-symlink file")
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(path.read_bytes(), object_pairs_hook=_object_pairs)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HydrationError("build allowlist is not valid JSON") from error
        document = _exact_keys(value, {"format", "files"}, "build allowlist")
        if document["format"] != BUILD_ALLOWLIST_FORMAT or not isinstance(
            document["files"], list
        ):
            raise HydrationError("build allowlist format is unsupported")
        entries: list[dict[str, str]] = []
        for index, raw in enumerate(document["files"]):
            item = _exact_keys(
                raw,
                {"source_path", "archive_path", "destination_path"},
                f"build allowlist files[{index}]",
            )
            entries.append(
                {
                    key: str(_relative(item[key], f"build allowlist files[{index}].{key}"))
                    for key in item
                }
            )
        return entries
    try:
        lines = path.read_text().splitlines()
    except UnicodeDecodeError as error:
        raise HydrationError("newline build allowlist must be UTF-8") from error
    if not lines or any(not line or line != line.strip() for line in lines):
        raise HydrationError("newline build allowlist must contain exact non-empty paths")
    return [
        {"source_path": line, "archive_path": line, "destination_path": line}
        for line in lines
    ]


def build_production_input_bundle(
    allowlist_path: Path | str,
    *,
    source_root: Path | str,
    archive_path: Path | str,
    manifest_path: Path | str,
    archive_uri: str | None = None,
) -> dict[str, Any]:
    """Build canonical gzip/USTAR bytes from an explicit path allowlist."""

    return build_production_input_bundle_from_entries(
        _load_build_allowlist(Path(allowlist_path).absolute()),
        source_root=source_root,
        archive_path=archive_path,
        manifest_path=manifest_path,
        archive_uri=archive_uri,
    )


def _open_relative_regular(root: Path, relative: PurePosixPath) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    directory = os.open(root, flags)
    try:
        for part in relative.parts[:-1]:
            next_directory = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        descriptor = os.open(
            relative.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory,
        )
    except OSError as error:
        raise HydrationError(f"source path is not safely readable: {relative}") from error
    finally:
        os.close(directory)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise HydrationError(f"source path is not a regular file: {relative}")
    return descriptor


class _HashingReader:
    def __init__(self, descriptor: int) -> None:
        self._descriptor = descriptor
        self._digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = 1024 * 1024
        chunk = os.read(self._descriptor, size)
        self._digest.update(chunk)
        self.bytes_read += len(chunk)
        return chunk

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def build_production_input_bundle_from_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    source_root: Path | str,
    archive_path: Path | str,
    manifest_path: Path | str,
    archive_uri: str | None = None,
) -> dict[str, Any]:
    """Build from already-derived entries, such as an ignored Vast path set."""

    root = Path(source_root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise HydrationError("source root must be an existing non-symlink directory")
    if not 1 <= len(entries) <= MAX_MEMBERS:
        raise HydrationError("build entries exceed the exact member-count bound")
    normalized: list[tuple[str, str, PurePosixPath]] = []
    seen_archive: set[str] = set()
    seen_destination: set[str] = set()
    for index, raw in enumerate(entries):
        item = _exact_keys(
            raw,
            {"source_path", "archive_path", "destination_path"},
            f"build entries[{index}]",
        )
        source_rel = _relative(item["source_path"], f"build entries[{index}].source_path")
        archive_rel = _relative(item["archive_path"], f"build entries[{index}].archive_path")
        destination_rel = _relative(
            item["destination_path"], f"build entries[{index}].destination_path"
        )
        if str(archive_rel) in seen_archive or str(destination_rel) in seen_destination:
            raise HydrationError("build archive and destination paths must be unique")
        seen_archive.add(str(archive_rel))
        seen_destination.add(str(destination_rel))
        normalized.append((str(archive_rel), str(destination_rel), source_rel))
    normalized.sort(key=lambda item: item[0].encode("utf-8"))

    tar_buffer = tempfile.SpooledTemporaryFile(max_size=32 * 1024 * 1024)
    built: list[tuple[str, str, int, str]] = []
    total = 0
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for archive_name, destination, source_rel in normalized:
            descriptor = _open_relative_regular(root, source_rel)
            try:
                size_bytes = os.fstat(descriptor).st_size
                total += size_bytes
                if total > MAX_TOTAL_PAYLOAD_BYTES:
                    raise HydrationError("build payload exceeds the aggregate byte cap")
                info = tarfile.TarInfo(archive_name)
                info.size = size_bytes
                info.mode = 0o644
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                hashing_stream = _HashingReader(descriptor)
                archive.addfile(info, hashing_stream)
                if hashing_stream.bytes_read != size_bytes:
                    raise HydrationError("build source changed during archive construction")
                built.append(
                    (archive_name, destination, size_bytes, hashing_stream.hexdigest())
                )
            finally:
                os.close(descriptor)
    output_archive = Path(archive_path).absolute().resolve(strict=False)
    output_manifest = Path(manifest_path).absolute().resolve(strict=False)
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    descriptor, compressed_name = tempfile.mkstemp(
        prefix=f".{output_archive.name}.", suffix=".part", dir=output_archive.parent
    )
    compressed_stage = Path(compressed_name)
    try:
        tar_buffer.seek(0)
        with os.fdopen(descriptor, "wb") as raw_output, gzip.GzipFile(
            fileobj=raw_output,
            mode="wb",
            filename="",
            mtime=0,
            compresslevel=9,
        ) as compressed:
            shutil.copyfileobj(tar_buffer, compressed, length=1024 * 1024)
        archive_size_bytes = compressed_stage.stat().st_size
        if archive_size_bytes > MAX_POLICY_ARCHIVE_BYTES:
            raise HydrationError("compressed build archive exceeds the byte cap")
        archive_sha256 = _hash_file(compressed_stage)
    except BaseException:
        compressed_stage.unlink(missing_ok=True)
        raise
    uri = archive_uri
    if uri is None:
        uri = os.path.relpath(output_archive, output_manifest.parent)
    manifest: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "archive": {
            "uri": uri,
            "sha256": archive_sha256,
            "size_bytes": archive_size_bytes,
            "compression": "gzip",
            "archive_format": "ustar",
        },
        "members": [
            {
                "archive_path": archive_name,
                "destination_path": destination,
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
            for archive_name, destination, size_bytes, sha256 in built
        ],
    }
    _validate_manifest(manifest)
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    try:
        _publish_build_pair(
            compressed_stage,
            output_archive,
            archive_sha256=archive_sha256,
            archive_size_bytes=archive_size_bytes,
            manifest_path=output_manifest,
            manifest_bytes=manifest_bytes,
        )
    finally:
        compressed_stage.unlink(missing_ok=True)
    return manifest


def entries_from_vast_job_config(
    config_path: Path | str,
    *,
    repo_root: Path | str,
    ignored_only: bool = False,
) -> list[dict[str, str]]:
    """Derive deterministic members from ``base_job.bundle_paths``."""

    config_file = Path(config_path).absolute()
    if config_file.is_symlink() or not config_file.is_file():
        raise HydrationError("Vast job config must be a regular non-symlink file")
    try:
        value = json.loads(config_file.read_bytes(), object_pairs_hook=_object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HydrationError("Vast job config is not valid JSON") from error
    if not isinstance(value, Mapping) or not isinstance(value.get("base_job"), Mapping):
        raise HydrationError("Vast job config lacks base_job")
    raw_paths = value["base_job"].get("bundle_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise HydrationError("Vast job config lacks a non-empty bundle_paths list")
    paths = [str(_relative(item, "base_job.bundle_paths item")) for item in raw_paths]
    if len(paths) != len(set(paths)):
        raise HydrationError("base_job.bundle_paths contains duplicates")
    if ignored_only:
        try:
            completed = subprocess.run(
                ["git", "check-ignore", "--stdin"],
                cwd=Path(repo_root).absolute(),
                input="".join(f"{path}\n" for path in paths),
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as error:
            raise HydrationError("cannot run git check-ignore") from error
        if completed.returncode not in {0, 1}:
            raise HydrationError("git check-ignore failed while deriving build inputs")
        ignored = set(completed.stdout.splitlines())
        if not ignored <= set(paths):
            raise HydrationError("git check-ignore returned an unexpected path")
        paths = [path for path in paths if path in ignored]
        if not paths:
            raise HydrationError("no ignored bundle paths were selected")
    result: list[dict[str, str]] = []
    for path in sorted(paths, key=lambda item: item.encode("utf-8")):
        archive_name = f"members/{hashlib.sha256(path.encode('utf-8')).hexdigest()}"
        result.append(
            {
                "source_path": path,
                "archive_path": archive_name,
                "destination_path": path,
            }
        )
    return result


def _validate_manifest(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    archive = dict(
        _exact_keys(
            manifest["archive"],
            {"uri", "sha256", "size_bytes", "compression", "archive_format"},
            "archive",
        )
    )
    if not isinstance(archive["uri"], str) or not archive["uri"]:
        raise HydrationError("archive URI must be non-empty")
    parsed_uri = urllib.parse.urlsplit(archive["uri"])
    if parsed_uri.scheme not in {"", "file", "s3"}:
        raise HydrationError("archive URI must use s3://, file://, or a local path")
    archive["sha256"] = _sha(archive["sha256"], "archive.sha256")
    archive["size_bytes"] = _size(
        archive["size_bytes"], "archive.size_bytes", maximum=MAX_POLICY_ARCHIVE_BYTES
    )
    if archive["size_bytes"] == 0:
        raise HydrationError("archive may not be empty")
    if archive["compression"] != "gzip" or archive["archive_format"] != "ustar":
        raise HydrationError("archive must be a gzip-compressed USTAR archive")
    raw_members = manifest["members"]
    if not isinstance(raw_members, list) or not 1 <= len(raw_members) <= MAX_MEMBERS:
        raise HydrationError("members must be a non-empty bounded list")
    members: list[dict[str, Any]] = []
    archive_names: set[str] = set()
    destinations: set[str] = set()
    for index, raw in enumerate(raw_members):
        member = dict(
            _exact_keys(
                raw,
                {"archive_path", "destination_path", "sha256", "size_bytes"},
                f"members[{index}]",
            )
        )
        archive_path = _relative(member["archive_path"], f"members[{index}].archive_path")
        destination = _relative(
            member["destination_path"], f"members[{index}].destination_path"
        )
        if str(archive_path) in archive_names or str(destination) in destinations:
            raise HydrationError("archive paths and destination paths must be unique")
        archive_names.add(str(archive_path))
        destinations.add(str(destination))
        members.append(
            {
                "archive_path": str(archive_path),
                "destination_path": str(destination),
                "sha256": _sha(member["sha256"], f"members[{index}].sha256"),
                "size_bytes": _size(
                    member["size_bytes"],
                    f"members[{index}].size_bytes",
                    maximum=MAX_TOTAL_PAYLOAD_BYTES,
                ),
            }
        )
    for paths, label in (
        (archive_names, "archive"),
        (destinations, "destination"),
    ):
        parts = [PurePosixPath(path).parts for path in paths]
        if any(
            left != right
            and len(left) < len(right)
            and right[: len(left)] == left
            for left in parts
            for right in parts
        ):
            raise HydrationError(f"{label} paths contain an ancestor collision")
    if sum(member["size_bytes"] for member in members) > MAX_TOTAL_PAYLOAD_BYTES:
        raise HydrationError("aggregate member payload exceeds the hydration cap")
    return archive, members


def _bounded_copy(source: Any, destination: Path, *, exact_bytes: int) -> None:
    written = 0
    with destination.open("xb") as output:
        while chunk := source.read(min(1024 * 1024, exact_bytes - written + 1)):
            written += len(chunk)
            if written > exact_bytes:
                raise HydrationError("archive exceeds its declared size")
            output.write(chunk)
    if written != exact_bytes:
        raise HydrationError("archive is shorter than its declared size")


def _fetch_archive(
    uri: str,
    destination: Path,
    *,
    manifest_dir: Path,
    s3_client: Any,
    exact_bytes: int,
) -> None:
    parsed = urllib.parse.urlsplit(uri)
    if parsed.scheme == "s3":
        if parsed.username or parsed.password or parsed.port or not parsed.netloc:
            raise HydrationError("invalid S3 archive URI")
        key = parsed.path[1:]
        if not key or parsed.query or parsed.fragment:
            raise HydrationError("invalid S3 archive URI")
        client = s3_client
        if client is None:
            try:
                import boto3
            except ImportError as error:  # pragma: no cover - optional production dependency
                raise HydrationError("boto3 is required for S3 hydration") from error
            client = boto3.client("s3")
        try:
            response = client.get_object(Bucket=parsed.netloc, Key=key)
        except Exception as error:
            raise HydrationError("S3 archive download failed") from error
        if response.get("ContentLength") != exact_bytes or "Body" not in response:
            raise HydrationError("S3 archive size differs from the manifest")
        try:
            _bounded_copy(response["Body"], destination, exact_bytes=exact_bytes)
        except Exception as error:
            if isinstance(error, HydrationError):
                raise
            raise HydrationError("S3 archive download failed") from error
        if destination.is_symlink() or not destination.is_file():
            raise HydrationError("S3 download did not produce a regular file")
        return
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
            raise HydrationError("invalid local file URI")
        source = Path(urllib.parse.unquote(parsed.path))
    elif parsed.scheme:
        raise HydrationError("archive URI must use s3://, file://, or a local path")
    else:
        source = Path(uri)
        if not source.is_absolute():
            source = manifest_dir / source
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise HydrationError("local archive must be a regular non-symlink file") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise HydrationError("local archive must be a regular non-symlink file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            _bounded_copy(stream, destination, exact_bytes=exact_bytes)
    finally:
        os.close(descriptor)


def _reject_symlinks(root: Path, relative: PurePosixPath) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise HydrationError("repo root must be an existing non-symlink directory")
    current = root.resolve(strict=True)
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise HydrationError("destination may not traverse a symlink")
        if current.exists() and not current.is_dir():
            raise HydrationError("destination parent must be a directory")
    target = current / relative.name
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise HydrationError("destination must be a regular file")
    return target


def _verify_existing(target: Path, member: Mapping[str, Any]) -> bool:
    if not target.exists():
        return False
    if target.stat().st_size != member["size_bytes"] or _hash_file(target) != member["sha256"]:
        raise HydrationError(f"existing destination has the wrong identity: {member['destination_path']}")
    return True


def _open_or_create_parent(root: Path, relative: PurePosixPath) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    directory = os.open(root, flags)
    try:
        for part in relative.parts[:-1]:
            try:
                os.mkdir(part, 0o755, dir_fd=directory)
            except FileExistsError:
                pass
            next_directory = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        return directory
    except BaseException:
        os.close(directory)
        raise


def _rollback_installed(
    root: Path, installed: list[tuple[PurePosixPath, tuple[int, int]]]
) -> None:
    for relative, identity in reversed(installed):
        try:
            parent = _open_or_create_parent(root, relative)
            try:
                metadata = os.stat(relative.name, dir_fd=parent, follow_symlinks=False)
                if (metadata.st_dev, metadata.st_ino) == identity:
                    os.unlink(relative.name, dir_fd=parent)
            finally:
                os.close(parent)
        except (FileNotFoundError, OSError):
            continue


def _strict_octal(raw: bytes, label: str) -> int:
    if raw and raw[0] & 0x80:
        raise HydrationError(f"USTAR {label} uses forbidden base-256 encoding")
    value = raw.rstrip(b"\0 ").lstrip(b" ")
    if not value:
        return 0
    if any(character not in b"01234567" for character in value):
        raise HydrationError(f"USTAR {label} is not strict octal")
    return int(value, 8)


def _strict_text(raw: bytes, label: str) -> str:
    value, separator, padding = raw.partition(b"\0")
    if not separator or any(padding):
        raise HydrationError(f"USTAR {label} lacks canonical NUL padding")
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HydrationError(f"USTAR {label} is not UTF-8") from error


def _pread(descriptor: int, offset: int, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - len(output)), offset + len(output))
        if not chunk:
            raise HydrationError("USTAR archive is truncated")
        output.extend(chunk)
    return bytes(output)


def _scan_exact_ustar(
    path: Path, members: list[dict[str, Any]], *, archive_cap: int
) -> dict[str, tuple[int, int]]:
    """Return payload extents after scanning every physical USTAR header."""

    expected = {member["archive_path"]: member for member in members}
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise HydrationError("cannot safely open decompressed USTAR") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > archive_cap
            or metadata.st_size % BLOCK_BYTES
        ):
            raise HydrationError("USTAR size is outside its exact bound")
        offset = 0
        found: dict[str, tuple[int, int]] = {}
        terminated = False
        while offset < metadata.st_size:
            header = _pread(descriptor, offset, BLOCK_BYTES)
            if header == b"\0" * BLOCK_BYTES:
                if _pread(descriptor, offset + BLOCK_BYTES, BLOCK_BYTES) != b"\0" * BLOCK_BYTES:
                    raise HydrationError("USTAR requires two zero terminator blocks")
                offset += 2 * BLOCK_BYTES
                terminated = True
                break
            if len(found) >= MAX_MEMBERS:
                raise HydrationError("USTAR exceeds the member-count bound")
            if header[257:263] != b"ustar\0" or header[263:265] != b"00":
                raise HydrationError("archive is not canonical USTAR")
            if header[156:157] != b"0":
                raise HydrationError("USTAR links, directories, extensions, and special members are forbidden")
            if any(header[157:257]) or any(header[265:329]) or any(header[500:512]):
                raise HydrationError("USTAR link, owner, group, or reserved metadata is forbidden")
            stored = _strict_octal(header[148:156], "checksum")
            calculated = sum(header[:148]) + 8 * ord(" ") + sum(header[156:])
            if stored != calculated:
                raise HydrationError("USTAR header checksum mismatch")
            name = _strict_text(header[0:100], "name")
            prefix = _strict_text(header[345:500], "prefix") if any(header[345:500]) else ""
            full_name = f"{prefix}/{name}" if prefix else name
            safe_name = str(_relative(full_name, "USTAR member name"))
            if safe_name in found or safe_name not in expected:
                raise HydrationError("USTAR member is duplicate, extra, or not allowlisted")
            size = _strict_octal(header[124:136], "size")
            if (
                size != expected[safe_name]["size_bytes"]
                or _strict_octal(header[100:108], "mode") != 0o644
                or _strict_octal(header[108:116], "uid") != 0
                or _strict_octal(header[116:124], "gid") != 0
                or _strict_octal(header[136:148], "mtime") != 0
                or _strict_octal(header[329:337], "devmajor") != 0
                or _strict_octal(header[337:345], "devminor") != 0
            ):
                raise HydrationError("USTAR member metadata or size differs from the manifest")
            payload_offset = offset + BLOCK_BYTES
            padded = ((size + BLOCK_BYTES - 1) // BLOCK_BYTES) * BLOCK_BYTES
            next_offset = payload_offset + padded
            if next_offset > metadata.st_size:
                raise HydrationError("USTAR member payload is truncated")
            if padded > size and any(_pread(descriptor, payload_offset + size, padded - size)):
                raise HydrationError("USTAR member padding is nonzero")
            found[safe_name] = (payload_offset, size)
            offset = next_offset
        if not terminated:
            raise HydrationError("USTAR archive lacks its terminator")
        while offset < metadata.st_size:
            if _pread(descriptor, offset, BLOCK_BYTES) != b"\0" * BLOCK_BYTES:
                raise HydrationError("USTAR archive has nonzero trailing blocks")
            offset += BLOCK_BYTES
        if set(found) != set(expected):
            raise HydrationError("USTAR inventory differs from the exact manifest allowlist")
        return found
    finally:
        os.close(descriptor)


def _copy_extent_and_hash(
    archive_path: Path, offset: int, size: int, destination: Path
) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(archive_path, flags)
    try:
        with destination.open("xb") as output:
            remaining = size
            while remaining:
                chunk = os.pread(descriptor, min(1024 * 1024, remaining), offset)
                if not chunk:
                    raise HydrationError("USTAR changed or truncated during member copy")
                output.write(chunk)
                digest.update(chunk)
                offset += len(chunk)
                remaining -= len(chunk)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def hydrate_production_inputs(
    manifest_path: Path | str,
    *,
    repo_root: Path | str,
    s3_client: Any = None,
) -> dict[str, Any]:
    """Hydrate or verify every exact member named by a committed manifest."""

    manifest_file = Path(manifest_path).absolute()
    root = Path(repo_root).absolute()
    manifest, manifest_sha256 = _load_manifest(manifest_file)
    archive_spec, members = _validate_manifest(manifest)
    destinations = [
        _reject_symlinks(root, PurePosixPath(member["destination_path"]))
        for member in members
    ]
    existing = [_verify_existing(target, member) for target, member in zip(destinations, members)]
    if all(existing):
        return _receipt(manifest_sha256, archive_spec, members, "verified_existing")

    with tempfile.TemporaryDirectory(prefix=".truth-editing-hydration-", dir=root) as temporary:
        stage = Path(temporary)
        compressed_path = stage / "bundle.tar.gz"
        _fetch_archive(
            archive_spec["uri"],
            compressed_path,
            manifest_dir=manifest_file.parent,
            s3_client=s3_client,
            exact_bytes=archive_spec["size_bytes"],
        )
        if (
            compressed_path.stat().st_size != archive_spec["size_bytes"]
            or _hash_file(compressed_path) != archive_spec["sha256"]
        ):
            raise HydrationError("archive hash/size mismatch")

        padded_payload = sum(
            ((member["size_bytes"] + BLOCK_BYTES - 1) // BLOCK_BYTES) * BLOCK_BYTES
            for member in members
        )
        ustar_cap = min(
            MAX_POLICY_ARCHIVE_BYTES,
            max(20 * BLOCK_BYTES, padded_payload + (len(members) + 20) * BLOCK_BYTES),
        )
        tar_path = stage / "bundle.tar"
        total = 0
        try:
            with gzip.open(compressed_path, "rb") as source, tar_path.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > ustar_cap:
                        raise HydrationError("decompressed USTAR exceeds its exact bound")
                    output.write(chunk)
        except (OSError, EOFError) as error:
            raise HydrationError("archive is not a valid gzip stream") from error

        extracted = stage / "members"
        extracted.mkdir()
        extents = _scan_exact_ustar(tar_path, members, archive_cap=ustar_cap)
        for index, member in enumerate(members):
            offset, size = extents[member["archive_path"]]
            digest = _copy_extent_and_hash(tar_path, offset, size, extracted / str(index))
            if digest != member["sha256"]:
                raise HydrationError(f"member hash mismatch: {member['archive_path']}")

        # Identity of every payload is now established. Create every destination
        # through no-follow directory descriptors and roll back the whole new set
        # if any create fails.
        installed: list[tuple[PurePosixPath, tuple[int, int]]] = []
        try:
            for index, (member, was_existing) in enumerate(zip(members, existing)):
                if was_existing:
                    continue
                relative = PurePosixPath(member["destination_path"])
                parent = _open_or_create_parent(root, relative)
                try:
                    os.link(
                        extracted / str(index),
                        relative.name,
                        dst_dir_fd=parent,
                        follow_symlinks=False,
                    )
                    metadata = os.stat(
                        relative.name, dir_fd=parent, follow_symlinks=False
                    )
                    os.chmod(
                        relative.name,
                        0o644,
                        dir_fd=parent,
                        follow_symlinks=False,
                    )
                    installed.append((relative, (metadata.st_dev, metadata.st_ino)))
                finally:
                    os.close(parent)
        except BaseException as error:
            _rollback_installed(root, installed)
            if isinstance(error, HydrationError):
                raise
            raise HydrationError("atomic destination installation failed") from error

    return _receipt(manifest_sha256, archive_spec, members, "hydrated")


def _receipt(
    manifest_sha256: str,
    archive: Mapping[str, Any],
    members: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    identity_members = [
        {
            "archive_path": member["archive_path"],
            "destination_path": member["destination_path"],
            "sha256": member["sha256"],
            "size_bytes": member["size_bytes"],
        }
        for member in members
    ]
    receipt = {
        "format": RECEIPT_FORMAT,
        "status": status,
        "manifest_sha256": manifest_sha256,
        "archive_sha256": archive["sha256"],
        "archive_size_bytes": archive["size_bytes"],
        "members": identity_members,
    }
    receipt["content_sha256"] = _canonical_hash(receipt)
    return receipt


def durable_hydration_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize operational first-run/rerun status into one durable identity."""

    receipt = dict(value)
    if receipt.get("format") != RECEIPT_FORMAT or receipt.get("status") not in {
        "hydrated",
        "verified_existing",
        "verified",
    }:
        raise HydrationError("hydration receipt cannot be normalized")
    receipt["status"] = "verified"
    receipt.pop("content_sha256", None)
    receipt["content_sha256"] = _canonical_hash(receipt)
    return receipt


def write_durable_hydration_receipt(path: Path | str, value: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically create or byte-verify the stable hydration receipt."""

    receipt = durable_hydration_receipt(value)
    payload = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    _write_new_or_verify(Path(path).absolute(), payload)
    return receipt
