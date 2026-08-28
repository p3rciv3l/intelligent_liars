"""Durable, exact-ID journals for restartable deterministic work.

The journal owns only persistence and identity checks.  Callers retain control of
the run contract and must semantically validate every recovered result.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar


ExactId = tuple[str, str]
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class JournalSchema:
    """Names that make one envelope usable by domain-specific journals."""

    row_format: str
    kind_field: str
    result_field: str
    run_identity_field: str
    label: str = "journal"


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def exact_id_filename(key: ExactId) -> str:
    """Return the non-revealing, collision-resistant filename for an exact ID."""

    return hashlib.sha256(f"{key[0]}\0{key[1]}".encode()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class ExactIdJournal:
    """Self-hashed, atomically persisted results keyed by two exact strings."""

    def __init__(
        self,
        root: Path,
        *,
        contract: Mapping[str, Any],
        run_identity_sha256: str,
        schema: JournalSchema,
        error_type: type[Exception] = ValueError,
    ) -> None:
        self.root = Path(root)
        self.rows_root = self.root / "rows"
        self.contract = dict(contract)
        self.run_identity_sha256 = run_identity_sha256
        self.schema = schema
        self.error_type = error_type
        if not _is_sha256(run_identity_sha256):
            self._raise("run identity must be a lowercase SHA-256 digest")
        self._initialize()

    def _raise(self, message: str, *, cause: Exception | None = None) -> None:
        error = self.error_type(message)
        if cause is None:
            raise error
        raise error from cause

    def _reject_symlink(self, path: Path, description: str) -> None:
        if path.is_symlink():
            self._raise(f"{self.schema.label} {description} must not be a symlink")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _atomic_write(self, path: Path, data: bytes) -> None:
        self._reject_symlink(path, path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            self._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _initialize(self) -> None:
        self._reject_symlink(self.root, "directory")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self._raise(f"{self.schema.label} directory is unusable", cause=error)
        self._reject_symlink(self.rows_root, "rows")
        try:
            self.rows_root.mkdir(exist_ok=True)
        except OSError as error:
            self._raise(f"{self.schema.label} rows directory is unusable", cause=error)
        contract_path = self.root / "contract.json"
        self._reject_symlink(contract_path, "contract")
        if contract_path.exists():
            if not contract_path.is_file():
                self._raise(f"{self.schema.label} contract must be regular")
            try:
                observed = json.loads(contract_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                self._raise(f"{self.schema.label} contract is unreadable", cause=error)
            if observed != self.contract:
                self._raise(f"{self.schema.label} contract differs")
        else:
            self._atomic_write(contract_path, canonical_json_bytes(self.contract))

    def read(
        self,
        expected_request_sha256: Mapping[ExactId, str],
        *,
        validate_result: Callable[[ExactId, dict[str, Any]], ResultT],
    ) -> dict[ExactId, ResultT]:
        """Read exact matching envelopes, then delegate semantic validation."""

        completed: dict[ExactId, ResultT] = {}
        for path in sorted(self.rows_root.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                self._raise(f"{self.schema.label} row must be regular")
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                self._raise(f"{self.schema.label} row is unreadable", cause=error)
            if (
                not isinstance(payload, dict)
                or payload.get("format") != self.schema.row_format
            ):
                self._raise(f"unsupported {self.schema.label} row")
            claimed = payload.get("content_sha256")
            if not _is_sha256(claimed):
                self._raise(
                    f"{self.schema.label} row content_sha256 must be a lowercase "
                    "SHA-256 digest"
                )
            unsigned = dict(payload)
            del unsigned["content_sha256"]
            if canonical_json_sha256(unsigned) != claimed:
                self._raise(f"{self.schema.label} row hash mismatch")
            key = (
                str(payload.get(self.schema.kind_field, "")),
                str(payload.get("record_id", "")),
            )
            result = payload.get(self.schema.result_field)
            if (
                key not in expected_request_sha256
                or key in completed
                or path.stem != exact_id_filename(key)
                or payload.get("request_sha256") != expected_request_sha256[key]
                or payload.get(self.schema.run_identity_field)
                != self.run_identity_sha256
                or not isinstance(result, dict)
            ):
                self._raise(f"{self.schema.label} row identity mismatch")
            completed[key] = validate_result(key, dict(result))
        return completed

    def append(
        self,
        key: ExactId,
        *,
        request_sha256: str,
        result: Mapping[str, Any],
    ) -> None:
        """Durably append one self-hashed result without overwriting an ID."""

        if not _is_sha256(request_sha256):
            self._raise("request_sha256 must be a lowercase SHA-256 digest")
        body = {
            "format": self.schema.row_format,
            self.schema.kind_field: key[0],
            "record_id": key[1],
            "request_sha256": request_sha256,
            self.schema.run_identity_field: self.run_identity_sha256,
            self.schema.result_field: dict(result),
        }
        payload = {**body, "content_sha256": canonical_json_sha256(body)}
        path = self.rows_root / f"{exact_id_filename(key)}.json"
        if path.exists() or path.is_symlink():
            self._raise(f"{self.schema.label} row already exists: {key[1]}")
        self._atomic_write(path, canonical_json_bytes(payload))
