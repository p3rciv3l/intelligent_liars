"""Exact-key OSWorld source materialization for preservation-only KL.

This module deliberately has no S3 listing operation.  A verified role ledger
decides which tasks are eligible, and the hash-pinned final task map supplies
the only admissible checkpoint prefixes.  For each eligible task the reader
downloads exactly one trajectory and the initial screenshot named and hashed
inside that trajectory.  Evaluators, recordings, logs, and capability-test
tasks are outside the interface.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import platform
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast
from urllib.parse import urlparse

from .truth_editing_osworld_roles import FIT_TIER_COUNTS


OSWORLD_PRESERVATION_SOURCE_BUILD_CONFIG_FORMAT = (
    "truth_editing_osworld_preservation_source_build_config_v1"
)
OSWORLD_PRESERVATION_SOURCE_BUILD_CONFIG_V2_FORMAT = (
    "truth_editing_osworld_preservation_source_build_config_v2"
)
OSWORLD_PRESERVATION_SOURCE_FORMAT = "truth_editing_recorded_computer_use_source_v2"
OSWORLD_PRESERVATION_SOURCE_RECEIPT_FORMAT = (
    "truth_editing_osworld_preservation_source_receipt_v3"
)
_SOURCE_RECEIPT_V2_FORMAT = "truth_editing_osworld_preservation_source_receipt_v2"
_HEX = frozenset("0123456789abcdef")
_MAX_TASK_MAP_BYTES = 8 * 1024 * 1024
_MAX_TRAJECTORY_BYTES = 32 * 1024 * 1024
_MAX_SCREENSHOT_BYTES = 20 * 1024 * 1024
_OBSERVATION_KL_ASSISTANT_PROBE = "Continue."


class OSWorldPreservationSourceError(RuntimeError):
    """OSWorld preservation input identity or eligibility failed closed."""


class _S3Client(Protocol):
    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]: ...


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise OSWorldPreservationSourceError("value is not canonical JSON") from error


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise OSWorldPreservationSourceError(f"file is unreadable: {path}") from error
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    return _hash_bytes(_canonical_bytes(value))


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OSWorldPreservationSourceError(f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise OSWorldPreservationSourceError(f"{name} must be an array")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise OSWorldPreservationSourceError(
            f"{name} fields differ; missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise OSWorldPreservationSourceError(f"{name} must be nonempty trimmed text")
    return value


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise OSWorldPreservationSourceError(f"{name} must be a lowercase SHA-256")
    return value


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OSWorldPreservationSourceError(f"{name} must be a regular non-symlink file")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), name)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OSWorldPreservationSourceError(f"{name} is not strict JSON") from error


def _read_s3(client: _S3Client, bucket: str, key: str, *, limit: int, name: str) -> bytes:
    try:
        response = _object(client.get_object(Bucket=bucket, Key=key), f"{name} response")
        length = response.get("ContentLength")
        if isinstance(length, bool) or not isinstance(length, int) or length < 1 or length > limit:
            raise OSWorldPreservationSourceError(f"{name} size is outside its strict limit")
        body = response.get("Body")
        if body is None or not hasattr(body, "read"):
            raise OSWorldPreservationSourceError(f"{name} response body is unreadable")
        payload = body.read(limit + 1)
    except OSWorldPreservationSourceError:
        raise
    except Exception as error:
        raise OSWorldPreservationSourceError(f"exact-key fetch failed for {name}") from error
    if not isinstance(payload, bytes) or len(payload) != length or len(payload) > limit:
        raise OSWorldPreservationSourceError(f"{name} response length differs")
    return payload


def _safe_relative(value: Any, name: str) -> PurePosixPath:
    text = _text(value, name)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise OSWorldPreservationSourceError(f"{name} is not a safe relative key")
    return path


def _checkpoint_key(
    value: Any, *, bucket: str, task_id: str, attempt: int
) -> tuple[str, str]:
    uri = _text(value, "checkpoint prefix")
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or parsed.netloc != bucket or parsed.params or parsed.query or parsed.fragment:
        raise OSWorldPreservationSourceError("checkpoint prefix bucket identity differs")
    key = parsed.path.lstrip("/")
    task_slug = task_id.replace("/", "_")
    parts = PurePosixPath(key).parts
    if len(parts) < 7 or parts[0] != "runs" or parts[2] != "tasks":
        raise OSWorldPreservationSourceError("checkpoint prefix run identity differs")
    checkpoint_run_id = _text(parts[1], "checkpoint run ID")
    expected_start = f"runs/{checkpoint_run_id}/tasks/{task_slug}-"
    expected_attempt = f"/attempt-{attempt:04d}/checkpoints/"
    if not key.startswith(expected_start) or expected_attempt not in key or not key.endswith("/"):
        raise OSWorldPreservationSourceError("checkpoint prefix task identity differs")
    path = PurePosixPath(key)
    if ".." in path.parts:
        raise OSWorldPreservationSourceError("checkpoint prefix is unsafe")
    return key, checkpoint_run_id


def _trajectory_rows(payload: bytes, *, task_id: str, run_id: str, attempt: int) -> tuple[str, str, str, str]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise OSWorldPreservationSourceError("trajectory is not UTF-8 JSONL") from error
    if len(lines) < 2 or any(not line or line.strip() != line for line in lines):
        raise OSWorldPreservationSourceError("trajectory is not canonical nonempty JSONL")
    try:
        rows = [_object(json.loads(line), f"trajectory line {index + 1}") for index, line in enumerate(lines)]
    except json.JSONDecodeError as error:
        raise OSWorldPreservationSourceError("trajectory is not strict JSONL") from error
    started = rows[0]
    if (
        started.get("event") != "attempt_started"
        or started.get("task_id") != task_id
        or started.get("run_id") != run_id
        or started.get("attempt") != attempt
    ):
        raise OSWorldPreservationSourceError("trajectory task identity differs")
    instruction_value = started.get("instruction")
    if not isinstance(instruction_value, str) or not instruction_value.strip():
        raise OSWorldPreservationSourceError("trajectory instruction must be nonempty text")
    instruction = instruction_value.strip()
    observations = [row for row in rows if row.get("event") == "initial_observation"]
    if len(observations) != 1:
        raise OSWorldPreservationSourceError("trajectory requires one initial observation")
    observation = observations[0]
    screenshot = str(_safe_relative(observation.get("screenshot"), "initial screenshot"))
    screenshot_sha = _sha(observation.get("screenshot_sha256"), "initial screenshot SHA")
    return instruction, screenshot, screenshot_sha, _OBSERVATION_KL_ASSISTANT_PROBE


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rename_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if platform.system() == "Darwin" and hasattr(libc, "renamex_np"):
        result = libc.renamex_np(os.fsencode(source), os.fsencode(destination), ctypes.c_uint(4))
    elif platform.system() == "Linux" and hasattr(libc, "renameat2"):
        result = libc.renameat2(
            ctypes.c_int(-100), os.fsencode(source), ctypes.c_int(-100),
            os.fsencode(destination), ctypes.c_uint(1),
        )
    else:  # pragma: no cover
        raise OSWorldPreservationSourceError("atomic no-replace publication is unsupported")
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise OSWorldPreservationSourceError(f"output already exists: {destination}")
    raise OSError(number, os.strerror(number), str(destination))


def _output_paths(output_dir: Path | str) -> tuple[Path, Path]:
    requested = Path(output_dir).expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    output = requested.parent.resolve() / requested.name
    if os.path.lexists(output):
        raise OSWorldPreservationSourceError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    return output, staging


def _client(region: str) -> _S3Client:
    try:
        import boto3
    except ImportError as error:  # pragma: no cover - exercised by production CLI
        raise OSWorldPreservationSourceError("boto3 is required for S3 materialization") from error
    return cast(_S3Client, boto3.client("s3", region_name=region))


def _open_bound_role_artifacts(
    ledger_path: Path, optuna_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open already-built artifacts against explicit caller-supplied file hashes.

    The source builder is not a second partition builder.  It verifies the
    immutable artifact identities, membership, and redaction boundary; the
    authoritative OSWorld checkout verification belongs to role-ledger build.
    """

    ledger = dict(_load_json(ledger_path, "OSWorld role ledger"))
    _exact(
        ledger,
        {"format", "partition_policy", "selection_seed", "source", "role_counts", "tasks", "ledger_id"},
        "OSWorld role ledger",
    )
    if ledger["format"] not in {
        "truth_editing_osworld_role_ledger_v1",
        "truth_editing_osworld_role_ledger_v2",
    }:
        raise OSWorldPreservationSourceError("OSWorld role ledger format is unsupported")
    source = _object(ledger["source"], "role ledger source")
    _sha(source.get("task_hashes_sha256"), "role ledger task hashes SHA")
    osworld_commit = _text(source.get("osworld_commit"), "role ledger OSWorld commit")
    if len(osworld_commit) != 40 or any(character not in _HEX for character in osworld_commit):
        raise OSWorldPreservationSourceError("role ledger OSWorld commit is invalid")
    tasks = _array(ledger["tasks"], "role ledger tasks")
    if len(tasks) != 361:
        raise OSWorldPreservationSourceError("role ledger must contain 361 tasks")
    role_counts = {"fit": 0, "validation": 0, "capability_test": 0}
    seen: set[str] = set()
    normalized_tasks: list[dict[str, Any]] = []
    for index, value in enumerate(tasks):
        row = _object(value, f"role ledger tasks[{index}]")
        _exact(row, {"task_id", "group", "task_config_sha256", "family_id", "role"}, f"role ledger tasks[{index}]")
        task_id = _text(row["task_id"], "role ledger task ID")
        group = _text(row["group"], "role ledger group")
        role = _text(row["role"], "role ledger role")
        if task_id in seen or not task_id.startswith(f"{group}/") or role not in role_counts:
            raise OSWorldPreservationSourceError("role ledger membership is invalid")
        _sha(row["task_config_sha256"], "role ledger task config SHA")
        seen.add(task_id)
        role_counts[role] += 1
        normalized_tasks.append(dict(row))
    if role_counts != {"fit": 265, "validation": 60, "capability_test": 36}:
        raise OSWorldPreservationSourceError("role ledger counts differ")
    observed_task_hashes = _hash_json(
        {row["task_id"]: row["task_config_sha256"] for row in normalized_tasks}
    )
    if observed_task_hashes != source["task_hashes_sha256"]:
        raise OSWorldPreservationSourceError("role ledger task hash identity differs")
    claimed_ledger_id = _sha(ledger["ledger_id"], "role ledger ID")
    if _hash_json({key: value for key, value in ledger.items() if key != "ledger_id"}) != claimed_ledger_id:
        raise OSWorldPreservationSourceError("role ledger identity differs")

    optuna = dict(_load_json(optuna_path, "OSWorld Optuna manifest"))
    _exact(
        optuna,
        {"format", "ledger_id", "source_identity", "fit_tiers", "validation_task_ids", "optimizer_visibility", "manifest_id"},
        "OSWorld Optuna manifest",
    )
    if (
        optuna["format"] not in {
            "truth_editing_osworld_optuna_manifest_v1",
            "truth_editing_osworld_optuna_manifest_v2",
        }
        or optuna["ledger_id"] != claimed_ledger_id
        or optuna["optimizer_visibility"] != "fit-and-promoted-validation-only"
    ):
        raise OSWorldPreservationSourceError("OSWorld Optuna manifest identity differs")
    manifest_source = _object(optuna["source_identity"], "Optuna source identity")
    if manifest_source.get("osworld_commit") != osworld_commit or manifest_source.get(
        "task_hashes_sha256"
    ) != source["task_hashes_sha256"]:
        raise OSWorldPreservationSourceError("OSWorld role artifact source identities differ")
    tiers = _object(optuna["fit_tiers"], "Optuna fit tiers")
    if set(tiers) != set(FIT_TIER_COUNTS):
        raise OSWorldPreservationSourceError("Optuna fit tiers differ")
    by_role = {
        role: {row["task_id"] for row in normalized_tasks if row["role"] == role}
        for role in role_counts
    }
    normalized_tiers: dict[str, list[str]] = {}
    for tier, count in FIT_TIER_COUNTS.items():
        ids = [_text(value, f"Optuna {tier} task ID") for value in _array(tiers[tier], f"Optuna {tier} tasks")]
        if ids != sorted(ids) or len(ids) != count or len(set(ids)) != count or not set(ids) <= by_role["fit"]:
            raise OSWorldPreservationSourceError("Optuna fit tier membership differs")
        normalized_tiers[tier] = ids
    if not set(normalized_tiers["discovery"]) < set(normalized_tiers["promoted"]) < set(normalized_tiers["finalist"]):
        raise OSWorldPreservationSourceError("Optuna fit tiers are not nested")
    validation = [
        _text(value, "Optuna validation task ID")
        for value in _array(optuna["validation_task_ids"], "Optuna validation tasks")
    ]
    if validation != sorted(validation) or set(validation) != by_role["validation"]:
        raise OSWorldPreservationSourceError("Optuna validation membership differs")
    if set(normalized_tiers["finalist"]) != by_role["fit"]:
        raise OSWorldPreservationSourceError("Optuna finalist membership differs")
    claimed_manifest_id = _sha(optuna["manifest_id"], "Optuna manifest ID")
    if _hash_json({key: value for key, value in optuna.items() if key != "manifest_id"}) != claimed_manifest_id:
        raise OSWorldPreservationSourceError("Optuna manifest identity differs")
    return ledger, optuna


def _verify_role_build_receipt(
    path: Path,
    *,
    expected_file_sha256: str,
    ledger: Mapping[str, Any],
    optuna: Mapping[str, Any],
) -> dict[str, Any]:
    if _hash_file(path) != expected_file_sha256:
        raise OSWorldPreservationSourceError("role build receipt content hash differs")
    receipt = dict(_load_json(path, "OSWorld role build receipt"))
    _exact(
        receipt,
        {
            "format", "ledger_id", "optuna_manifest_id", "role_counts", "fit_tier_counts",
            "source_trust", "ledger_sha256", "optimizer_manifest_sha256", "receipt_id",
        },
        "OSWorld role build receipt",
    )
    if receipt["format"] != "truth_editing_osworld_role_ledger_build_receipt_v2":
        raise OSWorldPreservationSourceError("role build receipt format is unsupported")
    if (
        receipt["ledger_id"] != ledger["ledger_id"]
        or receipt["optuna_manifest_id"] != optuna["manifest_id"]
        or receipt["ledger_sha256"] != _hash_json(ledger)
        or receipt["optimizer_manifest_sha256"] != _hash_json(optuna)
    ):
        raise OSWorldPreservationSourceError("role build receipt artifact binding differs")
    claimed = _sha(receipt["receipt_id"], "role build receipt ID")
    if _hash_json({key: value for key, value in receipt.items() if key != "receipt_id"}) != claimed:
        raise OSWorldPreservationSourceError("role build receipt identity differs")
    return receipt


def _verified_reuse_index(root: Path | None) -> dict[str, dict[str, Any]]:
    if root is None:
        return {}
    open_osworld_preservation_source(root)
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    reusable: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = row["source_identity"]
        screenshot_relative = Path(row["screenshots"][0]["path"])
        screenshot_path = (root / screenshot_relative).resolve(strict=True)
        trajectory_path = screenshot_path.parent / "trajectory.jsonl"
        if (
            not trajectory_path.is_file()
            or trajectory_path.is_symlink()
            or _hash_file(trajectory_path) != identity["trajectory_sha256"]
        ):
            raise OSWorldPreservationSourceError("reusable trajectory identity differs")
        reusable[identity["task_id"]] = {
            "trajectory_key": identity["trajectory_key"],
            "trajectory_sha256": identity["trajectory_sha256"],
            "trajectory": trajectory_path.read_bytes(),
            "screenshot_key": identity["screenshot_key"],
            "screenshot_sha256": identity["screenshot_sha256"],
            "screenshot": screenshot_path.read_bytes(),
        }
    return reusable


def materialize_osworld_preservation_source(
    config_path: Path | str,
    output_dir: Path | str,
    *,
    s3_client: _S3Client | None = None,
) -> dict[str, Any]:
    """Materialize immutable fit/validation OSWorld KL inputs via exact S3 keys."""

    config_file = Path(config_path).resolve(strict=True)
    config = _load_json(config_file, "OSWorld preservation source config")
    config_format = config.get("format")
    expected_config_fields = {
            "format", "bucket", "region", "run_id", "task_map_key", "task_map_sha256",
            "role_ledger_path", "role_ledger_sha256", "optuna_manifest_path",
            "optuna_manifest_sha256", "fit_tier", "include_validation",
    }
    if config_format == OSWORLD_PRESERVATION_SOURCE_BUILD_CONFIG_V2_FORMAT:
        expected_config_fields.update(
            {"role_build_receipt_path", "role_build_receipt_sha256", "reuse_source_root"}
        )
    _exact(config, expected_config_fields, "OSWorld preservation source config")
    if config_format not in {
        OSWORLD_PRESERVATION_SOURCE_BUILD_CONFIG_FORMAT,
        OSWORLD_PRESERVATION_SOURCE_BUILD_CONFIG_V2_FORMAT,
    }:
        raise OSWorldPreservationSourceError("unsupported OSWorld preservation source config")
    bucket = _text(config["bucket"], "config.bucket")
    region = _text(config["region"], "config.region")
    run_id = _text(config["run_id"], "config.run_id")
    task_map_key = str(_safe_relative(config["task_map_key"], "config.task_map_key"))
    expected_task_map_sha = _sha(config["task_map_sha256"], "config.task_map_sha256")
    fit_tier = _text(config["fit_tier"], "config.fit_tier")
    if fit_tier not in FIT_TIER_COUNTS:
        raise OSWorldPreservationSourceError("config.fit_tier is unsupported")
    if not isinstance(config["include_validation"], bool):
        raise OSWorldPreservationSourceError("config.include_validation must be boolean")
    ledger_path = Path(_text(config["role_ledger_path"], "config.role_ledger_path"))
    optuna_path = Path(_text(config["optuna_manifest_path"], "config.optuna_manifest_path"))
    if not ledger_path.is_absolute():
        ledger_path = config_file.parent / ledger_path
    if not optuna_path.is_absolute():
        optuna_path = config_file.parent / optuna_path
    if _hash_file(ledger_path) != _sha(config["role_ledger_sha256"], "role ledger SHA"):
        raise OSWorldPreservationSourceError("role ledger content hash differs")
    if _hash_file(optuna_path) != _sha(config["optuna_manifest_sha256"], "Optuna manifest SHA"):
        raise OSWorldPreservationSourceError("Optuna manifest content hash differs")
    ledger, optuna = _open_bound_role_artifacts(ledger_path, optuna_path)
    role_receipt_id: str | None = None
    reuse_root: Path | None = None
    if config_format == OSWORLD_PRESERVATION_SOURCE_BUILD_CONFIG_V2_FORMAT:
        receipt_path = Path(
            _text(config["role_build_receipt_path"], "config.role_build_receipt_path")
        )
        if not receipt_path.is_absolute():
            receipt_path = config_file.parent / receipt_path
        verified_role_receipt = _verify_role_build_receipt(
            receipt_path,
            expected_file_sha256=_sha(
                config["role_build_receipt_sha256"], "config.role_build_receipt_sha256"
            ),
            ledger=ledger,
            optuna=optuna,
        )
        role_receipt_id = verified_role_receipt["receipt_id"]
        raw_reuse = config["reuse_source_root"]
        if raw_reuse is not None:
            reuse_root = Path(_text(raw_reuse, "config.reuse_source_root"))
            if not reuse_root.is_absolute():
                reuse_root = (config_file.parent / reuse_root).resolve(strict=True)
    source = _object(ledger["source"], "role ledger source")
    if optuna["source_identity"]["osworld_commit"] != source["osworld_commit"]:
        raise OSWorldPreservationSourceError("role artifact source commits differ")
    selected = set(optuna["fit_tiers"][fit_tier])
    if config["include_validation"]:
        selected.update(optuna["validation_task_ids"])
    roles = {row["task_id"]: row for row in ledger["tasks"]}
    if any(roles[task_id]["role"] == "capability_test" for task_id in selected):
        raise OSWorldPreservationSourceError("capability-test task crossed optimizer boundary")
    reuse_index = _verified_reuse_index(reuse_root)

    client = s3_client or _client(region)
    task_map_bytes = _read_s3(
        client, bucket, task_map_key, limit=_MAX_TASK_MAP_BYTES, name="final task map"
    )
    if _hash_bytes(task_map_bytes) != expected_task_map_sha:
        raise OSWorldPreservationSourceError("final task map content hash differs")
    try:
        task_map = _object(json.loads(task_map_bytes), "final task map")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OSWorldPreservationSourceError("final task map is not strict JSON") from error
    _exact(
        task_map,
        {"schema_version", "canonical_total", "remote_attempt_files_verified", "remote_tasks_verified", "elapsed_seconds", "verified"},
        "final task map",
    )
    if task_map["schema_version"] != 3 or task_map["canonical_total"] != 361:
        raise OSWorldPreservationSourceError("final task map catalog identity differs")
    verified = _array(task_map["verified"], "final task map.verified")
    if len(verified) != 361:
        raise OSWorldPreservationSourceError("final task map must contain 361 tasks")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(verified):
        row = _object(value, f"final task map.verified[{index}]")
        _exact(
            row,
            {"task_id", "attempt", "bundle_sha256", "remote_checkpoint_prefix", "action_count", "agent_termination", "evaluator_score", "remote_attempt_files_verified", "source"},
            f"final task map.verified[{index}]",
        )
        task_id = _text(row["task_id"], "task map task ID")
        if task_id in by_id or task_id not in roles:
            raise OSWorldPreservationSourceError("final task map task set differs from ledger")
        _sha(row["bundle_sha256"], "task bundle SHA")
        by_id[task_id] = row
    if set(by_id) != set(roles):
        raise OSWorldPreservationSourceError("final task map task set differs from ledger")

    output, staging = _output_paths(output_dir)
    rows: list[dict[str, Any]] = []
    fetched: list[dict[str, Any]] = [
        {
            "key": task_map_key,
            "sha256": expected_task_map_sha,
            "bytes": len(task_map_bytes),
            "acquisition": "s3_exact_get",
        }
    ]
    try:
        for index, task_id in enumerate(sorted(selected)):
            map_row = by_id[task_id]
            attempt = map_row["attempt"]
            if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
                raise OSWorldPreservationSourceError("task attempt must be positive")
            prefix, checkpoint_run_id = _checkpoint_key(
                map_row["remote_checkpoint_prefix"], bucket=bucket,
                task_id=task_id, attempt=attempt,
            )
            trajectory_key = f"{prefix}attempt/trajectory.jsonl"
            reusable = reuse_index.get(task_id)
            reuse_trajectory = (
                reusable is not None
                and reusable["trajectory_key"] == trajectory_key
                and reusable["trajectory_sha256"] == _hash_bytes(reusable["trajectory"])
            )
            if reuse_trajectory and reusable is not None:
                trajectory = reusable["trajectory"]
            else:
                trajectory = _read_s3(
                    client, bucket, trajectory_key, limit=_MAX_TRAJECTORY_BYTES,
                    name=f"trajectory for {task_id}",
                )
            trajectory_sha = _hash_bytes(trajectory)
            instruction, screenshot_name, screenshot_sha, response = _trajectory_rows(
                trajectory, task_id=task_id, run_id=checkpoint_run_id, attempt=attempt
            )
            screenshot_key = f"{prefix}attempt/{screenshot_name}"
            reuse_screenshot = (
                reusable is not None
                and reusable["screenshot_key"] == screenshot_key
                and reusable["screenshot_sha256"] == screenshot_sha
                and _hash_bytes(reusable["screenshot"]) == screenshot_sha
            )
            if reuse_screenshot and reusable is not None:
                screenshot = reusable["screenshot"]
            else:
                screenshot = _read_s3(
                    client, bucket, screenshot_key, limit=_MAX_SCREENSHOT_BYTES,
                    name=f"initial screenshot for {task_id}",
                )
            if _hash_bytes(screenshot) != screenshot_sha:
                raise OSWorldPreservationSourceError("initial screenshot content hash differs")
            task_dir = staging / "tasks" / f"{index:04d}"
            task_dir.mkdir(parents=True)
            trajectory_path = task_dir / "trajectory.jsonl"
            screenshot_path = task_dir / "initial.png"
            trace_path = task_dir / "trace.json"
            trajectory_path.write_bytes(trajectory)
            screenshot_path.write_bytes(screenshot)
            trace = {
                "format": "recorded_computer_use_trace_v2",
                "semantics": "observation_instruction_kl_only",
                "events": [
                    {
                        "sequence_index": 0,
                        "event_type": "observation",
                        "payload": {"screenshot_sha256": screenshot_sha},
                    }
                ],
            }
            _write_json(trace_path, trace)
            role = roles[task_id]["role"]
            row = {
                "format": OSWORLD_PRESERVATION_SOURCE_FORMAT,
                "record_id": f"osworld:{task_id}:initial-observation",
                "split": "development_preservation_recorded_computer_use",
                "preservation_category": roles[task_id]["group"],
                "prompt": instruction,
                "assistant_response": response,
                "required_action_token_id": None,
                "trace": {
                    "path": str(trace_path.relative_to(staging)),
                    "sha256": _hash_file(trace_path),
                },
                "screenshots": [
                    {
                        "path": str(screenshot_path.relative_to(staging)),
                        "sha256": screenshot_sha,
                    }
                ],
                "source_identity": {
                    "task_id": task_id,
                    "role": role,
                    "checkpoint_run_id": checkpoint_run_id,
                    "task_config_sha256": roles[task_id]["task_config_sha256"],
                    "osworld_commit": source["osworld_commit"],
                    "ledger_id": ledger["ledger_id"],
                    "optuna_manifest_id": optuna["manifest_id"],
                    "task_map_sha256": expected_task_map_sha,
                    "bundle_sha256": map_row["bundle_sha256"],
                    "trajectory_key": trajectory_key,
                    "trajectory_sha256": trajectory_sha,
                    "screenshot_key": screenshot_key,
                    "screenshot_sha256": screenshot_sha,
                },
            }
            rows.append(row)
            fetched.extend(
                [
                    {
                        "key": trajectory_key,
                        "sha256": trajectory_sha,
                        "bytes": len(trajectory),
                        "acquisition": "verified_local_reuse" if reuse_trajectory else "s3_exact_get",
                    },
                    {
                        "key": screenshot_key,
                        "sha256": screenshot_sha,
                        "bytes": len(screenshot),
                        "acquisition": "verified_local_reuse" if reuse_screenshot else "s3_exact_get",
                    },
                ]
            )
        manifest_bytes = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
        (staging / "manifest.jsonl").write_bytes(manifest_bytes)
        receipt_unsigned = {
            "format": OSWORLD_PRESERVATION_SOURCE_RECEIPT_FORMAT,
            "bucket": bucket,
            "region": region,
            "run_id": run_id,
            "ledger_id": ledger["ledger_id"],
            "optuna_manifest_id": optuna["manifest_id"],
            "role_build_receipt_id": role_receipt_id,
            "reuse_source_receipt_sha256": (
                open_osworld_preservation_source(reuse_root)["self_sha256"]
                if reuse_root is not None
                else None
            ),
            "task_map_key": task_map_key,
            "task_map_sha256": expected_task_map_sha,
            "fit_tier": fit_tier,
            "include_validation": config["include_validation"],
            "semantics": "observation_instruction_kl_only",
            "assistant_probe": _OBSERVATION_KL_ASSISTANT_PROBE,
            "record_count": len(rows),
            "checkpoint_run_ids": sorted(
                {row["source_identity"]["checkpoint_run_id"] for row in rows}
            ),
            "manifest_sha256": _hash_bytes(manifest_bytes),
            "exact_keys": fetched,
            "total_source_bytes": sum(item["bytes"] for item in fetched),
        }
        receipt = {**receipt_unsigned, "self_sha256": _hash_json(receipt_unsigned)}
        _write_json(staging / "source-receipt.json", receipt)
        _rename_no_replace(staging, output)
        return receipt
    except OSWorldPreservationSourceError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        raise OSWorldPreservationSourceError(
            f"OSWorld preservation source materialization failed: {type(error).__name__}: {error}"
        ) from error


def open_osworld_preservation_source(root: Path | str) -> dict[str, Any]:
    """Strict-open a materialized source and reverify every local artifact."""

    source_root = Path(root).resolve(strict=True)
    if Path(root).is_symlink() or not source_root.is_dir():
        raise OSWorldPreservationSourceError("source root must be a regular directory")
    receipt = dict(_load_json(source_root / "source-receipt.json", "source receipt"))
    common_receipt_fields = {
            "format", "bucket", "region", "run_id", "ledger_id", "optuna_manifest_id",
            "task_map_key", "task_map_sha256", "fit_tier", "include_validation",
            "semantics", "assistant_probe", "checkpoint_run_ids", "record_count",
            "manifest_sha256", "exact_keys", "total_source_bytes", "self_sha256",
    }
    if receipt.get("format") == OSWORLD_PRESERVATION_SOURCE_RECEIPT_FORMAT:
        common_receipt_fields.update(
            {"role_build_receipt_id", "reuse_source_receipt_sha256"}
        )
    _exact(receipt, common_receipt_fields, "source receipt")
    if (
        receipt["format"] not in {
            OSWORLD_PRESERVATION_SOURCE_RECEIPT_FORMAT,
            _SOURCE_RECEIPT_V2_FORMAT,
        }
        or receipt["semantics"] != "observation_instruction_kl_only"
        or receipt["assistant_probe"] != _OBSERVATION_KL_ASSISTANT_PROBE
    ):
        raise OSWorldPreservationSourceError("source receipt semantics are unsupported")
    claimed = _sha(receipt["self_sha256"], "source receipt self SHA")
    if _hash_json({key: value for key, value in receipt.items() if key != "self_sha256"}) != claimed:
        raise OSWorldPreservationSourceError("source receipt identity differs")
    manifest_path = source_root / "manifest.jsonl"
    manifest_bytes = manifest_path.read_bytes()
    if _hash_bytes(manifest_bytes) != _sha(receipt["manifest_sha256"], "manifest SHA"):
        raise OSWorldPreservationSourceError("source manifest content hash differs")
    try:
        rows = [
            _object(json.loads(line), f"source manifest line {index + 1}")
            for index, line in enumerate(manifest_bytes.decode("utf-8").splitlines())
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OSWorldPreservationSourceError("source manifest is not strict JSONL") from error
    if receipt["record_count"] != len(rows) or not rows:
        raise OSWorldPreservationSourceError("source receipt record count differs")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        _exact(
            row,
            {
                "format", "record_id", "split", "preservation_category", "prompt",
                "assistant_response", "required_action_token_id", "trace", "screenshots",
                "source_identity",
            },
            f"source manifest line {index + 1}",
        )
        if (
            row["format"] != OSWORLD_PRESERVATION_SOURCE_FORMAT
            or row["split"] != "development_preservation_recorded_computer_use"
            or row["required_action_token_id"] is not None
            or row["assistant_response"] != _OBSERVATION_KL_ASSISTANT_PROBE
        ):
            raise OSWorldPreservationSourceError("source row KL semantics differ")
        identity = _object(row["source_identity"], "source row identity")
        _exact(
            identity,
            {
                "task_id", "role", "checkpoint_run_id", "task_config_sha256",
                "osworld_commit", "ledger_id", "optuna_manifest_id", "task_map_sha256",
                "bundle_sha256", "trajectory_key", "trajectory_sha256", "screenshot_key",
                "screenshot_sha256",
            },
            "source row identity",
        )
        role = identity.get("role")
        if role not in {"fit", "validation"}:
            raise OSWorldPreservationSourceError("source row crosses capability-test boundary")
        task_id = _text(identity.get("task_id"), "source row task ID")
        if task_id in seen:
            raise OSWorldPreservationSourceError("source row task IDs are duplicated")
        seen.add(task_id)
        if (
            identity["ledger_id"] != receipt["ledger_id"]
            or identity["optuna_manifest_id"] != receipt["optuna_manifest_id"]
            or identity["task_map_sha256"] != receipt["task_map_sha256"]
            or identity["checkpoint_run_id"] not in receipt["checkpoint_run_ids"]
        ):
            raise OSWorldPreservationSourceError("source row receipt binding differs")
        for field in (
            "task_config_sha256", "bundle_sha256", "trajectory_sha256", "screenshot_sha256"
        ):
            _sha(identity[field], f"source row identity.{field}")
        trace = _object(row["trace"], "source row trace")
        screenshots = _array(row["screenshots"], "source row screenshots")
        if len(screenshots) != 1:
            raise OSWorldPreservationSourceError("source row requires one screenshot")
        for item, name in ((trace, "trace"), (_object(screenshots[0], "screenshot"), "screenshot")):
            _exact(item, {"path", "sha256"}, f"source row {name}")
            relative = Path(str(_safe_relative(item["path"], f"source row {name} path")))
            path = source_root / relative
            resolved = path.resolve(strict=True)
            if path.is_symlink() or not resolved.is_file() or not resolved.is_relative_to(source_root):
                raise OSWorldPreservationSourceError(f"source row {name} path is invalid")
            if _hash_file(resolved) != _sha(item["sha256"], f"source row {name} SHA"):
                raise OSWorldPreservationSourceError(f"source row {name} content hash differs")
        if row["screenshots"][0]["sha256"] != identity["screenshot_sha256"]:
            raise OSWorldPreservationSourceError("source row screenshot identity differs")
    keys = _array(receipt["exact_keys"], "source receipt exact keys")
    if len(keys) != 1 + 2 * len(rows):
        raise OSWorldPreservationSourceError("source receipt exact-key count differs")
    total = 0
    for index, value in enumerate(keys):
        item = _object(value, f"source receipt exact_keys[{index}]")
        expected_key_fields = {"key", "sha256", "bytes"}
        if receipt["format"] == OSWORLD_PRESERVATION_SOURCE_RECEIPT_FORMAT:
            expected_key_fields.add("acquisition")
        _exact(item, expected_key_fields, f"source receipt exact_keys[{index}]")
        if "acquisition" in item and item["acquisition"] not in {
            "s3_exact_get", "verified_local_reuse"
        }:
            raise OSWorldPreservationSourceError("source receipt acquisition is invalid")
        _safe_relative(item["key"], "source receipt exact key")
        _sha(item["sha256"], "source receipt exact-key SHA")
        size = item["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise OSWorldPreservationSourceError("source receipt exact-key bytes are invalid")
        total += size
    if receipt["total_source_bytes"] != total:
        raise OSWorldPreservationSourceError("source receipt byte total differs")
    return receipt


__all__ = [
    "OSWORLD_PRESERVATION_SOURCE_BUILD_CONFIG_FORMAT",
    "OSWORLD_PRESERVATION_SOURCE_FORMAT",
    "OSWORLD_PRESERVATION_SOURCE_RECEIPT_FORMAT",
    "OSWorldPreservationSourceError",
    "materialize_osworld_preservation_source",
    "open_osworld_preservation_source",
]
