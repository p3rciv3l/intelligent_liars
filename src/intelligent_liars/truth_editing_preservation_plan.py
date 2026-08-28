"""Deterministic, offline construction of capability-preservation plans.

The builder admits only development preservation rows from hash-bound text and
vision corpora plus an explicit recorded-computer-use source root.  It freezes
teacher-forced inputs, creates nested category-stratified tiers, emits the
baseline capture plan, and records a strict bridge that can later combine a
verified capture bundle with those inputs into a materialization plan.
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
from pathlib import Path
from typing import Any


PRESERVATION_PLAN_BUILD_CONFIG_FORMAT = (
    "truth_editing_preservation_plan_build_config_v1"
)
PRESERVATION_POST_CAPTURE_BRIDGE_FORMAT = (
    "truth_editing_preservation_post_capture_bridge_v1"
)
PRESERVATION_PLAN_BUILD_RECEIPT_FORMAT = (
    "truth_editing_preservation_plan_build_receipt_v1"
)
PRESERVATION_CAPTURE_PLAN_FORMAT = (
    "truth_editing_preservation_baseline_capture_plan_v2"
)
PRESERVATION_CAPTURE_RUN_FORMAT = (
    "truth_editing_preservation_baseline_capture_run_v2"
)
PRESERVATION_CAPTURE_RECEIPT_FORMAT = (
    "truth_editing_preservation_base_logits_capture_receipt_v2"
)
PRESERVATION_COMPACT_CAPTURE_REPRESENTATION = (
    "assistant_top64_plus_other_token_id_tiebreak_v2"
)
PRESERVATION_MATERIALIZATION_PLAN_FORMAT = (
    "truth_editing_preservation_materialization_plan_v1"
)

_STRATA = ("text", "vision", "recorded_computer_use")
_TIERS = ("trial", "promoted", "finalist")
_IDENTITY_FIELDS = (
    "base_model_sha256",
    "tokenizer_sha256",
    "processor_sha256",
    "chat_template_sha256",
    "inference_runtime_sha256",
)
_DEVELOPMENT_SPLITS = {
    "text": "development_preservation_text",
    "vision": "development_preservation_vision",
    "recorded_computer_use": "development_preservation_recorded_computer_use",
}
_HEX = frozenset("0123456789abcdef")


class PreservationPlanError(RuntimeError):
    """A preservation source or plan bridge failed strict validation."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PreservationPlanError("value is not canonical JSON") from error


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PreservationPlanError(f"source file is unreadable: {path}") from error
    return digest.hexdigest()


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreservationPlanError(f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PreservationPlanError(f"{name} must be an array")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise PreservationPlanError(
            f"{name} fields differ; missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PreservationPlanError(f"{name} must be nonempty trimmed text")
    return value


def _message_text(value: Any, name: str) -> str:
    """Validate substantive message text without rewriting source formatting."""

    if not isinstance(value, str) or not value.strip():
        raise PreservationPlanError(f"{name} must contain non-whitespace text")
    return value


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise PreservationPlanError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PreservationPlanError(f"{name} must be a positive integer")
    return value


def _load_json(path: Path, name: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreservationPlanError(f"{name} is not strict JSON") from error


def _load_jsonl(path: Path, name: str) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise PreservationPlanError(f"{name} is unreadable") from error
    if not lines:
        raise PreservationPlanError(f"{name} must not be empty")
    rows: list[Mapping[str, Any]] = []
    for index, line in enumerate(lines):
        if not line or line.strip() != line:
            raise PreservationPlanError(f"{name} line {index + 1} is not canonical JSONL")
        try:
            rows.append(_object(json.loads(line), f"{name} line {index + 1}"))
        except json.JSONDecodeError as error:
            raise PreservationPlanError(
                f"{name} line {index + 1} is not strict JSON"
            ) from error
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _source_path(root: Path, value: Any, name: str, *, directory: bool = False) -> Path:
    relative = Path(_text(value, name))
    if relative.is_absolute() or ".." in relative.parts:
        raise PreservationPlanError(f"{name} must stay below its declared source root")
    try:
        resolved_root = root.resolve(strict=True)
        candidate = root / relative
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise PreservationPlanError(f"{name} is missing or unreadable") from error
    expected_kind = resolved.is_dir() if directory else resolved.is_file()
    if candidate.is_symlink() or not expected_kind or not resolved.is_relative_to(resolved_root):
        kind = "directory" if directory else "file"
        raise PreservationPlanError(f"{name} must be a regular non-symlink {kind}")
    return resolved


def _media_source_path(
    config_root: Path, media_root: Path, value: Any, name: str
) -> Path:
    """Resolve repo-relative media paths while retaining legacy root-relative input."""
    relative = Path(_text(value, name))
    if relative.is_absolute() or ".." in relative.parts:
        raise PreservationPlanError(f"{name} must stay below its declared source root")
    try:
        resolved_media_root = media_root.resolve(strict=True)
    except OSError as error:
        raise PreservationPlanError(f"{name} is missing or unreadable") from error

    repo_candidate = config_root / relative
    try:
        resolved = repo_candidate.resolve(strict=True)
    except OSError:
        return _source_path(media_root, str(relative), name)
    if not resolved.is_relative_to(resolved_media_root):
        raise PreservationPlanError(f"{name} must stay below its declared source root")
    if repo_candidate.is_symlink() or not resolved.is_file():
        raise PreservationPlanError(f"{name} must be a regular non-symlink file")
    return resolved


def _rename_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    system = platform.system()
    if system == "Darwin" and hasattr(libc, "renamex_np"):
        result = libc.renamex_np(
            os.fsencode(source), os.fsencode(destination), ctypes.c_uint(0x00000004)
        )
    elif system == "Linux" and hasattr(libc, "renameat2"):
        result = libc.renameat2(
            ctypes.c_int(-100),
            os.fsencode(source),
            ctypes.c_int(-100),
            os.fsencode(destination),
            ctypes.c_uint(1),
        )
    else:  # pragma: no cover - production platforms are macOS and Linux
        raise PreservationPlanError("atomic no-replace publication is unsupported")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise PreservationPlanError(f"output already exists: {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _new_output(output_dir: Path | str) -> tuple[Path, Path]:
    requested = Path(output_dir).expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    output = requested.parent.resolve() / requested.name
    if os.path.lexists(output):
        raise PreservationPlanError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    return output, staging


def _validate_messages(value: Any, name: str) -> list[dict[str, Any]]:
    rows = _array(value, name)
    if len(rows) < 2:
        raise PreservationPlanError(f"{name} must contain a prompt and response")
    normalized: list[dict[str, Any]] = []
    for index, raw_value in enumerate(rows):
        raw = _object(raw_value, f"{name}[{index}]")
        _exact(raw, {"role", "content"}, f"{name}[{index}]")
        role = _text(raw["role"], f"{name}[{index}].role")
        if role not in {"system", "user", "assistant"}:
            raise PreservationPlanError(f"{name}[{index}] has an unsupported role")
        content = raw["content"]
        if isinstance(content, str):
            normalized_content: Any = _message_text(
                content, f"{name}[{index}].content"
            )
        else:
            blocks = _array(content, f"{name}[{index}].content")
            if not blocks:
                raise PreservationPlanError(f"{name}[{index}].content must not be empty")
            normalized_content = [dict(_object(block, "message content block")) for block in blocks]
        normalized.append({"role": role, "content": normalized_content})
    if normalized[-1]["role"] != "assistant" or not isinstance(
        normalized[-1]["content"], str
    ):
        raise PreservationPlanError(f"{name} must end with a textual assistant response")
    return normalized


def _admit_common(row: Mapping[str, Any], stratum: str, name: str) -> tuple[str, str]:
    if row.get("format") != "tinylora_step5_example_v1":
        raise PreservationPlanError(f"{name} has an unsupported source format")
    if row.get("split") != _DEVELOPMENT_SPLITS[stratum]:
        raise PreservationPlanError(f"{name} belongs to a sealed or non-development split")
    if row.get("kind") != "preservation" or row.get("objective") != "preservation_kl":
        raise PreservationPlanError(f"{name} is not a preservation-only record")
    return (
        _text(row.get("record_id"), f"{name}.record_id"),
        _text(row.get("preservation_category"), f"{name}.preservation_category"),
    )


def _copy_media(
    source: Path, destination: Path, expected_sha256: str, *, name: str
) -> dict[str, str]:
    if _hash_file(source) != expected_sha256:
        raise PreservationPlanError(f"{name} content hash differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _hash_file(destination) != expected_sha256:
        raise PreservationPlanError(f"{name} copied content hash differs")
    return {"path": str(destination), "sha256": expected_sha256}


def _validate_recorded_trace(
    path: Path, name: str, *, observation_only: bool = False
) -> None:
    raw = _object(_load_json(path, f"{name} trace"), f"{name} trace")
    expected_fields = {"format", "events", "semantics"} if observation_only else {"format", "events"}
    _exact(raw, expected_fields, f"{name} trace")
    if observation_only:
        if (
            raw["format"] != "recorded_computer_use_trace_v2"
            or raw["semantics"] != "observation_instruction_kl_only"
        ):
            raise PreservationPlanError("unsupported observation-only computer-use trace")
    elif raw["format"] != "recorded_computer_use_trace_v1":
        raise PreservationPlanError("unsupported recorded computer-use trace format")
    events = _array(raw["events"], f"{name} trace events")
    if not events:
        raise PreservationPlanError("recorded computer-use trace events must be nonempty")
    action_types = {"click", "keypress", "type_text", "scroll", "navigate"}
    allowed_types = action_types | {"observation", "wait"}
    observed_action = False
    for index, value in enumerate(events):
        event = _object(value, f"{name} trace event {index}")
        _exact(
            event,
            {"sequence_index", "event_type", "payload"},
            f"{name} trace event {index}",
        )
        if event["sequence_index"] != index:
            raise PreservationPlanError(
                "recorded computer-use trace sequence indices must be contiguous"
            )
        event_type = _text(event["event_type"], f"{name} trace event_type")
        if event_type not in allowed_types:
            raise PreservationPlanError("recorded computer-use trace event type is unsupported")
        _object(event["payload"], f"{name} trace event payload")
        observed_action = observed_action or event_type in action_types
    if not observed_action and not observation_only:
        raise PreservationPlanError("recorded computer-use trace requires an action event")


def _osworld_source_identity(value: Any, name: str) -> dict[str, str]:
    raw = _object(value, name)
    _exact(
        raw,
        {
            "task_id", "role", "checkpoint_run_id", "task_config_sha256", "osworld_commit", "ledger_id",
            "optuna_manifest_id", "task_map_sha256", "bundle_sha256", "trajectory_key",
            "trajectory_sha256", "screenshot_key", "screenshot_sha256",
        },
        name,
    )
    role = _text(raw["role"], f"{name}.role")
    if role not in {"fit", "validation"}:
        raise PreservationPlanError(f"{name}.role crosses the capability-test boundary")
    commit = _text(raw["osworld_commit"], f"{name}.osworld_commit")
    if len(commit) != 40 or any(character not in _HEX for character in commit):
        raise PreservationPlanError(f"{name}.osworld_commit must be a lowercase git SHA-1")
    normalized = {
        "task_id": _text(raw["task_id"], f"{name}.task_id"),
        "role": role,
        "checkpoint_run_id": _text(raw["checkpoint_run_id"], f"{name}.checkpoint_run_id"),
        "osworld_commit": commit,
        "trajectory_key": _text(raw["trajectory_key"], f"{name}.trajectory_key"),
        "screenshot_key": _text(raw["screenshot_key"], f"{name}.screenshot_key"),
    }
    for field in (
        "task_config_sha256", "ledger_id", "optuna_manifest_id", "task_map_sha256",
        "bundle_sha256", "trajectory_sha256", "screenshot_sha256",
    ):
        normalized[field] = _sha(raw[field], f"{name}.{field}")
    return normalized


def _parse_candidates(
    config_root: Path, sources: Mapping[str, Any], staging: Path
) -> dict[str, list[dict[str, Any]]]:
    _exact(sources, set(_STRATA), "config.sources")
    candidates: dict[str, list[dict[str, Any]]] = {stratum: [] for stratum in _STRATA}

    for stratum in ("text", "vision"):
        source_raw = _object(sources[stratum], f"sources.{stratum}")
        expected_fields = {"path", "sha256"}
        if stratum == "vision":
            expected_fields.add("media_root")
        _exact(source_raw, expected_fields, f"sources.{stratum}")
        source_path = _source_path(config_root, source_raw["path"], f"sources.{stratum}.path")
        expected_source_sha = _sha(source_raw["sha256"], f"sources.{stratum}.sha256")
        if _hash_file(source_path) != expected_source_sha:
            raise PreservationPlanError(f"{stratum} source content hash differs")
        media_root = None
        if stratum == "vision":
            media_root = _source_path(
                config_root,
                source_raw["media_root"],
                "sources.vision.media_root",
                directory=True,
            )
        for row_index, row in enumerate(_load_jsonl(source_path, f"{stratum} source")):
            name = f"{stratum} source row {row_index + 1}"
            record_id, category = _admit_common(row, stratum, name)
            messages = _validate_messages(row.get("messages"), f"{name}.messages")
            media: list[dict[str, str]] = []
            if stratum == "text":
                if any(not isinstance(message["content"], str) for message in messages):
                    raise PreservationPlanError("text preservation rows cannot bind media")
                normalized_messages = messages
            else:
                assert media_root is not None
                image_sha = _sha(row.get("image_sha256"), f"{name}.image_sha256")
                image_blocks: list[tuple[int, int, Mapping[str, Any]]] = []
                for message_index, message in enumerate(messages[:-1]):
                    if isinstance(message["content"], list):
                        for block_index, block_value in enumerate(message["content"]):
                            block = _object(block_value, "vision content block")
                            if block.get("type") == "image":
                                image_blocks.append((message_index, block_index, block))
                if len(image_blocks) != 1:
                    raise PreservationPlanError("vision preservation rows require exactly one image")
                message_index, block_index, image_block = image_blocks[0]
                _exact(image_block, {"type", "image"}, "vision image block")
                image_source = _media_source_path(
                    config_root, media_root, image_block["image"], "vision image path"
                )
                if _hash_file(image_source) != image_sha:
                    raise PreservationPlanError("vision image content hash differs")
                media_id = "image-0"
                suffix = image_source.suffix.lower() or ".bin"
                relative_media = Path("media") / "pending" / f"{media_id}{suffix}"
                media.append(
                    {
                        "media_id": media_id,
                        "media_type": "image",
                        "source_path": str(image_source),
                        "sha256": image_sha,
                        "relative_template": str(relative_media),
                    }
                )
                normalized_messages = json.loads(json.dumps(messages))
                normalized_messages[message_index]["content"][block_index] = {
                    "type": "image",
                    "media_id": media_id,
                }
            candidates[stratum].append(
                {
                    "record_id": record_id,
                    "category": category,
                    "stratum": stratum,
                    "required_action_token_id": None,
                    "messages": normalized_messages,
                    "media": media,
                }
            )

    computer_raw = _object(sources["recorded_computer_use"], "sources.recorded_computer_use")
    _exact(
        computer_raw,
        {"source_root", "manifest_path", "manifest_sha256"},
        "sources.recorded_computer_use",
    )
    computer_root = _source_path(
        config_root,
        computer_raw["source_root"],
        "sources.recorded_computer_use.source_root",
        directory=True,
    )
    manifest_path = _source_path(
        computer_root,
        computer_raw["manifest_path"],
        "sources.recorded_computer_use.manifest_path",
    )
    manifest_sha = _sha(
        computer_raw["manifest_sha256"],
        "sources.recorded_computer_use.manifest_sha256",
    )
    if _hash_file(manifest_path) != manifest_sha:
        raise PreservationPlanError("recorded computer-use manifest content hash differs")
    for row_index, row in enumerate(_load_jsonl(manifest_path, "recorded computer-use manifest")):
        name = f"recorded computer-use row {row_index + 1}"
        source_format = row.get("format")
        expected_fields = {
            "format", "record_id", "split", "preservation_category", "prompt",
            "assistant_response", "required_action_token_id", "trace", "screenshots",
        }
        if source_format == "truth_editing_recorded_computer_use_source_v2":
            expected_fields.add("source_identity")
        _exact(row, expected_fields, name)
        if source_format not in {
            "truth_editing_recorded_computer_use_source_v1",
            "truth_editing_recorded_computer_use_source_v2",
        }:
            raise PreservationPlanError(f"{name} has an unsupported source format")
        if row["split"] != _DEVELOPMENT_SPLITS["recorded_computer_use"]:
            raise PreservationPlanError(f"{name} belongs to a sealed or non-development split")
        record_id = _text(row["record_id"], f"{name}.record_id")
        category = _text(row["preservation_category"], f"{name}.preservation_category")
        if source_format == "truth_editing_recorded_computer_use_source_v2":
            if row["required_action_token_id"] is not None:
                raise PreservationPlanError(
                    f"{name}.required_action_token_id must be null for observation-only KL"
                )
            action = None
            source_identity: dict[str, str] | None = _osworld_source_identity(
                row["source_identity"], f"{name}.source_identity"
            )
        else:
            action = _positive_int(
                row["required_action_token_id"], f"{name}.required_action_token_id"
            )
            source_identity = None
        computer_media: list[dict[str, str]] = []
        content: list[dict[str, str]] = []
        screenshots_raw = _array(row["screenshots"], f"{name}.screenshots")
        if not screenshots_raw:
            raise PreservationPlanError(f"{name} requires at least one screenshot")
        for screenshot_index, value in enumerate(screenshots_raw):
            screenshot = _object(value, f"{name}.screenshots[{screenshot_index}]")
            _exact(screenshot, {"path", "sha256"}, f"{name}.screenshots[{screenshot_index}]")
            source = _source_path(computer_root, screenshot["path"], "computer screenshot path")
            digest = _sha(screenshot["sha256"], "computer screenshot SHA")
            if _hash_file(source) != digest:
                raise PreservationPlanError("recorded computer-use screenshot content hash differs")
            media_id = f"screenshot-{screenshot_index}"
            computer_media.append(
                {
                    "media_id": media_id,
                    "media_type": "image",
                    "source_path": str(source),
                    "sha256": digest,
                    "relative_template": str(Path("media") / "pending" / f"{media_id}{source.suffix.lower()}"),
                }
            )
            content.append({"type": "image", "media_id": media_id})
        if (
            source_identity is not None
            and source_identity["screenshot_sha256"] != screenshots_raw[0]["sha256"]
        ):
            raise PreservationPlanError("OSWorld source screenshot identity differs")
        trace = _object(row["trace"], f"{name}.trace")
        _exact(trace, {"path", "sha256"}, f"{name}.trace")
        trace_source = _source_path(computer_root, trace["path"], "computer trace path")
        trace_sha = _sha(trace["sha256"], "computer trace SHA")
        if _hash_file(trace_source) != trace_sha:
            raise PreservationPlanError("recorded computer-use trace content hash differs")
        _validate_recorded_trace(
            trace_source,
            name,
            observation_only=source_format == "truth_editing_recorded_computer_use_source_v2",
        )
        computer_media.append(
            {
                "media_id": "trace-0",
                "media_type": "recorded_computer_use_trace",
                "source_path": str(trace_source),
                "sha256": trace_sha,
                "relative_template": str(Path("media") / "pending" / "trace-0.json"),
            }
        )
        content.extend(
            [
                {"type": "recorded_computer_use_trace", "media_id": "trace-0"},
                {"type": "text", "text": _text(row["prompt"], f"{name}.prompt")},
            ]
        )
        candidates["recorded_computer_use"].append(
            {
                "record_id": record_id,
                "category": category,
                "stratum": "recorded_computer_use",
                "required_action_token_id": action,
                "source_identity": source_identity,
                "messages": [
                    {"role": "user", "content": content},
                    {
                        "role": "assistant",
                        "content": _text(row["assistant_response"], f"{name}.assistant_response"),
                    },
                ],
                "media": computer_media,
            }
        )
    return candidates


def _stratified_order(rows: Sequence[dict[str, Any]], seed: str, stratum: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["category"], []).append(row)
    for category, group in groups.items():
        group.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}\0{stratum}\0{category}\0{row['record_id']}".encode()
            ).hexdigest()
        )
    ordered: list[dict[str, Any]] = []
    categories = sorted(groups)
    offset = 0
    while True:
        appended = False
        for category in categories:
            if offset < len(groups[category]):
                ordered.append(groups[category][offset])
                appended = True
        if not appended:
            return ordered
        offset += 1


def _materialize_inputs(
    selected: Sequence[dict[str, Any]], staging: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        media_records: list[dict[str, str]] = []
        for media in row["media"]:
            source = Path(media["source_path"])
            suffix = source.suffix.lower() or ".bin"
            relative = Path("inputs") / "media" / f"{index:04d}" / f"{media['media_id']}{suffix}"
            expected = _sha(media["sha256"], "media SHA")
            _copy_media(source, staging / relative, expected, name=f"media for {row['record_id']}")
            media_records.append(
                {
                    "media_id": media["media_id"],
                    "media_type": media["media_type"],
                    "path": str(Path("media") / f"{index:04d}" / f"{media['media_id']}{suffix}"),
                    "sha256": expected,
                }
            )
        input_payload = {"messages": row["messages"], "media": media_records}
        input_path = Path("inputs") / f"{index:04d}.json"
        _write_json(staging / input_path, input_payload)
        records.append(
            {
                "record_id": row["record_id"],
                "category": row["category"],
                "stratum": row["stratum"],
                "required_action_token_id": row["required_action_token_id"],
                "source_identity": row.get("source_identity"),
                "input_path": str(input_path),
                "input_sha256": _hash_file(staging / input_path),
            }
        )
    return records


def _build(config_path: Path, staging: Path) -> dict[str, Any]:
    config = _object(_load_json(config_path, "preservation plan build config"), "config")
    _exact(
        config,
        {
            "format",
            "spec_id",
            "selection_seed",
            *_IDENTITY_FIELDS,
            "vision_tower_sha256",
            "batch_size",
            "top_k",
            "temperature",
            "sources",
            "tier_counts_per_stratum",
        },
        "preservation plan build config",
    )
    if config["format"] != PRESERVATION_PLAN_BUILD_CONFIG_FORMAT:
        raise PreservationPlanError("unsupported preservation plan build config")
    if config["top_k"] != 64:
        raise PreservationPlanError("preservation top_k must be exactly 64")
    temperature = config["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature <= 0:
        raise PreservationPlanError("temperature must be positive")
    identity = {field: _sha(config[field], f"config.{field}") for field in _IDENTITY_FIELDS}
    vision_tower_sha = _sha(config["vision_tower_sha256"], "config.vision_tower_sha256")
    counts_raw = _object(config["tier_counts_per_stratum"], "tier_counts_per_stratum")
    _exact(counts_raw, set(_TIERS), "tier_counts_per_stratum")
    counts = {tier: _positive_int(counts_raw[tier], f"tier count {tier}") for tier in _TIERS}
    if not counts["trial"] < counts["promoted"] < counts["finalist"]:
        raise PreservationPlanError("tier counts must be strictly nested")
    seed = _text(config["selection_seed"], "config.selection_seed")
    candidates = _parse_candidates(
        config_path.parent, _object(config["sources"], "config.sources"), staging
    )
    all_ids = [row["record_id"] for rows in candidates.values() for row in rows]
    if len(set(all_ids)) != len(all_ids):
        raise PreservationPlanError("preservation source record IDs must be globally unique")
    selected: list[dict[str, Any]] = []
    tiers: dict[str, list[str]] = {tier: [] for tier in _TIERS}
    for stratum in _STRATA:
        ordered = _stratified_order(candidates[stratum], seed, stratum)
        if len(ordered) < counts["finalist"]:
            raise PreservationPlanError(f"{stratum} has insufficient admitted records")
        chosen = ordered[: counts["finalist"]]
        selected.extend(chosen)
        for tier in _TIERS:
            tiers[tier].extend(row["record_id"] for row in chosen[: counts[tier]])
    records = _materialize_inputs(selected, staging)
    capture_plan = {
        "format": PRESERVATION_CAPTURE_PLAN_FORMAT,
        **identity,
        "batch_size": _positive_int(config["batch_size"], "config.batch_size"),
        "top_k": 64,
        "temperature": float(temperature),
        "records": [
            {
                "record_id": record["record_id"],
                "input_path": record["input_path"],
                "input_sha256": record["input_sha256"],
                "required_action_token_id": record["required_action_token_id"],
            }
            for record in records
        ],
    }
    capture_sha = _hash_json(capture_plan)
    _write_json(staging / "capture-plan.json", capture_plan)
    bridge_unsigned = {
        "format": PRESERVATION_POST_CAPTURE_BRIDGE_FORMAT,
        "spec_id": _text(config["spec_id"], "config.spec_id"),
        **identity,
        "vision_tower_sha256": vision_tower_sha,
        "top_k": 64,
        "temperature": float(temperature),
        "capture_plan_path": "capture-plan.json",
        "capture_plan_sha256": capture_sha,
        "records": records,
        "tiers": tiers,
    }
    bridge = {**bridge_unsigned, "self_sha256": _hash_json(bridge_unsigned)}
    _write_json(staging / "post-capture-materialization-bridge.json", bridge)
    artifact_hashes = {
        str(path.relative_to(staging)): _hash_file(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file()
    }
    receipt_unsigned = {
        "format": PRESERVATION_PLAN_BUILD_RECEIPT_FORMAT,
        "config_sha256": _hash_json(config),
        "capture_plan_sha256": capture_sha,
        "bridge_sha256": bridge["self_sha256"],
        "record_count": len(records),
        "tier_counts": {tier: len(ids) for tier, ids in tiers.items()},
        "artifact_sha256": artifact_hashes,
    }
    receipt = {**receipt_unsigned, "self_sha256": _hash_json(receipt_unsigned)}
    _write_json(staging / "plan-build-receipt.json", receipt)
    return receipt


def build_preservation_capture_plan(
    config_path: Path | str, output_dir: Path | str
) -> dict[str, Any]:
    """Build an immutable capture-plan bundle from admitted local sources."""

    try:
        config = Path(config_path).resolve(strict=True)
    except OSError as error:
        raise PreservationPlanError("preservation plan build config is missing") from error
    output, staging = _new_output(output_dir)
    try:
        receipt = _build(config, staging)
        _rename_no_replace(staging, output)
        return receipt
    except PreservationPlanError:
        shutil.rmtree(staging)
        raise
    except Exception as error:
        shutil.rmtree(staging)
        raise PreservationPlanError(
            f"preservation plan build failed: {type(error).__name__}: {error}"
        ) from error


def _validate_bridge(path: Path) -> dict[str, Any]:
    raw = _object(_load_json(path, "post-capture materialization bridge"), "bridge")
    expected = {
        "format",
        "spec_id",
        *_IDENTITY_FIELDS,
        "vision_tower_sha256",
        "top_k",
        "temperature",
        "capture_plan_path",
        "capture_plan_sha256",
        "records",
        "tiers",
        "self_sha256",
    }
    _exact(raw, expected, "post-capture materialization bridge")
    if raw["format"] != PRESERVATION_POST_CAPTURE_BRIDGE_FORMAT:
        raise PreservationPlanError("unsupported post-capture materialization bridge")
    claimed = _sha(raw["self_sha256"], "bridge.self_sha256")
    unsigned = dict(raw)
    del unsigned["self_sha256"]
    if _hash_json(unsigned) != claimed:
        raise PreservationPlanError("post-capture bridge hash mismatch")
    capture_plan_path = _source_path(path.parent, raw["capture_plan_path"], "bridge capture plan")
    capture_plan = _object(_load_json(capture_plan_path, "capture plan"), "capture plan")
    if _hash_json(capture_plan) != _sha(raw["capture_plan_sha256"], "bridge.capture_plan_sha256"):
        raise PreservationPlanError("capture plan identity differs from bridge")
    return dict(raw)


def _validate_capture_run(
    capture_root: Path, bridge: Mapping[str, Any]
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    run = _object(_load_json(capture_root / "capture-run-receipt.json", "capture run receipt"), "capture run receipt")
    _exact(
        run,
        {
            "format",
            "plan_sha256",
            "record_count",
            "backend_identity",
            "records",
            "artifact_sha256",
            "self_sha256",
        },
        "capture run receipt",
    )
    if run["format"] != PRESERVATION_CAPTURE_RUN_FORMAT:
        raise PreservationPlanError("unsupported capture run receipt")
    claimed = _sha(run["self_sha256"], "capture run receipt.self_sha256")
    unsigned = dict(run)
    del unsigned["self_sha256"]
    if _hash_json(unsigned) != claimed:
        raise PreservationPlanError("capture run receipt hash mismatch")
    if run["plan_sha256"] != bridge["capture_plan_sha256"]:
        raise PreservationPlanError("capture run plan identity differs from bridge")
    identity = _object(run["backend_identity"], "capture backend identity")
    _exact(identity, set(_IDENTITY_FIELDS), "capture backend identity")
    if any(identity[field] != bridge[field] for field in _IDENTITY_FIELDS):
        raise PreservationPlanError("capture backend identity differs from bridge")
    records = list(_array(run["records"], "capture run records"))
    bridge_records = list(_array(bridge["records"], "bridge records"))
    if run["record_count"] != len(records) or len(records) != len(bridge_records):
        raise PreservationPlanError("capture run record count differs from bridge")
    artifacts = _object(run["artifact_sha256"], "capture run artifacts")
    for raw_path, digest_value in artifacts.items():
        artifact = _source_path(capture_root, raw_path, "capture artifact")
        digest = _sha(digest_value, "capture artifact SHA")
        if _hash_file(artifact) != digest:
            raise PreservationPlanError("capture artifact content hash differs")
    normalized_records: list[Mapping[str, Any]] = []
    for index, (record_value, bridge_value) in enumerate(zip(records, bridge_records, strict=True)):
        record = _object(record_value, f"capture run records[{index}]")
        _exact(
            record,
            {
                "record_id",
                "input_sha256",
                "base_logits_path",
                "base_logits_sha256",
                "base_logits_capture_receipt_path",
                "base_logits_capture_receipt_sha256",
            },
            f"capture run records[{index}]",
        )
        bridge_record = _object(bridge_value, f"bridge records[{index}]")
        if record["record_id"] != bridge_record["record_id"] or record["input_sha256"] != bridge_record["input_sha256"]:
            raise PreservationPlanError("capture record identity or order differs from bridge")
        for path_field, sha_field in (
            ("base_logits_path", "base_logits_sha256"),
            ("base_logits_capture_receipt_path", "base_logits_capture_receipt_sha256"),
        ):
            source = _source_path(capture_root, record[path_field], f"capture record {path_field}")
            digest = _sha(record[sha_field], f"capture record {sha_field}")
            if artifacts.get(str(Path(record[path_field]))) != digest or _hash_file(source) != digest:
                raise PreservationPlanError("capture record artifact binding differs")
        receipt_path = _source_path(
            capture_root,
            record["base_logits_capture_receipt_path"],
            "capture receipt path",
        )
        receipt = _object(_load_json(receipt_path, "capture receipt"), "capture receipt")
        _exact(
            receipt,
            {
                "format",
                "record_id",
                "base_logits_sha256",
                "input_sha256",
                "representation",
                "top_k",
                "temperature",
                "sequence_length",
                "assistant_position_count",
                *_IDENTITY_FIELDS,
                "self_sha256",
            },
            "capture receipt",
        )
        if receipt["format"] != PRESERVATION_CAPTURE_RECEIPT_FORMAT:
            raise PreservationPlanError("unsupported capture receipt")
        if (
            receipt["representation"] != PRESERVATION_COMPACT_CAPTURE_REPRESENTATION
            or receipt["top_k"] != bridge["top_k"]
            or receipt["temperature"] != bridge["temperature"]
        ):
            raise PreservationPlanError("capture receipt representation differs from bridge")
        receipt_unsigned = dict(receipt)
        receipt_claimed = _sha(receipt_unsigned.pop("self_sha256"), "capture receipt self SHA")
        if _hash_json(receipt_unsigned) != receipt_claimed:
            raise PreservationPlanError("capture receipt hash mismatch")
        expected_receipt = {
            "record_id": record["record_id"],
            "base_logits_sha256": record["base_logits_sha256"],
            "input_sha256": record["input_sha256"],
            **{field: bridge[field] for field in _IDENTITY_FIELDS},
        }
        if any(receipt.get(field) != expected for field, expected in expected_receipt.items()):
            raise PreservationPlanError("capture receipt binding differs from bridge")
        normalized_records.append(record)
    return run, normalized_records


def _copy_tree_no_symlinks(source: Path, destination: Path) -> None:
    if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
        raise PreservationPlanError("frozen input tree must not contain symlinks")
    shutil.copytree(source, destination)


def materialize_post_capture_plan(
    bridge_path: Path | str,
    capture_root: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Resolve a bridge and verified capture into a materializer source packet."""

    try:
        bridge_file = Path(bridge_path).resolve(strict=True)
        captured = Path(capture_root).resolve(strict=True)
    except OSError as error:
        raise PreservationPlanError("bridge or capture root is missing") from error
    if not captured.is_dir() or Path(capture_root).is_symlink():
        raise PreservationPlanError("capture root must be a regular directory")
    bridge = _validate_bridge(bridge_file)
    _, captured_records = _validate_capture_run(captured, bridge)
    output, staging = _new_output(output_dir)
    try:
        _copy_tree_no_symlinks(bridge_file.parent / "inputs", staging / "inputs")
        materialization_records: list[dict[str, Any]] = []
        bridge_records = list(_array(bridge["records"], "bridge records"))
        for index, (bridge_value, capture_value) in enumerate(
            zip(bridge_records, captured_records, strict=True)
        ):
            bridge_record = _object(bridge_value, f"bridge records[{index}]")
            capture_record = _object(capture_value, f"capture records[{index}]")
            logits_relative = Path("captured") / "base-logits" / f"{index:04d}.safetensors"
            receipt_relative = Path("captured") / "capture-receipts" / f"{index:04d}.json"
            logits_source = _source_path(captured, capture_record["base_logits_path"], "capture logits")
            receipt_source = _source_path(
                captured,
                capture_record["base_logits_capture_receipt_path"],
                "capture receipt",
            )
            (staging / logits_relative).parent.mkdir(parents=True, exist_ok=True)
            (staging / receipt_relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(logits_source, staging / logits_relative)
            shutil.copyfile(receipt_source, staging / receipt_relative)
            materialization_records.append(
                {
                    "record_id": bridge_record["record_id"],
                    "stratum": bridge_record["stratum"],
                    "required_action_token_id": bridge_record["required_action_token_id"],
                    "input_path": bridge_record["input_path"],
                    "input_sha256": bridge_record["input_sha256"],
                    "base_logits_path": str(logits_relative),
                    "base_logits_sha256": capture_record["base_logits_sha256"],
                    "base_logits_capture_receipt_path": str(receipt_relative),
                    "base_logits_capture_receipt_sha256": capture_record[
                        "base_logits_capture_receipt_sha256"
                    ],
                }
            )
        plan = {
            "format": PRESERVATION_MATERIALIZATION_PLAN_FORMAT,
            "spec_id": bridge["spec_id"],
            "base_model_sha256": bridge["base_model_sha256"],
            "tokenizer_sha256": bridge["tokenizer_sha256"],
            "processor_sha256": bridge["processor_sha256"],
            "vision_tower_sha256": bridge["vision_tower_sha256"],
            "chat_template_sha256": bridge["chat_template_sha256"],
            "top_k": bridge["top_k"],
            "temperature": bridge["temperature"],
            "records": materialization_records,
            "tiers": bridge["tiers"],
        }
        _write_json(staging / "materialization-plan.json", plan)
        _rename_no_replace(staging, output)
        return plan
    except PreservationPlanError:
        shutil.rmtree(staging)
        raise
    except Exception as error:
        shutil.rmtree(staging)
        raise PreservationPlanError(
            f"post-capture plan materialization failed: {type(error).__name__}: {error}"
        ) from error


__all__ = [
    "PRESERVATION_PLAN_BUILD_CONFIG_FORMAT",
    "PRESERVATION_POST_CAPTURE_BRIDGE_FORMAT",
    "PreservationPlanError",
    "build_preservation_capture_plan",
    "materialize_post_capture_plan",
]
