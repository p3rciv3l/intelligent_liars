"""Lease-scoped capability-preservation collection for truth-editing trials.

The module's single public collector loads an exact preservation packet once,
then evaluates one edited model through an injected inference adapter.  The
Qwen trial runtime calls ``collect`` while its writer-edit lease is active, so
the adapter necessarily observes edited weights.  Frozen base distributions
remain local cache artifacts and are never recomputed during optimization.
"""

from __future__ import annotations

import hashlib
import base64
import json
import math
import mimetypes
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

import torch

from .truth_editing_preservation import (
    PRESERVATION_RECEIPT_FORMAT,
    CachedPreservationBaseline,
    PreservationError,
    PreservationReceipt,
    PreservationSpec,
    PreservationTier,
    VisionTowerIdentityReceipt,
    evaluate_preservation_stream,
    tensor_is_finite_in_chunks,
)
from .truth_editing_weight_editor import WriterEditError, require_unedited_writer_model


PRESERVATION_RUNTIME_CONFIG_FORMAT: Literal[
    "truth_editing_preservation_runtime_config_v1"
] = "truth_editing_preservation_runtime_config_v1"
PRESERVATION_RUNTIME_RECEIPT_FORMAT: Literal[
    "truth_editing_preservation_runtime_receipt_v1"
] = "truth_editing_preservation_runtime_receipt_v1"
PRESERVATION_BASE_REPEAT_RECEIPT_FORMAT: Literal[
    "truth_editing_preservation_base_repeat_receipt_v1"
] = "truth_editing_preservation_base_repeat_receipt_v1"
_HEX_SHA = re.compile(r"^[0-9a-f]{64}$")
_TIERS = frozenset({"trial", "promoted", "finalist"})


class PreservationRuntimeError(RuntimeError):
    """Preservation runtime evidence is incomplete, substituted, or tampered."""


def _canonical_bytes(value: Any) -> bytes:
    normalized = _canonical_json_value(value)
    try:
        return json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PreservationRuntimeError("value is not canonical JSON") from error


def _canonical_json_value(value: Any, path: str = "$") -> Any:
    """Own accepted Mapping containers and identify bad leaves without values."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PreservationRuntimeError(
                f"value is not canonical JSON at {path} (non-finite float)"
            )
        return value
    if isinstance(value, Mapping):
        owned: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PreservationRuntimeError(
                    f"value is not canonical JSON at {path} "
                    "(non-string object key)"
                )
            owned[key] = _canonical_json_value(item, f"{path}.{key}")
        return owned
    if isinstance(value, (list, tuple)):
        return [
            _canonical_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise PreservationRuntimeError(
        f"value is not canonical JSON at {path} "
        f"(unsupported {type(value).__name__})"
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreservationRuntimeError(f"{name} must be an object")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise PreservationRuntimeError(
            f"{name} fields differ; missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX_SHA.fullmatch(value) is None:
        raise PreservationRuntimeError(f"{name} must be a lowercase SHA-256")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PreservationRuntimeError(f"{name} must be a nonempty trimmed string")
    return value


def _regular_file(path: Path, name: str, *, root: Path | None = None) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PreservationRuntimeError(f"{name} is missing or unreadable") from error
    if path.is_symlink() or not resolved.is_file():
        raise PreservationRuntimeError(f"{name} must be a regular non-symlink file")
    if root is not None and not resolved.is_relative_to(root.resolve(strict=True)):
        raise PreservationRuntimeError(f"{name} resolves outside its allowed directory")
    return resolved


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class EditedPreservationOutput:
    """Identity-bound edited logits returned by an inference adapter."""

    record_id: str
    prompt_sha256: str
    chat_template_sha256: str
    direct_target: bool
    logits: torch.Tensor

    def __post_init__(self) -> None:
        _text(self.record_id, "edited output.record_id")
        _sha(self.prompt_sha256, "edited output.prompt_sha256")
        _sha(self.chat_template_sha256, "edited output.chat_template_sha256")
        if not isinstance(self.direct_target, bool):
            raise PreservationRuntimeError("edited output.direct_target must be boolean")
        if not isinstance(self.logits, torch.Tensor) or self.logits.ndim != 3:
            raise PreservationRuntimeError("edited output logits must be rank three")
        if not tensor_is_finite_in_chunks(self.logits):
            raise PreservationRuntimeError("edited output logits must be finite")


@dataclass(frozen=True)
class FrozenMediaReference:
    media_id: str
    media_type: Literal["image", "video", "recorded_computer_use_trace"]
    path: Path
    sha256: str
    content: bytes

    # The worker keeps one verified media payload for its entire lifetime.
    # Cache only the expensive serialization result; ``resolved_content_block``
    # still returns a fresh mapping on every call, so a processor/backend cannot
    # mutate the cached evidence.  The source bytes and digest remain available
    # to ``verify_current`` on every trial.
    _encoded_content: str | None = field(default=None, init=False, compare=False, repr=False)

    def verify_current(self) -> None:
        try:
            current = self.path.read_bytes()
        except OSError as error:
            raise PreservationRuntimeError("preservation media disappeared after loading") from error
        if hashlib.sha256(current).hexdigest() != self.sha256:
            raise PreservationRuntimeError("preservation media changed after loading")

    def resolved_content_block(self) -> dict[str, Any]:
        if self.media_type == "recorded_computer_use_trace":
            try:
                text = self.content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise PreservationRuntimeError(
                    "recorded computer-use trace must be UTF-8"
                ) from error
            return {"type": "text", "text": text}
        mime = mimetypes.guess_type(self.path.name)[0]
        if mime is None or not mime.startswith(f"{self.media_type}/"):
            mime = f"{self.media_type}/png" if self.media_type == "image" else "video/mp4"
        encoded = self._encoded_content
        if encoded is None:
            encoded = base64.b64encode(self.content).decode("ascii")
            object.__setattr__(self, "_encoded_content", encoded)
        return {
            "type": self.media_type,
            self.media_type: f"data:{mime};base64,{encoded}",
        }


@dataclass(frozen=True)
class FrozenPreservationInput:
    """Verified messages and local content-addressed media for one record."""

    record_id: str
    messages: tuple[Mapping[str, Any], ...]
    media: tuple[FrozenMediaReference, ...]
    source_sha256: str

    def identity_mapping(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_sha256": self.source_sha256,
            "media": [
                {
                    "media_id": item.media_id,
                    "media_type": item.media_type,
                    "sha256": item.sha256,
                }
                for item in self.media
            ],
        }

    def resolved_messages(self) -> tuple[dict[str, Any], ...]:
        """Return Qwen chat messages with media IDs replaced by verified paths."""

        media = {item.media_id: item for item in self.media}
        resolved: list[dict[str, Any]] = []
        for message in self.messages:
            content = message["content"]
            if isinstance(content, str):
                resolved_content: Any = content
            else:
                resolved_content = []
                for block in content:
                    if block["type"] == "text":
                        resolved_content.append(dict(block))
                    else:
                        reference = media[block["media_id"]]
                        resolved_content.append(reference.resolved_content_block())
            resolved.append({"role": message["role"], "content": resolved_content})
        return tuple(resolved)


class PreservationInferenceBackend(Protocol):
    """Adapter seam for model-specific preservation input rendering/inference."""

    @property
    def identity(self) -> Mapping[str, Any]: ...

    def infer_edited_logits(
        self,
        bundle: Any,
        *,
        record_id: str,
        input_payload: FrozenPreservationInput,
        expected_prompt_sha256: str,
        expected_chat_template_sha256: str,
    ) -> EditedPreservationOutput: ...

    def vision_tower_sha256(self, bundle: Any) -> str: ...


@dataclass(frozen=True)
class _BaselineLocation:
    record_id: str
    path: Path
    cache_sha256: str
    input_path: Path
    input_sha256: str


@dataclass(frozen=True)
class PreservationRuntimeConfig:
    spec_path: Path
    tier: PreservationTier
    chat_template_sha256: str
    base_vision_tower_sha256: str
    baselines: tuple[_BaselineLocation, ...]
    config_sha256: str

    @classmethod
    def load(cls, path: Path) -> PreservationRuntimeConfig:
        config_path = _regular_file(Path(path), "preservation runtime config")
        try:
            raw_value = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PreservationRuntimeError("preservation runtime config is unreadable") from error
        raw = _object(raw_value, "preservation runtime config")
        _exact(
            raw,
            {
                "format",
                "spec_path",
                "tier",
                "chat_template_sha256",
                "base_vision_tower_sha256",
                "baselines",
            },
            "preservation runtime config",
        )
        if raw["format"] != PRESERVATION_RUNTIME_CONFIG_FORMAT:
            raise PreservationRuntimeError("unsupported preservation runtime config format")
        tier = _text(raw["tier"], "preservation runtime tier")
        if tier not in _TIERS:
            raise PreservationRuntimeError("unknown preservation runtime tier")
        baseline_values = raw["baselines"]
        if isinstance(baseline_values, (str, bytes)) or not isinstance(
            baseline_values, Sequence
        ):
            raise PreservationRuntimeError("preservation baselines must be an array")
        baselines: list[_BaselineLocation] = []
        for index, value in enumerate(baseline_values):
            item = _object(value, f"baselines[{index}]")
            _exact(
                item,
                {"record_id", "path", "cache_sha256", "input_path", "input_sha256"},
                f"baselines[{index}]",
            )
            relative = Path(_text(item["path"], f"baselines[{index}].path"))
            input_relative = Path(
                _text(item["input_path"], f"baselines[{index}].input_path")
            )
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or input_relative.is_absolute()
                or ".." in input_relative.parts
            ):
                raise PreservationRuntimeError("baseline paths must stay below the config directory")
            baselines.append(
                _BaselineLocation(
                    record_id=_text(item["record_id"], f"baselines[{index}].record_id"),
                    path=_regular_file(
                        config_path.parent / relative,
                        f"baselines[{index}].path",
                        root=config_path.parent,
                    ),
                    cache_sha256=_sha(
                        item["cache_sha256"], f"baselines[{index}].cache_sha256"
                    ),
                    input_path=_regular_file(
                        config_path.parent / input_relative,
                        f"baselines[{index}].input_path",
                        root=config_path.parent,
                    ),
                    input_sha256=_sha(
                        item["input_sha256"], f"baselines[{index}].input_sha256"
                    ),
                )
            )
        ids = tuple(item.record_id for item in baselines)
        if not ids or len(set(ids)) != len(ids):
            raise PreservationRuntimeError("baseline record IDs must be nonempty and unique")
        spec_relative = Path(_text(raw["spec_path"], "spec_path"))
        if spec_relative.is_absolute() or ".." in spec_relative.parts:
            raise PreservationRuntimeError("spec_path must stay below the config directory")
        return cls(
            spec_path=_regular_file(
                config_path.parent / spec_relative,
                "spec_path",
                root=config_path.parent,
            ),
            tier=tier,  # type: ignore[arg-type]
            chat_template_sha256=_sha(raw["chat_template_sha256"], "chat_template_sha256"),
            base_vision_tower_sha256=_sha(
                raw["base_vision_tower_sha256"], "base_vision_tower_sha256"
            ),
            baselines=tuple(baselines),
            config_sha256=_hash(raw),
        )


@dataclass(frozen=True)
class PreservationRuntimeReceipt:
    batch_sha256: str
    recipe_id: str
    model_sha256: str
    basis_set_sha256: str
    tier: PreservationTier
    collector_identity_sha256: str
    preservation_receipt: Mapping[str, Any]
    self_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format": PRESERVATION_RUNTIME_RECEIPT_FORMAT,
            "batch_sha256": self.batch_sha256,
            "recipe_id": self.recipe_id,
            "model_sha256": self.model_sha256,
            "basis_set_sha256": self.basis_set_sha256,
            "tier": self.tier,
            "collector_identity_sha256": self.collector_identity_sha256,
            "preservation_receipt": _deep_thaw(self.preservation_receipt),
            "self_sha256": self.self_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> PreservationRuntimeReceipt:
        raw = _object(value, "preservation runtime receipt")
        fields = {
            "format",
            "batch_sha256",
            "recipe_id",
            "model_sha256",
            "basis_set_sha256",
            "tier",
            "collector_identity_sha256",
            "preservation_receipt",
            "self_sha256",
        }
        _exact(raw, fields, "preservation runtime receipt")
        if raw["format"] != PRESERVATION_RUNTIME_RECEIPT_FORMAT:
            raise PreservationRuntimeError("unsupported preservation runtime receipt format")
        claimed = _sha(raw["self_sha256"], "preservation runtime receipt.self_sha256")
        unsigned = dict(raw)
        del unsigned["self_sha256"]
        if _hash(unsigned) != claimed:
            raise PreservationRuntimeError("preservation runtime receipt hash mismatch")
        tier = _text(raw["tier"], "preservation runtime receipt.tier")
        if tier not in _TIERS:
            raise PreservationRuntimeError("unknown preservation runtime receipt tier")
        preservation = _validate_preservation_receipt_mapping(
            raw["preservation_receipt"], expected_tier=tier
        )
        batch_sha256 = _sha(raw["batch_sha256"], "batch_sha256")
        recipe_id = _text(raw["recipe_id"], "recipe_id")
        basis_set_sha256 = _sha(raw["basis_set_sha256"], "basis_set_sha256")
        expected_edit_binding = _edit_binding_sha256(
            batch_sha256=batch_sha256,
            recipe_id=recipe_id,
            basis_set_sha256=basis_set_sha256,
        )
        if preservation["edited_model_sha256"] != expected_edit_binding:
            raise PreservationRuntimeError(
                "embedded edited-model identity differs from outer trial binding"
            )
        return cls(
            batch_sha256=batch_sha256,
            recipe_id=recipe_id,
            model_sha256=_sha(raw["model_sha256"], "model_sha256"),
            basis_set_sha256=basis_set_sha256,
            tier=tier,  # type: ignore[arg-type]
            collector_identity_sha256=_sha(
                raw["collector_identity_sha256"], "collector_identity_sha256"
            ),
            preservation_receipt=_deep_freeze(preservation),
            self_sha256=claimed,
        )


class TrialPreservationCollector:
    """Load one frozen packet and score edited logits for a configured tier."""

    def __init__(
        self,
        *,
        config: PreservationRuntimeConfig,
        spec: PreservationSpec,
        baselines: tuple[CachedPreservationBaseline, ...],
        inputs: Mapping[str, FrozenPreservationInput],
        backend: PreservationInferenceBackend,
    ) -> None:
        self._config = config
        self._spec = spec
        self._baselines = baselines
        self._inputs = MappingProxyType(dict(inputs))
        self._backend = backend
        backend_identity = dict(_object(backend.identity, "preservation backend identity"))
        _canonical_bytes(backend_identity)
        self._identity = MappingProxyType(
            {
                "format": "truth_editing_preservation_collector_identity_v1",
                "config_sha256": config.config_sha256,
                "spec_sha256": spec.self_sha256,
                "base_model_sha256": spec.base_model_sha256,
                "tokenizer_sha256": spec.tokenizer_sha256,
                "processor_sha256": spec.processor_sha256,
                "vision_tower_sha256": spec.vision_tower_sha256,
                "chat_template_sha256": config.chat_template_sha256,
                "tier": config.tier,
                "cache_bundle_sha256": _hash([item.to_dict() for item in baselines]),
                "input_bundle_sha256": _hash(
                    [
                        self._inputs[record.record_id].identity_mapping()
                        for record in spec.records
                    ]
                ),
                "backend_identity_sha256": _hash(backend_identity),
            }
        )

    @classmethod
    def from_config(
        cls, path: Path, *, backend: PreservationInferenceBackend
    ) -> TrialPreservationCollector:
        config = PreservationRuntimeConfig.load(path)
        try:
            spec_value = json.loads(config.spec_path.read_text(encoding="utf-8"))
            spec = PreservationSpec.from_dict(spec_value)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, PreservationError) as error:
            raise PreservationRuntimeError("preservation spec is invalid") from error
        if spec.base_model_sha256 == "0" * 64:
            raise PreservationRuntimeError("preservation spec cannot use placeholder model identity")
        if config.base_vision_tower_sha256 != spec.vision_tower_sha256:
            raise PreservationRuntimeError("cached base vision identity differs from the spec")
        locations = {item.record_id: item for item in config.baselines}
        spec_ids = {record.record_id for record in spec.records}
        if set(locations) != spec_ids:
            raise PreservationRuntimeError("baseline locations differ from preservation spec records")
        baselines: list[CachedPreservationBaseline] = []
        inputs: dict[str, FrozenPreservationInput] = {}
        try:
            for record in spec.records:
                location = locations[record.record_id]
                value = json.loads(location.path.read_text(encoding="utf-8"))
                baseline = CachedPreservationBaseline.from_dict(value)
                if baseline.record_id != record.record_id:
                    raise PreservationRuntimeError("baseline record identity differs from its location")
                if baseline.cache_sha256 != location.cache_sha256:
                    raise PreservationRuntimeError("baseline declared hash differs from cached content")
                baselines.append(baseline)
                input_value = json.loads(location.input_path.read_text(encoding="utf-8"))
                input_payload = _object(input_value, f"preservation input {record.record_id}")
                _exact(
                    input_payload,
                    {"messages", "media"},
                    f"preservation input {record.record_id}",
                )
                if _hash(input_payload) != location.input_sha256:
                    raise PreservationRuntimeError("preservation input content hash differs")
                if location.input_sha256 != record.prompt_sha256:
                    raise PreservationRuntimeError("preservation input identity differs from spec")
                messages = _parse_messages(input_payload["messages"], record.record_id)
                media = _parse_media(
                    input_payload["media"], location.input_path.parent, record.record_id
                )
                _validate_media_bindings(messages, media, record.record_id)
                _validate_stratum_modality(record.stratum, media, record.record_id)
                if record.stratum == "recorded_computer_use":
                    trace = next(
                        item
                        for item in media
                        if item.media_type == "recorded_computer_use_trace"
                    )
                    observation_only = _computer_trace_is_observation_only(trace.content)
                    if observation_only != (record.required_action_token_id is None):
                        raise PreservationRuntimeError(
                            "computer-use trace semantics differ from action-token contract"
                        )
                inputs[record.record_id] = FrozenPreservationInput(
                    record_id=record.record_id,
                    messages=messages,
                    media=media,
                    source_sha256=location.input_sha256,
                )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, PreservationError) as error:
            raise PreservationRuntimeError("cached preservation baseline is invalid") from error
        # The evaluator performs the full per-record identity check; doing a dry
        # shape check here keeps construction fail-closed without model inference.
        required = set(spec.records_for_tier(config.tier))
        if not required <= {item.record_id for item in baselines}:
            raise PreservationRuntimeError("configured tier is missing cached baselines")
        return cls(
            config=config,
            spec=spec,
            baselines=tuple(baselines),
            inputs=inputs,
            backend=backend,
        )

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    def collect(self, bundle: Any, batch: Any) -> Mapping[str, Any]:
        if batch.model_sha256 != self._spec.base_model_sha256:
            raise PreservationRuntimeError("trial model identity differs from preservation base")
        basis_set_sha256 = _sha(
            batch.basis_set.basis_set_sha256, "trial basis_set_sha256"
        )
        required_ids = self._spec.records_for_tier(self._config.tier)
        for frozen_input in self._inputs.values():
            for media in frozen_input.media:
                media.verify_current()
        records = {record.record_id: record for record in self._spec.records}
        measured_vision_sha256 = _sha(
            self._backend.vision_tower_sha256(bundle), "edited vision tower identity"
        )
        vision = VisionTowerIdentityReceipt(
            base_model_sha256=self._spec.base_model_sha256,
            edited_model_sha256=_edit_binding_sha256(
                batch_sha256=_sha(batch.batch_sha256, "trial batch_sha256"),
                recipe_id=_text(batch.recipe_id, "trial recipe_id"),
                basis_set_sha256=basis_set_sha256,
            ),
            expected_vision_tower_sha256=self._spec.vision_tower_sha256,
            base_vision_tower_sha256=self._config.base_vision_tower_sha256,
            edited_vision_tower_sha256=measured_vision_sha256,
        )

        def logits_for_record(record_id: str) -> torch.Tensor:
            record = records[record_id]
            if record.direct_target:
                raise PreservationRuntimeError("preservation record is a direct target")
            output = self._backend.infer_edited_logits(
                bundle,
                record_id=record.record_id,
                input_payload=self._inputs[record.record_id],
                expected_prompt_sha256=record.prompt_sha256,
                expected_chat_template_sha256=self._config.chat_template_sha256,
            )
            if output.direct_target:
                raise PreservationRuntimeError("backend marked preservation input as a direct target")
            if (
                output.record_id != record.record_id
                or output.prompt_sha256 != record.prompt_sha256
                or output.chat_template_sha256 != self._config.chat_template_sha256
            ):
                raise PreservationRuntimeError("edited preservation input identity was substituted")
            return output.logits

        tier_baselines = tuple(
            baseline for baseline in self._baselines if baseline.record_id in set(required_ids)
        )
        try:
            preservation = evaluate_preservation_stream(
                self._spec,
                tier_baselines,
                logits_for_record,
                tier=self._config.tier,
                vision_receipt=vision,
            )
        except PreservationError as error:
            raise PreservationRuntimeError("preservation evaluation failed closed") from error
        preservation_mapping = _preservation_receipt_mapping(preservation)
        unsigned = {
            "format": PRESERVATION_RUNTIME_RECEIPT_FORMAT,
            "batch_sha256": batch.batch_sha256,
            "recipe_id": _text(batch.recipe_id, "trial recipe_id"),
            "model_sha256": batch.model_sha256,
            "basis_set_sha256": basis_set_sha256,
            "tier": self._config.tier,
            "collector_identity_sha256": _hash(dict(self.identity)),
            "preservation_receipt": preservation_mapping,
        }
        receipt = dict(unsigned)
        receipt["self_sha256"] = _hash(unsigned)
        # Reparse before crossing the seam so only strict evidence can escape.
        return PreservationRuntimeReceipt.from_mapping(receipt).to_mapping()

    def collect_base_repeat(
        self,
        bundle: Any,
        *,
        repeat_plan_sha256: str,
        repeat_index: int,
    ) -> Mapping[str, Any]:
        """Score the verified unedited base for repeat-threshold calibration."""

        plan_sha = _sha(repeat_plan_sha256, "base repeat plan SHA")
        if isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or repeat_index < 0:
            raise PreservationRuntimeError("base repeat index must be non-negative integer")
        verified_snapshot = getattr(bundle, "verified_snapshot", None)
        expected_snapshot_fields = {
            "model_id",
            "revision",
            "model_sha256",
            "snapshot_manifest_sha256",
        }
        if (
            not isinstance(verified_snapshot, Mapping)
            or set(verified_snapshot) != expected_snapshot_fields
            or _sha(
                verified_snapshot.get("model_sha256"),
                "base repeat verified model SHA",
            )
            != self._spec.base_model_sha256
        ):
            raise PreservationRuntimeError(
                "base repeat bundle lacks the exact verified frozen snapshot identity"
            )
        _sha(
            verified_snapshot.get("snapshot_manifest_sha256"),
            "base repeat verified snapshot manifest SHA",
        )
        try:
            require_unedited_writer_model(bundle.model)
        except (AttributeError, WriterEditError) as error:
            raise PreservationRuntimeError(
                "base repeat bundle is not in verified unedited writer state"
            ) from error
        required_ids = self._spec.records_for_tier(self._config.tier)
        for frozen_input in self._inputs.values():
            for media in frozen_input.media:
                media.verify_current()
        records = {record.record_id: record for record in self._spec.records}
        measured_vision_sha256 = _sha(
            self._backend.vision_tower_sha256(bundle), "base repeat vision tower identity"
        )
        vision = VisionTowerIdentityReceipt(
            base_model_sha256=self._spec.base_model_sha256,
            edited_model_sha256=self._spec.base_model_sha256,
            expected_vision_tower_sha256=self._spec.vision_tower_sha256,
            base_vision_tower_sha256=self._config.base_vision_tower_sha256,
            edited_vision_tower_sha256=measured_vision_sha256,
        )

        def logits_for_record(record_id: str) -> torch.Tensor:
            record = records[record_id]
            output = self._backend.infer_edited_logits(
                bundle,
                record_id=record.record_id,
                input_payload=self._inputs[record.record_id],
                expected_prompt_sha256=record.prompt_sha256,
                expected_chat_template_sha256=self._config.chat_template_sha256,
            )
            if (
                output.direct_target
                or output.record_id != record.record_id
                or output.prompt_sha256 != record.prompt_sha256
                or output.chat_template_sha256 != self._config.chat_template_sha256
            ):
                raise PreservationRuntimeError(
                    "base repeat preservation input identity was substituted"
                )
            return output.logits

        tier_baselines = tuple(
            baseline
            for baseline in self._baselines
            if baseline.record_id in set(required_ids)
        )
        try:
            preservation = evaluate_preservation_stream(
                self._spec,
                tier_baselines,
                logits_for_record,
                tier=self._config.tier,
                vision_receipt=vision,
            )
        except PreservationError as error:
            raise PreservationRuntimeError(
                "base repeat preservation evaluation failed closed"
            ) from error
        unsigned = {
            "format": PRESERVATION_BASE_REPEAT_RECEIPT_FORMAT,
            "repeat_plan_sha256": plan_sha,
            "repeat_index": repeat_index,
            "base_model_sha256": self._spec.base_model_sha256,
            "tier": self._config.tier,
            "collector_identity_sha256": _hash(dict(self.identity)),
            "preservation_receipt": _preservation_receipt_mapping(preservation),
        }
        return {**unsigned, "self_sha256": _hash(unsigned)}


def _preservation_receipt_mapping(receipt: PreservationReceipt) -> dict[str, Any]:
    normalized_strata = [
        {
            "stratum": item.stratum,
            "record_count": item.record_count,
            "assistant_token_count": item.assistant_token_count,
            "forward_kl": max(0.0, item.forward_kl),
        }
        for item in receipt.strata
    ]
    total_tokens = sum(item["assistant_token_count"] for item in normalized_strata)
    # Reconstruct the weighted mean without forming ``loss * token_count``.
    # That product can overflow even when every loss and the true mean are
    # finite.  The incremental mean stays within the range of its inputs.
    aggregate = 0.0
    accumulated_tokens = 0
    for item in normalized_strata:
        next_tokens = accumulated_tokens + item["assistant_token_count"]
        aggregate += (item["forward_kl"] - aggregate) * (
            item["assistant_token_count"] / next_tokens
        )
        accumulated_tokens = next_tokens
    if accumulated_tokens != total_tokens or not math.isfinite(aggregate):
        raise PreservationRuntimeError(
            "preservation aggregate cannot be represented as finite JSON"
        )
    unsigned = {
        "format": receipt.format,
        "spec_sha256": receipt.spec_sha256,
        "edited_model_sha256": receipt.edited_model_sha256,
        "tier": receipt.tier,
        "strata": normalized_strata,
        "aggregate_kl": aggregate,
        "vision_tower_byte_identical": receipt.vision_tower_byte_identical,
    }
    return {**unsigned, "self_sha256": _hash(unsigned)}


def _edit_binding_sha256(
    *, batch_sha256: str, recipe_id: str, basis_set_sha256: str
) -> str:
    return _hash(
        {
            "format": "truth_editing_preservation_edit_binding_v1",
            "batch_sha256": batch_sha256,
            "recipe_id": recipe_id,
            "basis_set_sha256": basis_set_sha256,
        }
    )


def _parse_messages(value: Any, record_id: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise PreservationRuntimeError(f"preservation input {record_id} messages are invalid")
    messages: list[Mapping[str, Any]] = []
    for index, raw_value in enumerate(value):
        raw = _object(raw_value, f"preservation input {record_id} messages[{index}]")
        _exact(raw, {"role", "content"}, f"preservation input {record_id} messages[{index}]")
        if raw["role"] not in {"system", "user", "assistant"}:
            raise PreservationRuntimeError("preservation input message role is unsupported")
        content = raw["content"]
        if isinstance(content, str):
            if not content:
                raise PreservationRuntimeError("preservation text content must be nonempty")
            frozen_content: Any = content
        elif (
            isinstance(content, Sequence)
            and not isinstance(content, (str, bytes))
            and content
        ):
            blocks: list[Mapping[str, Any]] = []
            for block_index, block_value in enumerate(content):
                block = _object(
                    block_value,
                    f"preservation input {record_id} messages[{index}].content[{block_index}]",
                )
                block_type = block.get("type")
                if block_type == "text":
                    _exact(block, {"type", "text"}, "preservation text block")
                    _text(block["text"], "preservation text block.text")
                elif block_type in {"image", "video", "recorded_computer_use_trace"}:
                    _exact(block, {"type", "media_id"}, "preservation media block")
                    _text(block["media_id"], "preservation media block.media_id")
                else:
                    raise PreservationRuntimeError(
                        "preservation content block type is unsupported"
                    )
                blocks.append(MappingProxyType(dict(block)))
            frozen_content = tuple(blocks)
        else:
            raise PreservationRuntimeError("preservation message content is invalid")
        messages.append(
            MappingProxyType({"role": raw["role"], "content": frozen_content})
        )
    return tuple(messages)


def _parse_media(
    value: Any, parent: Path, record_id: str
) -> tuple[FrozenMediaReference, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PreservationRuntimeError(f"preservation input {record_id} media is invalid")
    media: list[FrozenMediaReference] = []
    for index, raw_value in enumerate(value):
        raw = _object(raw_value, f"preservation input {record_id} media[{index}]")
        _exact(
            raw,
            {"media_id", "media_type", "path", "sha256"},
            f"preservation input {record_id} media[{index}]",
        )
        relative = Path(_text(raw["path"], f"preservation media {record_id}.path"))
        if relative.is_absolute() or ".." in relative.parts:
            raise PreservationRuntimeError("preservation media path escapes input directory")
        path = _regular_file(
            parent / relative, f"preservation media {record_id}", root=parent
        )
        expected = _sha(raw["sha256"], f"preservation media {record_id}.sha256")
        try:
            content = path.read_bytes()
        except OSError as error:
            raise PreservationRuntimeError("preservation media is unreadable") from error
        if hashlib.sha256(content).hexdigest() != expected:
            raise PreservationRuntimeError("preservation media content hash differs")
        media_type = _media_type(raw["media_type"])
        if media_type == "recorded_computer_use_trace":
            _validate_computer_use_trace(content, record_id)
        media.append(
            FrozenMediaReference(
                media_id=_text(raw["media_id"], f"preservation media {record_id}.media_id"),
                media_type=media_type,
                path=path,
                sha256=expected,
                content=content,
            )
        )
    ids = tuple(item.media_id for item in media)
    if len(set(ids)) != len(ids):
        raise PreservationRuntimeError("preservation media IDs must be unique")
    return tuple(media)


def _validate_media_bindings(
    messages: Sequence[Mapping[str, Any]],
    media: Sequence[FrozenMediaReference],
    record_id: str,
) -> None:
    references: list[tuple[str, str]] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, tuple):
            references.extend(
                (block["media_id"], block["type"])
                for block in content
                if block["type"] in {"image", "video", "recorded_computer_use_trace"}
            )
    declared = {item.media_id: item.media_type for item in media}
    reference_ids = [media_id for media_id, _ in references]
    if len(reference_ids) != len(set(reference_ids)):
        raise PreservationRuntimeError("preservation media reference is duplicated")
    if set(reference_ids) != set(declared):
        raise PreservationRuntimeError(
            f"preservation input {record_id} media declarations and references differ"
        )
    if any(declared[media_id] != media_type for media_id, media_type in references):
        raise PreservationRuntimeError("preservation media reference type differs")


def _validate_stratum_modality(
    stratum: str, media: Sequence[FrozenMediaReference], record_id: str
) -> None:
    types = {item.media_type for item in media}
    if stratum == "text" and types:
        raise PreservationRuntimeError(f"text preservation record {record_id} cannot use media")
    if stratum == "vision" and not types.intersection({"image", "video"}):
        raise PreservationRuntimeError(
            f"vision preservation record {record_id} requires image or video media"
        )
    if stratum == "vision" and "recorded_computer_use_trace" in types:
        raise PreservationRuntimeError(
            f"vision preservation record {record_id} cannot use a computer-use trace"
        )
    if stratum == "recorded_computer_use" and "recorded_computer_use_trace" not in types:
        raise PreservationRuntimeError(
            f"recorded computer-use record {record_id} requires an explicit trace"
        )


def _media_type(
    value: Any,
) -> Literal["image", "video", "recorded_computer_use_trace"]:
    if value not in {"image", "video", "recorded_computer_use_trace"}:
        raise PreservationRuntimeError(
            "preservation media type must be image, video, or recorded_computer_use_trace"
        )
    return value


def _validate_computer_use_trace(content: bytes, record_id: str) -> None:
    try:
        raw_value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreservationRuntimeError(
            f"recorded computer-use trace {record_id} is not strict JSON"
        ) from error
    raw = _object(raw_value, f"recorded computer-use trace {record_id}")
    trace_format = raw.get("format")
    observation_only = trace_format == "recorded_computer_use_trace_v2"
    _exact(
        raw,
        {"format", "events", "semantics"} if observation_only else {"format", "events"},
        f"recorded computer-use trace {record_id}",
    )
    if observation_only:
        if raw["semantics"] != "observation_instruction_kl_only":
            raise PreservationRuntimeError("unsupported observation-only computer-use trace")
    elif trace_format != "recorded_computer_use_trace_v1":
        raise PreservationRuntimeError("unsupported recorded computer-use trace format")
    events = raw["events"]
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence) or not events:
        raise PreservationRuntimeError("recorded computer-use trace events must be nonempty")
    action_types = {"click", "keypress", "type_text", "scroll", "navigate"}
    allowed_types = action_types | {"observation", "wait"}
    observed_actions: set[str] = set()
    for index, event_value in enumerate(events):
        event = _object(event_value, f"recorded computer-use trace event {index}")
        _exact(
            event,
            {"sequence_index", "event_type", "payload"},
            f"recorded computer-use trace event {index}",
        )
        if event["sequence_index"] != index:
            raise PreservationRuntimeError(
                "recorded computer-use trace sequence indices must be contiguous"
            )
        event_type = _text(event["event_type"], "computer-use event_type")
        if event_type not in allowed_types:
            raise PreservationRuntimeError("recorded computer-use event type is unsupported")
        _object(event["payload"], "recorded computer-use event payload")
        if event_type in action_types:
            observed_actions.add(event_type)
    if not observed_actions and not observation_only:
        raise PreservationRuntimeError(
            "recorded computer-use trace must contain at least one action"
        )


def _computer_trace_is_observation_only(content: bytes) -> bool:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:  # validated first
        raise PreservationRuntimeError("recorded computer-use trace changed during load") from error
    return isinstance(value, Mapping) and value.get("format") == "recorded_computer_use_trace_v2"


def _validate_preservation_receipt_mapping(
    value: Any, *, expected_tier: str
) -> Mapping[str, Any]:
    raw = _object(value, "embedded preservation receipt")
    _exact(
        raw,
        {
            "format",
            "spec_sha256",
            "edited_model_sha256",
            "tier",
            "strata",
            "aggregate_kl",
            "vision_tower_byte_identical",
            "self_sha256",
        },
        "embedded preservation receipt",
    )
    if raw["format"] != PRESERVATION_RECEIPT_FORMAT or raw["tier"] != expected_tier:
        raise PreservationRuntimeError("embedded preservation receipt identity differs")
    claimed = _sha(raw["self_sha256"], "embedded preservation receipt.self_sha256")
    unsigned = dict(raw)
    del unsigned["self_sha256"]
    if _hash(unsigned) != claimed:
        raise PreservationRuntimeError("embedded preservation receipt hash mismatch")
    strata = raw["strata"]
    if isinstance(strata, (str, bytes)) or not isinstance(strata, Sequence):
        raise PreservationRuntimeError("embedded preservation strata must be an array")
    seen: set[str] = set()
    for index, value_item in enumerate(strata):
        item = _object(value_item, f"embedded strata[{index}]")
        _exact(
            item,
            {"stratum", "record_count", "assistant_token_count", "forward_kl"},
            f"embedded strata[{index}]",
        )
        stratum = _text(item["stratum"], f"embedded strata[{index}].stratum")
        if stratum not in {"text", "vision", "recorded_computer_use"}:
            raise PreservationRuntimeError("embedded preservation stratum is unknown")
        seen.add(stratum)
        for count_name in ("record_count", "assistant_token_count"):
            count = item[count_name]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise PreservationRuntimeError("embedded preservation counts must be positive")
        loss = item["forward_kl"]
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(loss)
            or loss < 0
        ):
            raise PreservationRuntimeError(
                "embedded preservation KL must be finite and non-negative"
            )
    if seen != {"text", "vision", "recorded_computer_use"}:
        raise PreservationRuntimeError("embedded preservation receipt must cover every stratum")
    if len(strata) != 3:
        raise PreservationRuntimeError("embedded preservation receipt has duplicate strata")
    if raw["vision_tower_byte_identical"] is not True:
        raise PreservationRuntimeError("embedded preservation vision identity is not exact")
    aggregate = raw["aggregate_kl"]
    if (
        isinstance(aggregate, bool)
        or not isinstance(aggregate, (int, float))
        or not math.isfinite(aggregate)
        or aggregate < 0
    ):
        raise PreservationRuntimeError("embedded aggregate KL must be finite and non-negative")
    expected_aggregate = sum(
        float(item["forward_kl"]) * int(item["assistant_token_count"])
        for item in strata
    ) / sum(int(item["assistant_token_count"]) for item in strata)
    if not math.isclose(float(aggregate), expected_aggregate, rel_tol=1e-12, abs_tol=1e-12):
        raise PreservationRuntimeError(
            "embedded aggregate KL differs from its assistant-token-weighted strata"
        )
    _sha(raw["spec_sha256"], "embedded preservation spec_sha256")
    _sha(raw["edited_model_sha256"], "embedded preservation edited_model_sha256")
    return raw


__all__ = [
    "EditedPreservationOutput",
    "FrozenMediaReference",
    "FrozenPreservationInput",
    "PRESERVATION_RUNTIME_CONFIG_FORMAT",
    "PRESERVATION_RUNTIME_RECEIPT_FORMAT",
    "PRESERVATION_BASE_REPEAT_RECEIPT_FORMAT",
    "PreservationInferenceBackend",
    "PreservationRuntimeConfig",
    "PreservationRuntimeError",
    "PreservationRuntimeReceipt",
    "TrialPreservationCollector",
]
