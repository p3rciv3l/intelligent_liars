"""Strict offline contracts and planning for Qwen-specific refusal directions.

The module deliberately stops before model execution.  A prompt manifest is the
only admissible row selector; a bank can only bind vectors extracted from the
exact checkpoint, tokenizer, and chat template frozen by the config.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal


CONFIG_FORMAT = "truth_editing_refusal_direction_config_v1"
PROMPT_FORMAT = "truth_editing_refusal_prompt_manifest_v1"
LAYER_RECEIPT_FORMAT = "truth_editing_refusal_direction_layer_receipt_v1"
BANK_FORMAT = "truth_editing_refusal_direction_bank_v1"
ORTHOGONAL_FORMAT = "truth_editing_truth_orthogonalized_refusal_basis_v1"
RAW_JOINT_FORMAT = "truth_editing_raw_joint_basis_manifest_v1"
PLAN_FORMAT = "truth_editing_refusal_direction_extraction_plan_v1"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_REV = re.compile(r"^[0-9a-f]{40}$")


class RefusalDirectionError(ValueError):
    """A refusal-direction artifact violated a frozen invariant."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise RefusalDirectionError("value is not canonical finite JSON") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RefusalDirectionError(f"{name} must be an object")
    return value


def _exact(value: dict[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise RefusalDirectionError(
            f"{name} fields differ; missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RefusalDirectionError(f"{name} must be a nonempty trimmed string")
    return value


def _hash(value: Any, name: str, pattern: re.Pattern[str] = _SHA) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise RefusalDirectionError(f"{name} is not a valid hash")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RefusalDirectionError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RefusalDirectionError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise RefusalDirectionError(f"{name} must be finite and >= {minimum}")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RefusalDirectionError(f"{name} must be boolean")
    return value


def _verify_self_hash(raw: dict[str, Any], name: str) -> str:
    claimed = _hash(raw.get("self_sha256"), f"{name}.self_sha256")
    unsigned = dict(raw)
    del unsigned["self_sha256"]
    if canonical_sha256(unsigned) != claimed:
        raise RefusalDirectionError(f"{name} self hash mismatch")
    return claimed


@dataclass(frozen=True)
class RefusalModelIdentity:
    repository: str
    revision: str
    model_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    decoder_layer_count: int
    hidden_width: int


@dataclass(frozen=True)
class RefusalExtractionSpec:
    transformers_version: Literal["4.57.1"]
    system_prompt: Literal["You are a helpful assistant."]
    message_layout: Literal["system_then_user_text_v1"]
    add_generation_prompt: Literal[True]
    tokenize_chat_template: Literal[False]
    response_prefix: Literal[""]
    max_new_tokens: Literal[1]
    do_sample: Literal[False]
    use_cache: Literal[False]
    output_hidden_states: Literal[True]
    return_dict_in_generate: Literal[True]
    residual_location: Literal["decoder_layer_output_first_generated_token_v1"]
    direction_formula: Literal["unit_l2(mean_harmful_minus_mean_harmless)"]
    dtype: Literal["float64"]
    layers: tuple[int, ...]


@dataclass(frozen=True)
class PromptSourceSpec:
    role: Literal["harmless", "harmful"]
    repository: str
    revision: str
    split: str
    text_field: str
    construction_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]


@dataclass(frozen=True)
class RefusalDirectionConfig:
    format: Literal["truth_editing_refusal_direction_config_v1"]
    config_id: str
    model: RefusalModelIdentity
    extraction: RefusalExtractionSpec
    sources: tuple[PromptSourceSpec, ...]
    output_root: str
    self_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(asdict(self)))


@dataclass(frozen=True)
class RefusalPromptRow:
    prompt_id: str
    role: Literal["harmless", "harmful"]
    partition: Literal["construction", "evaluation"]
    source_repository: str
    source_revision: str
    source_split: str
    source_index: int
    prompt_text: str
    formatted_prompt_sha256: str


@dataclass(frozen=True)
class RefusalPromptManifest:
    format: Literal["truth_editing_refusal_prompt_manifest_v1"]
    config_sha256: str
    rows: tuple[RefusalPromptRow, ...]
    self_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(asdict(self)))


@dataclass(frozen=True)
class RefusalDirectionLayerReceipt:
    format: Literal["truth_editing_refusal_direction_layer_receipt_v1"]
    receipt_id: str
    source_layer: int
    width: int
    construction_harmless_count: int
    construction_harmful_count: int
    harmless_mean_sha256: str
    harmful_mean_sha256: str
    vector_path: str
    vector_file_sha256: str
    vector_sha256: str
    finite: bool
    unit_norm: bool
    self_sha256: str


@dataclass(frozen=True)
class RefusalDirectionBank:
    format: Literal["truth_editing_refusal_direction_bank_v1"]
    bank_id: str
    config_sha256: str
    prompt_manifest_sha256: str
    model_sha256: str
    chat_template_sha256: str
    per_layer_receipts: tuple[RefusalDirectionLayerReceipt, ...]
    global_source_receipt_ids: tuple[str, ...]
    self_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(asdict(self)))


@dataclass(frozen=True)
class OrthogonalizedLayer:
    source_layer: int
    truth_basis_sha256: str
    raw_refusal_vector_sha256: str
    orthogonal_vector_path: str
    orthogonal_vector_sha256: str
    projection_norm: float
    residual_norm: float
    principal_angle_degrees: float
    qualified: bool


@dataclass(frozen=True)
class TruthOrthogonalizedManifest:
    format: Literal["truth_editing_truth_orthogonalized_refusal_basis_v1"]
    truth_basis_set_sha256: str
    raw_refusal_bank_sha256: str
    layers: tuple[OrthogonalizedLayer, ...]
    self_sha256: str


@dataclass(frozen=True)
class RawJointLayer:
    source_layer: int
    truth_basis_sha256: str
    raw_refusal_vector_sha256: str
    joint_basis_path: str
    joint_basis_sha256: str
    rank: int


@dataclass(frozen=True)
class RawJointBasisManifest:
    format: Literal["truth_editing_raw_joint_basis_manifest_v1"]
    diagnostic_only: Literal[True]
    truth_basis_set_sha256: str
    raw_refusal_bank_sha256: str
    layers: tuple[RawJointLayer, ...]
    self_sha256: str


@dataclass(frozen=True)
class RefusalExtractionPlan:
    format: Literal["truth_editing_refusal_direction_extraction_plan_v1"]
    config_sha256: str
    prompt_manifest_sha256: str | None
    ready: bool
    blockers: tuple[str, ...]
    requires_model_inference: Literal[True]
    prompt_count: int
    construction_prompt_count: int
    evaluation_prompt_count: int
    model_forward_prompt_count: int
    layer_count: int
    residual_vector_count: int
    estimated_residual_bytes: int
    output_direction_count: int
    self_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(asdict(self)))


def parse_refusal_direction_config(value: Any) -> RefusalDirectionConfig:
    raw = _object(value, "config")
    _exact(raw, {"format", "config_id", "model", "extraction", "sources", "output_root", "self_sha256"}, "config")
    if raw["format"] != CONFIG_FORMAT:
        raise RefusalDirectionError("config format is unsupported")
    self_hash = _verify_self_hash(raw, "config")
    model = _object(raw["model"], "config.model")
    _exact(model, {"repository", "revision", "model_sha256", "tokenizer_sha256", "chat_template_sha256", "decoder_layer_count", "hidden_width"}, "config.model")
    model_value = RefusalModelIdentity(
        repository=_text(model["repository"], "model.repository"),
        revision=_hash(model["revision"], "model.revision", _REV),
        model_sha256=_hash(model["model_sha256"], "model.model_sha256"),
        tokenizer_sha256=_hash(model["tokenizer_sha256"], "model.tokenizer_sha256"),
        chat_template_sha256=_hash(model["chat_template_sha256"], "model.chat_template_sha256"),
        decoder_layer_count=_integer(model["decoder_layer_count"], "model.decoder_layer_count", 1),
        hidden_width=_integer(model["hidden_width"], "model.hidden_width", 1),
    )
    extraction = _object(raw["extraction"], "config.extraction")
    _exact(extraction, {"transformers_version", "system_prompt", "message_layout", "add_generation_prompt", "tokenize_chat_template", "response_prefix", "max_new_tokens", "do_sample", "use_cache", "output_hidden_states", "return_dict_in_generate", "residual_location", "direction_formula", "dtype", "layers"}, "config.extraction")
    frozen_extraction = {
        "transformers_version": "4.57.1",
        "system_prompt": "You are a helpful assistant.",
        "message_layout": "system_then_user_text_v1",
        "add_generation_prompt": True,
        "tokenize_chat_template": False,
        "response_prefix": "",
        "max_new_tokens": 1,
        "do_sample": False,
        "use_cache": False,
        "output_hidden_states": True,
        "return_dict_in_generate": True,
        "residual_location": "decoder_layer_output_first_generated_token_v1",
        "direction_formula": "unit_l2(mean_harmful_minus_mean_harmless)",
        "dtype": "float64",
    }
    if any(extraction[key] != expected for key, expected in frozen_extraction.items()):
        raise RefusalDirectionError("extraction arithmetic is not the frozen v1 contract")
    if not isinstance(extraction["layers"], list):
        raise RefusalDirectionError("extraction.layers must be a list")
    layers = tuple(_integer(item, "extraction layer") for item in extraction["layers"])
    if layers != tuple(range(model_value.decoder_layer_count)):
        raise RefusalDirectionError("extraction must cover every decoder layer in order")
    if not isinstance(raw["sources"], list):
        raise RefusalDirectionError("config.sources must be a list")
    sources: list[PromptSourceSpec] = []
    for index, item in enumerate(raw["sources"]):
        source = _object(item, f"source[{index}]")
        _exact(source, {"role", "repository", "revision", "split", "text_field", "construction_range", "evaluation_range"}, f"source[{index}]")
        role = source["role"]
        if role not in {"harmless", "harmful"}:
            raise RefusalDirectionError("source role must be harmless or harmful")
        construction = _index_range(source["construction_range"], "construction range")
        evaluation = _index_range(source["evaluation_range"], "evaluation range")
        if set(construction) & set(evaluation):
            raise RefusalDirectionError("construction and evaluation source indices overlap")
        if not construction or not evaluation:
            raise RefusalDirectionError("each source needs construction and evaluation rows")
        sources.append(PromptSourceSpec(role, _text(source["repository"], "source.repository"), _hash(source["revision"], "source.revision", _REV), _text(source["split"], "source.split"), _text(source["text_field"], "source.text_field"), construction, evaluation))
    if tuple(source.role for source in sources) != ("harmless", "harmful"):
        raise RefusalDirectionError("sources must contain harmless then harmful exactly once")
    extraction_value = RefusalExtractionSpec(
        "4.57.1",
        "You are a helpful assistant.",
        "system_then_user_text_v1",
        True,
        False,
        "",
        1,
        False,
        False,
        True,
        True,
        "decoder_layer_output_first_generated_token_v1",
        "unit_l2(mean_harmful_minus_mean_harmless)",
        "float64",
        layers,
    )
    return RefusalDirectionConfig("truth_editing_refusal_direction_config_v1", _text(raw["config_id"], "config_id"), model_value, extraction_value, tuple(sources), _text(raw["output_root"], "output_root"), self_hash)


def _index_range(value: Any, name: str) -> tuple[int, ...]:
    raw = _object(value, name)
    _exact(raw, {"start", "stop"}, name)
    start = _integer(raw["start"], f"{name}.start")
    stop = _integer(raw["stop"], f"{name}.stop", 1)
    if stop <= start:
        raise RefusalDirectionError(f"{name} must be nonempty")
    return tuple(range(start, stop))


def parse_refusal_prompt_manifest(value: Any, config: RefusalDirectionConfig) -> RefusalPromptManifest:
    raw = _object(value, "prompt manifest")
    _exact(raw, {"format", "config_sha256", "rows", "self_sha256"}, "prompt manifest")
    if raw["format"] != PROMPT_FORMAT:
        raise RefusalDirectionError("prompt manifest format is unsupported")
    self_hash = _verify_self_hash(raw, "prompt manifest")
    if _hash(raw["config_sha256"], "prompt manifest.config_sha256") != config.self_sha256:
        raise RefusalDirectionError("prompt manifest config identity mismatch")
    if not isinstance(raw["rows"], list):
        raise RefusalDirectionError("prompt manifest rows must be a list")
    rows: list[RefusalPromptRow] = []
    source_keys: set[tuple[str, str, str, int]] = set()
    text_partitions: dict[str, str] = {}
    prompt_ids: set[str] = set()
    expected: set[tuple[str, str, int]] = set()
    expected_order: list[tuple[str, str, int]] = []
    source_by_role = {source.role: source for source in config.sources}
    for source in config.sources:
        expected.update((source.role, "construction", index) for index in source.construction_indices)
        expected.update((source.role, "evaluation", index) for index in source.evaluation_indices)
        expected_order.extend((source.role, "construction", index) for index in source.construction_indices)
        expected_order.extend((source.role, "evaluation", index) for index in source.evaluation_indices)
    observed: set[tuple[str, str, int]] = set()
    for index, item in enumerate(raw["rows"]):
        row = _object(item, f"prompt row[{index}]")
        _exact(row, {"prompt_id", "role", "partition", "source_repository", "source_revision", "source_split", "source_index", "prompt_text", "formatted_prompt_sha256"}, f"prompt row[{index}]")
        role = row["role"]
        partition = row["partition"]
        if role not in source_by_role or partition not in {"construction", "evaluation"}:
            raise RefusalDirectionError("prompt row role or partition is unsupported")
        source = source_by_role[role]
        source_index = _integer(row["source_index"], "prompt row.source_index")
        selection = (role, partition, source_index)
        if selection not in expected or selection in observed:
            raise RefusalDirectionError("prompt row selection differs from frozen selection")
        observed.add(selection)
        repository = _text(row["source_repository"], "prompt row.source_repository")
        revision = _hash(row["source_revision"], "prompt row.source_revision", _REV)
        split = _text(row["source_split"], "prompt row.source_split")
        if (repository, revision, split) != (source.repository, source.revision, source.split):
            raise RefusalDirectionError("prompt row source identity mismatch")
        source_key = (repository, revision, split, source_index)
        if source_key in source_keys:
            raise RefusalDirectionError("source row overlap across prompt manifest")
        source_keys.add(source_key)
        prompt_id = _text(row["prompt_id"], "prompt row.prompt_id")
        if prompt_id in prompt_ids:
            raise RefusalDirectionError("prompt_id is duplicated")
        prompt_ids.add(prompt_id)
        prompt_text = _text(row["prompt_text"], "prompt row.prompt_text")
        text_key = hashlib.sha256(prompt_text.strip().casefold().encode()).hexdigest()
        prior_partition = text_partitions.get(text_key)
        if prior_partition is not None and prior_partition != partition:
            raise RefusalDirectionError("prompt text overlap across construction and evaluation")
        text_partitions[text_key] = partition
        formatted_hash = _hash(row["formatted_prompt_sha256"], "prompt row.formatted_prompt_sha256")
        expected_formatted = _formatted_prompt_identity(config, prompt_text)
        if formatted_hash != expected_formatted:
            raise RefusalDirectionError("formatted prompt identity mismatch")
        rows.append(RefusalPromptRow(prompt_id, role, partition, repository, revision, split, source_index, prompt_text, formatted_hash))
    if observed != expected:
        raise RefusalDirectionError("prompt row selection is incomplete")
    actual_order = [(row.role, row.partition, row.source_index) for row in rows]
    if actual_order != expected_order:
        raise RefusalDirectionError("prompt rows differ from frozen source order")
    return RefusalPromptManifest("truth_editing_refusal_prompt_manifest_v1", config.self_sha256, tuple(rows), self_hash)


def _formatted_prompt_identity(config: RefusalDirectionConfig, prompt_text: str) -> str:
    return canonical_sha256(
        {
            "chat_template_sha256": config.model.chat_template_sha256,
            "transformers_version": config.extraction.transformers_version,
            "messages": [
                {"role": "system", "content": config.extraction.system_prompt},
                {"role": "user", "content": prompt_text},
            ],
            "add_generation_prompt": config.extraction.add_generation_prompt,
            "tokenize": config.extraction.tokenize_chat_template,
            "response_prefix": config.extraction.response_prefix,
        }
    )


def _parse_layer_receipt(value: Any, config: RefusalDirectionConfig, harmless_count: int, harmful_count: int, name: str) -> RefusalDirectionLayerReceipt:
    raw = _object(value, name)
    _exact(raw, {"format", "receipt_id", "source_layer", "width", "construction_harmless_count", "construction_harmful_count", "harmless_mean_sha256", "harmful_mean_sha256", "vector_path", "vector_file_sha256", "vector_sha256", "finite", "unit_norm", "self_sha256"}, name)
    if raw["format"] != LAYER_RECEIPT_FORMAT:
        raise RefusalDirectionError(f"{name} format is unsupported")
    self_hash = _verify_self_hash(raw, name)
    layer = _integer(raw["source_layer"], f"{name}.source_layer")
    width = _integer(raw["width"], f"{name}.width", 1)
    if layer not in config.extraction.layers or width != config.model.hidden_width:
        raise RefusalDirectionError("layer receipt shape/model identity mismatch")
    good = _integer(raw["construction_harmless_count"], f"{name}.construction_harmless_count", 1)
    bad = _integer(raw["construction_harmful_count"], f"{name}.construction_harmful_count", 1)
    if (good, bad) != (harmless_count, harmful_count):
        raise RefusalDirectionError("layer receipt construction counts mismatch")
    finite = _boolean(raw["finite"], f"{name}.finite")
    unit_norm = _boolean(raw["unit_norm"], f"{name}.unit_norm")
    if not finite or not unit_norm:
        raise RefusalDirectionError("layer receipt vector is not finite unit norm")
    return RefusalDirectionLayerReceipt("truth_editing_refusal_direction_layer_receipt_v1", _text(raw["receipt_id"], f"{name}.receipt_id"), layer, width, good, bad, _hash(raw["harmless_mean_sha256"], f"{name}.harmless_mean_sha256"), _hash(raw["harmful_mean_sha256"], f"{name}.harmful_mean_sha256"), _text(raw["vector_path"], f"{name}.vector_path"), _hash(raw["vector_file_sha256"], f"{name}.vector_file_sha256"), _hash(raw["vector_sha256"], f"{name}.vector_sha256"), finite, unit_norm, self_hash)


def parse_refusal_direction_bank(value: Any, config: RefusalDirectionConfig, prompts: RefusalPromptManifest) -> RefusalDirectionBank:
    raw = _object(value, "refusal bank")
    _exact(raw, {"format", "bank_id", "config_sha256", "prompt_manifest_sha256", "model_sha256", "chat_template_sha256", "per_layer_receipts", "global_source_receipt_ids", "self_sha256"}, "refusal bank")
    if raw["format"] != BANK_FORMAT:
        raise RefusalDirectionError("refusal bank format is unsupported")
    self_hash = _verify_self_hash(raw, "refusal bank")
    if _hash(raw["config_sha256"], "bank.config_sha256") != config.self_sha256 or _hash(raw["prompt_manifest_sha256"], "bank.prompt_manifest_sha256") != prompts.self_sha256:
        raise RefusalDirectionError("refusal bank input identity mismatch")
    if _hash(raw["model_sha256"], "bank.model_sha256") != config.model.model_sha256 or _hash(raw["chat_template_sha256"], "bank.chat_template_sha256") != config.model.chat_template_sha256:
        raise RefusalDirectionError("refusal bank model identity mismatch")
    if not isinstance(raw["per_layer_receipts"], list):
        raise RefusalDirectionError("per_layer_receipts must be a list")
    harmless_count = sum(row.role == "harmless" and row.partition == "construction" for row in prompts.rows)
    harmful_count = sum(row.role == "harmful" and row.partition == "construction" for row in prompts.rows)
    receipts = tuple(_parse_layer_receipt(item, config, harmless_count, harmful_count, f"layer receipt[{index}]") for index, item in enumerate(raw["per_layer_receipts"]))
    if tuple(receipt.source_layer for receipt in receipts) != config.extraction.layers:
        raise RefusalDirectionError("bank must contain every configured layer in order")
    ids = tuple(receipt.receipt_id for receipt in receipts)
    if len(set(ids)) != len(ids):
        raise RefusalDirectionError("layer receipt ids are duplicated")
    if not isinstance(raw["global_source_receipt_ids"], list):
        raise RefusalDirectionError("global_source_receipt_ids must be a list")
    global_ids = tuple(_text(item, "global source receipt id") for item in raw["global_source_receipt_ids"])
    if global_ids != ids:
        raise RefusalDirectionError("global direction choices must reference every layer receipt in order")
    return RefusalDirectionBank("truth_editing_refusal_direction_bank_v1", _text(raw["bank_id"], "bank_id"), config.self_sha256, prompts.self_sha256, config.model.model_sha256, config.model.chat_template_sha256, receipts, global_ids, self_hash)


def parse_truth_orthogonalized_manifest(value: Any) -> TruthOrthogonalizedManifest:
    raw = _object(value, "orthogonalized manifest")
    _exact(raw, {"format", "truth_basis_set_sha256", "raw_refusal_bank_sha256", "layers", "self_sha256"}, "orthogonalized manifest")
    if raw["format"] != ORTHOGONAL_FORMAT:
        raise RefusalDirectionError("orthogonalized manifest format is unsupported")
    self_hash = _verify_self_hash(raw, "orthogonalized manifest")
    layers = _parse_composition_layers(raw["layers"], orthogonal=True)
    return TruthOrthogonalizedManifest("truth_editing_truth_orthogonalized_refusal_basis_v1", _hash(raw["truth_basis_set_sha256"], "truth basis hash"), _hash(raw["raw_refusal_bank_sha256"], "raw refusal bank hash"), tuple(layers), self_hash)


def parse_raw_joint_basis_manifest(value: Any) -> RawJointBasisManifest:
    raw = _object(value, "raw joint manifest")
    _exact(raw, {"format", "diagnostic_only", "truth_basis_set_sha256", "raw_refusal_bank_sha256", "layers", "self_sha256"}, "raw joint manifest")
    if raw["format"] != RAW_JOINT_FORMAT:
        raise RefusalDirectionError("raw joint manifest format is unsupported")
    if raw["diagnostic_only"] is not True:
        raise RefusalDirectionError("raw joint basis must remain diagnostic_only")
    self_hash = _verify_self_hash(raw, "raw joint manifest")
    layers = _parse_composition_layers(raw["layers"], orthogonal=False)
    return RawJointBasisManifest("truth_editing_raw_joint_basis_manifest_v1", True, _hash(raw["truth_basis_set_sha256"], "truth basis hash"), _hash(raw["raw_refusal_bank_sha256"], "raw refusal bank hash"), tuple(layers), self_hash)


def _parse_composition_layers(value: Any, *, orthogonal: bool) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise RefusalDirectionError("composition layers must be a nonempty list")
    result: list[Any] = []
    for index, item in enumerate(value):
        raw = _object(item, f"composition layer[{index}]")
        common = {"source_layer", "truth_basis_sha256", "raw_refusal_vector_sha256"}
        if orthogonal:
            _exact(raw, common | {"orthogonal_vector_path", "orthogonal_vector_sha256", "projection_norm", "residual_norm", "principal_angle_degrees", "qualified"}, f"composition layer[{index}]")
            angle = _number(raw["principal_angle_degrees"], "principal angle")
            if angle > 90.0:
                raise RefusalDirectionError("principal angle must be <= 90 degrees")
            result.append(OrthogonalizedLayer(_integer(raw["source_layer"], "source_layer"), _hash(raw["truth_basis_sha256"], "truth basis hash"), _hash(raw["raw_refusal_vector_sha256"], "refusal vector hash"), _text(raw["orthogonal_vector_path"], "orthogonal vector path"), _hash(raw["orthogonal_vector_sha256"], "orthogonal vector hash"), _number(raw["projection_norm"], "projection norm"), _number(raw["residual_norm"], "residual norm"), angle, _boolean(raw["qualified"], "qualified")))
        else:
            _exact(raw, common | {"joint_basis_path", "joint_basis_sha256", "rank"}, f"composition layer[{index}]")
            result.append(RawJointLayer(_integer(raw["source_layer"], "source_layer"), _hash(raw["truth_basis_sha256"], "truth basis hash"), _hash(raw["raw_refusal_vector_sha256"], "refusal vector hash"), _text(raw["joint_basis_path"], "joint basis path"), _hash(raw["joint_basis_sha256"], "joint basis hash"), _integer(raw["rank"], "joint rank", 1)))
    layers = tuple(item.source_layer for item in result)
    if layers != tuple(sorted(set(layers))):
        raise RefusalDirectionError("composition layers must be sorted unique")
    return result


def build_refusal_extraction_plan(config: RefusalDirectionConfig, prompts: RefusalPromptManifest | None) -> RefusalExtractionPlan:
    construction_count = sum(len(source.construction_indices) for source in config.sources)
    evaluation_count = sum(len(source.evaluation_indices) for source in config.sources)
    prompt_count = construction_count + evaluation_count
    blockers: tuple[str, ...] = () if prompts is not None else ("materialized_prompt_manifest_missing",)
    prompt_hash = prompts.self_sha256 if prompts is not None else None
    layer_count = len(config.extraction.layers)
    unsigned = {
        "format": PLAN_FORMAT,
        "config_sha256": config.self_sha256,
        "prompt_manifest_sha256": prompt_hash,
        "ready": not blockers,
        "blockers": list(blockers),
        "requires_model_inference": True,
        "prompt_count": prompt_count,
        "construction_prompt_count": construction_count,
        "evaluation_prompt_count": evaluation_count,
        "model_forward_prompt_count": construction_count,
        "layer_count": layer_count,
        "residual_vector_count": construction_count * layer_count,
        "estimated_residual_bytes": construction_count * layer_count * config.model.hidden_width * 8,
        "output_direction_count": layer_count,
    }
    return RefusalExtractionPlan("truth_editing_refusal_direction_extraction_plan_v1", config.self_sha256, prompt_hash, not blockers, blockers, True, prompt_count, construction_count, evaluation_count, construction_count, layer_count, construction_count * layer_count, construction_count * layer_count * config.model.hidden_width * 8, layer_count, canonical_sha256(unsigned))
