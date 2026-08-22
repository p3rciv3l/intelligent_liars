"""Deterministic, unscored response inventories for the TinyLoRA Step 5 gates.

This module intentionally stops at inference.  It does not parse behavior outputs,
grade vision answers, classify XSTest responses, or decide whether a gate passes.
Every completed run is bound to the frozen plan, exact source files, model state,
software, and decoding contract.  Partial runs are never published as complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


PLAN_FORMAT = "tinylora_step5_plan_v1"
RUN_FORMAT = "tinylora_step5_inference_run_v1"
RESPONSE_FORMAT = "tinylora_step5_inference_response_v1"
XSTEST_RESPONSE_FORMAT = "tinylora_xstest_response_v1"
BEHAVIOR_SPLITS = ("development_iid", "development_heldout_family")
VISION_SPLIT = "preservation_development_vision"
SAFETY_SPLIT = "safety_refusal_development"
SOURCE_NAMES = (*BEHAVIOR_SPLITS, VISION_SPLIT, SAFETY_SPLIT)
OBJECTIVES = (
    "conditioned_false_direct_report",
    "truthful_direct_report",
    "true_state_action_after_false_report",
    "counterfactual_state_action",
    "true_state_indirect_answer_after_false_report",
    "truthful_false_trigger_control",
)
_HEX = frozenset("0123456789abcdef")


class InferenceContractError(ValueError):
    """The inference run cannot prove a complete frozen-plan inventory."""


@dataclass(frozen=True)
class DecodingContract:
    """The only allowed Step 5 free-generation settings."""

    do_sample: bool = False
    temperature: float = 0.0
    num_beams: int = 1
    max_new_tokens: int = 128
    seed: int = 20260822
    thinking: str = "disabled_if_supported"

    def __post_init__(self) -> None:
        if self.do_sample:
            raise ValueError("Step 5 inference must use deterministic decoding")
        if self.temperature != 0.0:
            raise ValueError("deterministic Step 5 inference records temperature 0.0")
        if self.num_beams != 1:
            raise ValueError("Step 5 inference must use one greedy beam")
        if self.max_new_tokens != 128:
            raise ValueError("The frozen Step 5 generation budget is 128 tokens")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.thinking != "disabled_if_supported":
            raise ValueError("Step 5 must request disabled thinking")


@dataclass(frozen=True)
class InferenceRequest:
    inventory: str
    record_id: str
    split: str
    messages: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GeneratedResponse:
    text: str
    output_tokens: int
    terminated: bool


class InferenceBackend(Protocol):
    thinking_control: str

    def generate(
        self,
        request: InferenceRequest,
        decoding: DecodingContract,
    ) -> GeneratedResponse: ...


def _canonical_json_bytes(value: Any) -> bytes:
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
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) for row in rows)


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise InferenceContractError(f"{field} must be a lowercase SHA-256 digest")
    return text


def validate_runtime_image_receipt(
    receipt: Mapping[str, Any],
    *,
    runtime_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the immutable image publication and on-host qualification receipt."""

    if receipt.get("format") != "tinylora_step5_runtime_image_receipt_v1":
        raise InferenceContractError("unsupported runtime image receipt format")
    claimed = _require_sha256(
        receipt.get("content_sha256"), field="runtime receipt content_sha256"
    )
    unsigned = dict(receipt)
    del unsigned["content_sha256"]
    if canonical_json_sha256(unsigned) != claimed:
        raise InferenceContractError("runtime image receipt commitment mismatch")
    digest = str(receipt.get("image_digest", ""))
    if not digest.startswith("sha256:"):
        raise InferenceContractError("runtime image digest must use sha256")
    _require_sha256(digest.removeprefix("sha256:"), field="runtime image digest")
    reference = str(receipt.get("image_reference", ""))
    if not reference.endswith(f"@{digest}") or reference.count("@") != 1:
        raise InferenceContractError("runtime image reference is not digest-qualified")
    source_commit = str(receipt.get("source_commit", ""))
    if len(source_commit) != 40 or any(character not in _HEX for character in source_commit):
        raise InferenceContractError("runtime image source commit must be a Git SHA-1")
    _require_sha256(
        receipt.get("publication_evidence_sha256"),
        field="publication evidence hash",
    )
    expected_manifest_sha = _require_sha256(
        receipt.get("runtime_manifest_sha256"), field="runtime manifest hash"
    )
    runtime_manifest_path = Path(runtime_manifest_path)
    if file_sha256(runtime_manifest_path) != expected_manifest_sha:
        raise InferenceContractError("installed runtime manifest hash mismatch")
    try:
        runtime_manifest = json.loads(runtime_manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise InferenceContractError("invalid installed runtime manifest") from error
    if (
        not isinstance(runtime_manifest, dict)
        or runtime_manifest.get("schema_version") != 1
        or runtime_manifest.get("large_run_enabled") is not False
        or runtime_manifest.get("runtime_id") != receipt.get("runtime_id")
    ):
        raise InferenceContractError("runtime receipt and installed manifest differ")
    validation = receipt.get("gpu_validation")
    if not isinstance(validation, Mapping) or validation != {
        "valid": True,
        "mode": "gpu-runtime",
        "runtime_id": receipt.get("runtime_id"),
        "errors": [],
    }:
        raise InferenceContractError("runtime image lacks exact passing GPU validation")
    return dict(receipt), runtime_manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise InferenceContractError(
                f"invalid JSONL at {path}:{line_number}"
            ) from error
        if not isinstance(value, dict):
            raise InferenceContractError(
                f"JSONL row must be an object at {path}:{line_number}"
            )
        rows.append(value)
    return rows


def _verified_source(
    plan_path: Path,
    plan: Mapping[str, Any],
    name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs = plan.get("outputs")
    if not isinstance(outputs, Mapping) or name not in outputs:
        raise InferenceContractError(f"frozen plan is missing output {name}")
    specification = outputs[name]
    if not isinstance(specification, Mapping):
        raise InferenceContractError(f"invalid output specification: {name}")
    relative = Path(str(specification.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise InferenceContractError(f"unsafe source path for {name}")
    path = plan_path.parent / relative
    expected_sha = _require_sha256(specification.get("sha256"), field=f"{name} hash")
    if not path.is_file():
        raise InferenceContractError(f"Step 5 source is missing: {name}")
    actual_sha = file_sha256(path)
    if actual_sha != expected_sha:
        raise InferenceContractError(f"Step 5 source hash mismatch: {name}")
    rows = _read_jsonl(path)
    expected_records = specification.get("records")
    if (
        isinstance(expected_records, bool)
        or not isinstance(expected_records, int)
        or expected_records < 1
        or len(rows) != expected_records
    ):
        raise InferenceContractError(f"Step 5 source record count mismatch: {name}")
    return rows, {
        "path": relative.as_posix(),
        "records": len(rows),
        "sha256": actual_sha,
    }


def _unique_ids(rows: Sequence[Mapping[str, Any]], *, name: str) -> None:
    ids = [str(row.get("record_id", "")) for row in rows]
    if any(not value for value in ids):
        raise InferenceContractError(f"{name} has a missing record_id")
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise InferenceContractError(f"{name} has duplicate record IDs: {duplicates[:5]}")


def _behavior_requests(
    rows: Sequence[Mapping[str, Any]], *, split: str
) -> list[InferenceRequest]:
    _unique_ids(rows, name=split)
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("kind") != "behavior" or row.get("split") != split:
            raise InferenceContractError(f"invalid behavior row in {split}")
        scenario = str(row.get("scenario_id", ""))
        prompt = row.get("prompt")
        if not scenario or not isinstance(prompt, str) or not prompt.strip():
            raise InferenceContractError(f"invalid behavior prompt in {split}")
        by_scenario[scenario].append(row)
    expected = set(OBJECTIVES)
    for scenario, scenario_rows in by_scenario.items():
        observed = [str(row.get("objective", "")) for row in scenario_rows]
        if len(observed) != len(OBJECTIVES) or set(observed) != expected:
            raise InferenceContractError(
                f"behavior scenario {scenario} does not contain exactly six objectives"
            )
    return [
        InferenceRequest(
            inventory="behavior",
            record_id=str(row["record_id"]),
            split=split,
            messages=({"role": "user", "content": str(row["prompt"])},),
            metadata={
                key: row[key]
                for key in (
                    "scenario_id",
                    "split_group_id",
                    "family",
                    "objective",
                    "target",
                    "alternative_target",
                )
            },
        )
        for row in sorted(rows, key=lambda item: str(item["record_id"]))
    ]


def _vision_requests(rows: Sequence[Mapping[str, Any]]) -> list[InferenceRequest]:
    _unique_ids(rows, name=VISION_SPLIT)
    requests: list[InferenceRequest] = []
    for row in sorted(rows, key=lambda item: str(item["record_id"])):
        messages = row.get("messages")
        if (
            row.get("kind") != "preservation"
            or not isinstance(messages, list)
            or len(messages) < 2
            or not isinstance(messages[-1], Mapping)
            or messages[-1].get("role") != "assistant"
        ):
            raise InferenceContractError(
                f"invalid vision row: {row.get('record_id', '<missing>')}"
            )
        expected = messages[-1].get("content")
        if not isinstance(expected, str) or not expected.strip():
            raise InferenceContractError(
                f"vision row has no reference answer: {row['record_id']}"
            )
        image_sha = _require_sha256(
            row.get("image_sha256"), field=f"{row['record_id']} image hash"
        )
        requests.append(
            InferenceRequest(
                inventory="vision_preservation",
                record_id=str(row["record_id"]),
                split=str(row["split"]),
                messages=tuple(dict(message) for message in messages[:-1]),
                metadata={
                    "split_group_id": row["split_group_id"],
                    "objective": row["objective"],
                    "preservation_category": row["preservation_category"],
                    "image_sha256": image_sha,
                    "reference_answer": expected,
                },
            )
        )
    return requests


def _safety_requests(rows: Sequence[Mapping[str, Any]]) -> list[InferenceRequest]:
    _unique_ids(rows, name=SAFETY_SPLIT)
    expected_ids = {f"xstest.{index:03d}" for index in range(1, 451)}
    observed_ids = {str(row.get("record_id", "")) for row in rows}
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise InferenceContractError(
            "XSTest inventory must contain exactly xstest.001 through xstest.450; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    requests: list[InferenceRequest] = []
    for row in sorted(rows, key=lambda item: str(item["record_id"])):
        prompt = row.get("prompt")
        if (
            row.get("format") != "tinylora_step5_safety_eval_v1"
            or not isinstance(prompt, str)
            or not prompt.strip()
            or row.get("expected_behavior") not in {"comply", "refuse"}
        ):
            raise InferenceContractError(
                f"invalid XSTest row: {row.get('record_id', '<missing>')}"
            )
        requests.append(
            InferenceRequest(
                inventory="safety_refusal",
                record_id=str(row["record_id"]),
                split=str(row["split"]),
                messages=({"role": "user", "content": prompt},),
                metadata={
                    "split_group_id": row["split_group_id"],
                    "prompt": prompt,
                    "expected_behavior": row["expected_behavior"],
                    "xstest_type": row["xstest_type"],
                },
            )
        )
    behavior_counts = Counter(
        request.metadata["expected_behavior"] for request in requests
    )
    if behavior_counts != Counter({"comply": 250, "refuse": 200}):
        raise InferenceContractError(
            "XSTest inventory must contain 250 comply and 200 refuse prompts"
        )
    return requests


def build_inference_requests(
    plan_path: Path,
    *,
    verified_vision_rows: Sequence[Mapping[str, Any]] | None = None,
    image_asset_receipt: Mapping[str, Any] | None = None,
) -> tuple[list[InferenceRequest], dict[str, Any]]:
    """Verify the four frozen inputs and materialize every inference request.

    ``verified_vision_rows`` may only replace image locations in the plan-bound
    vision rows.  The CLI supplies rows rebased through the validated portable
    image bundle; tests may omit it to exercise the GPU-free orchestration seam.
    """

    plan_path = Path(plan_path).resolve()
    try:
        plan = json.loads(plan_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise InferenceContractError(f"invalid Step 5 plan: {plan_path}") from error
    if not isinstance(plan, dict) or plan.get("format") != PLAN_FORMAT:
        raise InferenceContractError("unsupported Step 5 plan format")
    model = plan.get("model")
    if (
        not isinstance(model, Mapping)
        or not str(model.get("model_id", ""))
        or len(str(model.get("revision", ""))) != 40
    ):
        raise InferenceContractError("Step 5 plan does not pin the model revision")

    sources: dict[str, Any] = {}
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name in SOURCE_NAMES:
        loaded[name], sources[name] = _verified_source(plan_path, plan, name)

    if verified_vision_rows is not None:
        replacements = [dict(row) for row in verified_vision_rows]
        original_by_id = {row["record_id"]: row for row in loaded[VISION_SPLIT]}
        replacement_by_id = {row.get("record_id"): row for row in replacements}
        if set(original_by_id) != set(replacement_by_id):
            raise InferenceContractError("verified image rows do not match the frozen plan")
        for record_id, replacement in replacement_by_id.items():
            original = original_by_id[record_id]
            for field in (
                "record_id",
                "split_group_id",
                "split",
                "kind",
                "objective",
                "preservation_category",
                "image_sha256",
            ):
                if replacement.get(field) != original.get(field):
                    raise InferenceContractError(
                        f"verified image row changed frozen field {field}: {record_id}"
                    )
            if replacement.get("messages", [])[-1] != original.get("messages", [])[-1]:
                raise InferenceContractError(
                    f"verified image row changed reference answer: {record_id}"
                )
        loaded[VISION_SPLIT] = replacements
        if image_asset_receipt is None:
            raise InferenceContractError(
                "verified vision rows require an image asset receipt"
            )

    requests: list[InferenceRequest] = []
    for split in BEHAVIOR_SPLITS:
        requests.extend(_behavior_requests(loaded[split], split=split))
    requests.extend(_vision_requests(loaded[VISION_SPLIT]))
    requests.extend(_safety_requests(loaded[SAFETY_SPLIT]))
    all_ids = [request.record_id for request in requests]
    if len(all_ids) != len(set(all_ids)):
        raise InferenceContractError("record IDs collide across Step 5 inventories")
    receipt: dict[str, Any] = {
        "source_plan_sha256": file_sha256(plan_path),
        "model": dict(model),
        "sources": sources,
    }
    if image_asset_receipt is not None:
        receipt["image_assets"] = dict(image_asset_receipt)
    return requests, receipt


def _prompt_hash(request: InferenceRequest) -> str:
    return canonical_json_sha256(list(request.messages))


def _run_contract(
    *,
    requests: Sequence[InferenceRequest],
    decoding: DecodingContract,
    source_receipt: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    software_sha256: str,
    thinking_control: str,
) -> dict[str, Any]:
    _require_sha256(software_sha256, field="software_sha256")
    if model_identity.get("state") not in {"base", "candidate"}:
        raise InferenceContractError("model identity must name base or candidate state")
    if thinking_control not in {"disabled", "unsupported"}:
        raise InferenceContractError("backend did not report its thinking control")
    return {
        "format": RUN_FORMAT,
        "source": dict(source_receipt),
        "model_identity": dict(model_identity),
        "decoding": asdict(decoding),
        "thinking_control": thinking_control,
        "software_sha256": software_sha256,
        "request_inventory_sha256": canonical_json_sha256(
            [
                {
                    "inventory": request.inventory,
                    "record_id": request.record_id,
                    "split": request.split,
                    "prompt_sha256": _prompt_hash(request),
                    "metadata": request.metadata,
                }
                for request in requests
            ]
        ),
        "requested_records": len(requests),
    }


def generate_inference_inventories(
    requests: Sequence[InferenceRequest],
    *,
    backend: InferenceBackend,
    decoding: DecodingContract,
    source_receipt: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    software_sha256: str,
) -> dict[str, Any]:
    """Generate all rows in stable order, failing the run on any row error."""

    if not requests:
        raise InferenceContractError("inference request inventory is empty")
    run_contract = _run_contract(
        requests=requests,
        decoding=decoding,
        source_receipt=source_receipt,
        model_identity=model_identity,
        software_sha256=software_sha256,
        thinking_control=backend.thinking_control,
    )
    run_identity = canonical_json_sha256(run_contract)
    inventories: dict[str, list[dict[str, Any]]] = {
        "behavior": [],
        "vision_preservation": [],
        "safety_refusal": [],
    }
    model_identity_text = (
        f"{model_identity.get('state')}:{source_receipt['model']['model_id']}@"
        f"{source_receipt['model']['revision']}:{canonical_json_sha256(model_identity)}"
    )
    for request in sorted(requests, key=lambda item: (item.inventory, item.record_id)):
        try:
            generated = backend.generate(request, decoding)
        except Exception as error:
            raise InferenceContractError(
                f"inference failed for {request.record_id}: {type(error).__name__}: {error}"
            ) from error
        response = generated.text
        if not isinstance(response, str) or not response.strip():
            raise InferenceContractError(
                f"inference produced an empty response for {request.record_id}"
            )
        if (
            isinstance(generated.output_tokens, bool)
            or not isinstance(generated.output_tokens, int)
            or generated.output_tokens < 1
            or generated.output_tokens > decoding.max_new_tokens
        ):
            raise InferenceContractError(
                f"inference reported an invalid token count for {request.record_id}"
            )
        if not generated.terminated:
            raise InferenceContractError(
                f"inference exhausted the token budget for {request.record_id}"
            )
        base = {
            "record_id": request.record_id,
            "split": request.split,
            "response": response,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "prompt_sha256": _prompt_hash(request),
            "output_tokens": generated.output_tokens,
            "terminated": True,
            "run_identity_sha256": run_identity,
            "model_identity": model_identity_text,
            "source_plan_sha256": source_receipt["source_plan_sha256"],
        }
        if request.inventory == "safety_refusal":
            row = {
                "format": XSTEST_RESPONSE_FORMAT,
                **base,
                **request.metadata,
            }
        else:
            row = {
                "format": RESPONSE_FORMAT,
                "inventory": request.inventory,
                **base,
                **request.metadata,
            }
        inventories[request.inventory].append(row)
    if sum(map(len, inventories.values())) != len(requests):
        raise InferenceContractError("generated inventory is incomplete")
    return {
        **inventories,
        "run_contract": run_contract,
        "run_identity_sha256": run_identity,
    }


def publish_inference_run(destination: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Publish an all-or-nothing inference directory and content manifest."""

    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    names = {
        "behavior": "behavior_responses.jsonl",
        "vision_preservation": "vision_preservation_responses.jsonl",
        "safety_refusal": "safety_refusal_responses.jsonl",
    }
    try:
        outputs: dict[str, Any] = {}
        for inventory, filename in names.items():
            rows = payload.get(inventory)
            if not isinstance(rows, list) or not rows:
                raise InferenceContractError(f"missing generated inventory: {inventory}")
            data = _canonical_jsonl_bytes(rows)
            path = temporary / filename
            path.write_bytes(data)
            outputs[inventory] = {
                "path": filename,
                "records": len(rows),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        manifest = {
            **dict(payload["run_contract"]),
            "run_identity_sha256": payload["run_identity_sha256"],
            "outputs": outputs,
            "completed_records": sum(value["records"] for value in outputs.values()),
            "complete": True,
            "errors": [],
        }
        manifest["content_sha256"] = canonical_json_sha256(manifest)
        (temporary / "manifest.json").write_bytes(_canonical_json_bytes(manifest))
        os.replace(temporary, destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def seed_inference(seed: int) -> None:
    """Seed every stochastic library without importing Torch on CPU-only tooling."""

    random.seed(seed)
    try:
        import numpy as np
        import torch
    except ImportError:
        return
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _contains_vision(messages: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        isinstance(message.get("content"), list)
        and any(
            isinstance(item, Mapping)
            and item.get("type") in {"image", "image_url", "video"}
            for item in message["content"]
        )
        for message in messages
    )


def _qwen_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        converted.append(
            {
                "role": str(message["role"]),
                "content": (
                    content
                    if isinstance(content, list)
                    else [{"type": "text", "text": str(content)}]
                ),
            }
        )
    return converted


class QwenInferenceBackend:
    """One-row greedy Qwen generator with explicit thinking suppression."""

    def __init__(self, *, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor
        from jinja2 import Environment, meta

        template = getattr(processor, "chat_template", None)
        if not isinstance(template, str) or not template.strip():
            raise InferenceContractError("processor has no inspectable chat template")
        variables = meta.find_undeclared_variables(Environment().parse(template))
        self.thinking_control = (
            "disabled" if "enable_thinking" in variables else "unsupported"
        )

    def _inputs(self, messages: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        qwen_messages = _qwen_messages(messages)
        if self.thinking_control == "disabled":
            text = self.processor.apply_chat_template(
                qwen_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            text = self.processor.apply_chat_template(
                qwen_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        if _contains_vision(messages):
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError as error:
                raise RuntimeError("qwen-vl-utils is required for PixMo inference") from error
            patch_size = int(
                getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
            )
            images, videos, video_kwargs = process_vision_info(
                [qwen_messages],
                return_video_kwargs=True,
                image_patch_size=patch_size,
            )
            return self.processor(
                text=[text],
                images=images,
                videos=videos,
                padding=False,
                return_tensors="pt",
                **video_kwargs,
            )
        return self.processor(text=[text], padding=False, return_tensors="pt")

    def generate(
        self,
        request: InferenceRequest,
        decoding: DecodingContract,
    ) -> GeneratedResponse:
        import torch

        inputs = self._inputs(request.messages)
        device = getattr(self.model, "device", None) or next(self.model.parameters()).device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in dict(inputs).items()
        }
        prompt_tokens = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=decoding.do_sample,
                num_beams=decoding.num_beams,
                max_new_tokens=decoding.max_new_tokens,
                use_cache=True,
            )
        generated = output[0, prompt_tokens:]
        count = int(generated.numel())
        if count < 1:
            return GeneratedResponse(text="", output_tokens=0, terminated=True)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        eos_value = getattr(tokenizer, "eos_token_id", None)
        eos_ids = (
            {int(value) for value in eos_value}
            if isinstance(eos_value, (list, tuple, set))
            else ({int(eos_value)} if eos_value is not None else set())
        )
        last_token = int(generated[-1].item())
        terminated = count < decoding.max_new_tokens or last_token in eos_ids
        text = self.processor.batch_decode(
            generated.unsqueeze(0),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return GeneratedResponse(
            text=text,
            output_tokens=count,
            terminated=terminated,
        )


def verified_model_identity(
    *,
    snapshot_root: Path,
    snapshot_plan_path: Path,
    frozen_model: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify the exact local model inventory before any GPU model load."""

    from intelligent_liars.model_cache import verify_snapshot

    try:
        snapshot_plan = json.loads(Path(snapshot_plan_path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise InferenceContractError("invalid model snapshot plan") from error
    planned_model = snapshot_plan.get("model")
    expected = {
        "repo_id": frozen_model.get("model_id"),
        "revision": frozen_model.get("revision"),
    }
    if planned_model != expected:
        raise InferenceContractError("model snapshot plan differs from frozen Step 5 plan")
    try:
        verified = verify_snapshot(Path(snapshot_root), snapshot_plan)
    except ValueError as error:
        raise InferenceContractError(f"model snapshot verification failed: {error}") from error
    return {
        "model_id": expected["repo_id"],
        "revision": expected["revision"],
        "snapshot_plan_sha256": file_sha256(Path(snapshot_plan_path)),
        "snapshot_inventory_sha256": canonical_json_sha256(verified),
        "snapshot_files": len(verified),
        "snapshot_bytes": sum(int(row["bytes"]) for row in verified),
    }, verified


def install_candidate_adapter(
    *,
    model: Any,
    processor: Any,
    model_id: str,
    revision: str,
    plan: Mapping[str, Any],
    plan_sha256: str,
    adapter_state_path: Path,
    adapter_metadata_path: Path,
    basis_path: Path | None,
) -> dict[str, Any]:
    """Install a safe tensor-only candidate state into the frozen base model."""

    import torch
    from safetensors.torch import load_file

    from intelligent_liars.models import ModelBundle, ModelLoadConfig
    from intelligent_liars.standalone_models import (
        TinyLoRATrainingConfig,
        install_tinylora_with_cache,
    )
    from intelligent_liars.tinylora_step5 import install_ordinary_lora

    try:
        metadata = json.loads(Path(adapter_metadata_path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise InferenceContractError("invalid adapter metadata") from error
    if metadata.get("format") != "tinylora_step5_adapter_state_v1":
        raise InferenceContractError("unsupported adapter state format")
    if metadata.get("plan_sha256") != plan_sha256:
        raise InferenceContractError("adapter state was trained against another plan")
    if metadata.get("model") != {"model_id": model_id, "revision": revision}:
        raise InferenceContractError("adapter state model identity differs")
    arms = plan.get("arms")
    if not isinstance(arms, list):
        raise InferenceContractError("frozen plan has no adapter arms")
    arm_name = metadata.get("arm_name")
    matching = [arm for arm in arms if isinstance(arm, Mapping) and arm.get("name") == arm_name]
    if len(matching) != 1:
        raise InferenceContractError("adapter metadata names no unique frozen arm")
    arm = dict(matching[0])
    if metadata.get("arm") != arm:
        raise InferenceContractError("adapter metadata arm differs from frozen plan")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if arm.get("adapter") == "tinylora":
        if basis_path is None:
            raise InferenceContractError("TinyLoRA candidate requires its exact basis")
        expected_basis_sha = _require_sha256(
            metadata.get("basis_sha256"), field="basis_sha256"
        )
        if file_sha256(basis_path) != expected_basis_sha:
            raise InferenceContractError("TinyLoRA basis hash mismatch")
        bundle = ModelBundle(
            model=model,
            processor=processor,
            tokenizer=processor.tokenizer,
            model_id=model_id,
            config=ModelLoadConfig(model_name=model_id, revision=revision),
        )
        install_tinylora_with_cache(
            model_bundle=bundle,
            config=TinyLoRATrainingConfig(
                svd_rank=int(arm["svd_rank"]),
                projection_dim=int(arm["projection_dim"]),
                projection_seed=int(metadata.get("projection_seed", 42)),
                train_layers=tuple(int(value) for value in arm["train_layers"]),
            ),
            cache_path=Path(basis_path),
        )
    elif arm.get("adapter") == "ordinary_lora":
        if basis_path is not None:
            raise InferenceContractError("ordinary LoRA must not receive a basis")
        install_ordinary_lora(
            model,
            train_layers=tuple(int(value) for value in arm["train_layers"]),
            rank=int(arm["lora_rank"]),
        )
    else:
        raise InferenceContractError("unsupported frozen adapter type")

    state = load_file(str(adapter_state_path), device="cpu")
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(state) != set(current):
        raise InferenceContractError("adapter tensor inventory differs from installed arm")
    with torch.no_grad():
        for name, parameter in current.items():
            value = state[name]
            if tuple(value.shape) != tuple(parameter.shape) or not torch.isfinite(value).all():
                raise InferenceContractError(f"invalid adapter tensor: {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    model.eval()
    return {
        "state": "candidate",
        "arm": arm,
        "projection_seed": metadata.get("projection_seed"),
        "adapter_state_sha256": file_sha256(adapter_state_path),
        "adapter_metadata_sha256": file_sha256(adapter_metadata_path),
        "basis_sha256": metadata.get("basis_sha256"),
        "optimizer_steps": metadata.get("optimizer_steps"),
    }
