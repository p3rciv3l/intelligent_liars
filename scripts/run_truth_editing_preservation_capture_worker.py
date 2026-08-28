#!/usr/bin/env python3
"""Capture compact preservation baselines and five real base repeats per tier."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import torch

from intelligent_liars.models import ModelLoadConfig, load_model_and_processor
from intelligent_liars.truth_editing_preservation_capture import (
    VerifiedQwenBaselineCaptureBackend,
    _default_processor_identity,
    capture_preservation_baselines,
    qwen_preservation_capture_runtime_sha256,
)
from intelligent_liars.truth_editing_preservation_materialization import (
    materialize_preservation_runtime_packet,
)
from intelligent_liars.truth_editing_preservation_plan import (
    build_preservation_capture_plan,
    materialize_post_capture_plan,
)
from intelligent_liars.truth_editing_preservation_runtime import (
    EditedPreservationOutput,
    FrozenPreservationInput,
    TrialPreservationCollector,
)
from intelligent_liars.truth_editing_preservation_thresholds import (
    build_preservation_threshold_calibration,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("TRUTH_EDITING_PRESERVATION_OUTPUT", "/workspace/outputs"))
CACHE = Path(os.environ.get("HF_HOME", "/workspace/cache/huggingface"))
MANIFEST = Path(
    os.environ.get(
        "TRUTH_EDITING_MODEL_CACHE_MANIFEST",
        "/workspace/cache/model-cache-manifest.json",
    )
)
CONTRACT = ROOT / "configs/truth_editing_preservation_capture_v2_r4.json"


class QwenPreservationInferenceBackend:
    """Minimal production adapter shared by base repeats and vision identity."""

    identity = {
        "adapter": "qwen_preservation_inference_backend_v1",
        "rendering": "processor_chat_template_plus_qwen_vl_utils",
        "vision_identity": "named_parameter_name_shape_dtype_bytes_sha256_v1",
    }

    def infer_edited_logits(
        self,
        bundle: object,
        *,
        record_id: str,
        input_payload: FrozenPreservationInput,
        expected_prompt_sha256: str,
        expected_chat_template_sha256: str,
    ) -> EditedPreservationOutput:
        if input_payload.record_id != record_id or input_payload.source_sha256 != expected_prompt_sha256:
            raise RuntimeError("preservation input identity differs")
        processor = bundle.processor
        template = getattr(processor, "chat_template", None) or getattr(bundle.tokenizer, "chat_template", None)
        if not isinstance(template, str) or hashlib.sha256(template.encode()).hexdigest() != expected_chat_template_sha256:
            raise RuntimeError("preservation chat template identity differs")
        conversation = input_payload.resolved_messages()
        text = processor.apply_chat_template(
            [dict(message) for message in conversation],
            tokenize=False,
            add_generation_prompt=not (conversation and conversation[-1].get("role") == "assistant"),
        )
        kwargs: dict[str, object] = {"text": [text], "padding": True, "return_tensors": "pt"}
        if input_payload.media:
            from qwen_vl_utils import process_vision_info

            images, videos = process_vision_info(list(conversation))
            if images:
                kwargs["images"] = images
            if videos:
                kwargs["videos"] = videos
        inputs = processor(**kwargs)
        if not isinstance(inputs, Mapping) or "input_ids" not in inputs:
            raise RuntimeError("preservation processor returned no input IDs")
        model = bundle.model
        device = next(model.parameters()).device
        moved = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**moved, use_cache=False)
        return EditedPreservationOutput(
            record_id=record_id,
            prompt_sha256=input_payload.source_sha256,
            chat_template_sha256=expected_chat_template_sha256,
            direct_target=False,
            logits=output.logits.detach(),
        )

    def vision_tower_sha256(self, bundle: object) -> str:
        visual = getattr(getattr(bundle.model, "model", None), "visual", None)
        if visual is None:
            visual = getattr(bundle.model, "visual", None)
        if visual is None:
            raise RuntimeError("Qwen vision tower cannot be located")
        digest = hashlib.sha256(b"truth_editing_qwen_vision_parameters_v1\0")
        for name, parameter in visual.named_parameters():
            tensor = parameter.detach().to(device="cpu").contiguous()
            digest.update(_canonical({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}))
            digest.update(b"\0")
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_new(path: Path, value: object) -> None:
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    contract = json.loads(CONTRACT.read_text())
    model_config = ModelLoadConfig(cache_dir=str(CACHE), snapshot_manifest_path=str(MANIFEST))
    bundle = load_model_and_processor(model_config)
    processor_sha = _default_processor_identity(bundle)
    runtime_sha = qwen_preservation_capture_runtime_sha256(model_config)
    inference_backend = QwenPreservationInferenceBackend()
    vision_sha = inference_backend.vision_tower_sha256(bundle)

    identity = {
        "format": "truth_editing_preservation_capture_identity_v1",
        "contract_sha256": _sha(contract),
        "base_model_sha256": contract["model"]["model_sha256"],
        "snapshot_manifest_sha256": contract["model"]["snapshot_manifest_sha256"],
        "tokenizer_sha256": contract["model"]["tokenizer_sha256"],
        "processor_sha256": processor_sha,
        "vision_tower_sha256": vision_sha,
        "chat_template_sha256": contract["model"]["chat_template_sha256"],
        "inference_runtime_sha256": runtime_sha,
    }
    identity["self_sha256"] = _sha(identity)
    _write_new(OUTPUT / "capture-identity.json", identity)

    build_config = {
        "format": "truth_editing_preservation_plan_build_config_v1",
        "spec_id": contract["contract_id"],
        "selection_seed": contract["selection_seed"],
        "base_model_sha256": identity["base_model_sha256"],
        "tokenizer_sha256": identity["tokenizer_sha256"],
        "processor_sha256": processor_sha,
        "vision_tower_sha256": vision_sha,
        "chat_template_sha256": identity["chat_template_sha256"],
        "inference_runtime_sha256": runtime_sha,
        "batch_size": contract["capture"]["batch_size"],
        "top_k": 64,
        "temperature": 1.0,
        "sources": {
            "text": {
                "path": "corpora/tinylora_deception_action_v1/step5_v1/preservation_development_text.jsonl",
                "sha256": contract["sources"]["text"]["sha256"],
            },
            "vision": {
                "path": "corpora/tinylora_deception_action_v1/step5_v1/preservation_development_vision.jsonl",
                "sha256": contract["sources"]["vision"]["sha256"],
                "media_root": "data/tinylora_preservation_snapshots/v1/pixmo_docs_images",
            },
            "recorded_computer_use": {
                "source_root": "artifacts/truth-editing/osworld-preservation-source-v2-r4",
                "manifest_path": "manifest.jsonl",
                "manifest_sha256": contract["sources"]["recorded_computer_use"]["manifest_sha256"],
            },
        },
        "tier_counts_per_stratum": contract["capture"]["tier_counts_per_stratum"],
    }
    build_path = ROOT / ".preservation-build-v2-r4.json"
    _write_new(build_path, build_config)
    plan_root = OUTPUT / "capture-plan"
    build_preservation_capture_plan(build_path, plan_root)

    capture_backend = VerifiedQwenBaselineCaptureBackend(
        model_config=model_config,
        bundle_loader=lambda _config: bundle,
        expected_tokenizer_sha256=identity["tokenizer_sha256"],
        expected_processor_sha256=processor_sha,
        expected_chat_template_sha256=identity["chat_template_sha256"],
        enforce_production_runtime=False,
        inference_runtime_sha256=runtime_sha,
    )
    capture_root = OUTPUT / "baseline-capture"
    capture_preservation_baselines(
        plan_root / "capture-plan.json", capture_root, backend=capture_backend
    )
    source_packet = OUTPUT / "materialization-source"
    materialize_post_capture_plan(
        plan_root / "post-capture-materialization-bridge.json",
        capture_root,
        source_packet,
    )
    runtime_root = OUTPUT / "preservation-runtime"
    materialize_preservation_runtime_packet(
        source_packet / "materialization-plan.json", runtime_root
    )

    repeat_plan = {
        "format": "truth_editing_preservation_repeat_plan_v1",
        "contract_sha256": _sha(contract),
        "capture_identity_sha256": identity["self_sha256"],
        "tiers": contract["repeats"]["tiers"],
        "repeat_indices": contract["repeats"]["repeat_indices"],
    }
    repeat_plan_sha = _sha(repeat_plan)
    repeat_plan["self_sha256"] = repeat_plan_sha
    _write_new(OUTPUT / "repeat-plan.json", repeat_plan)
    receipts: list[Path] = []
    for tier in contract["repeats"]["tiers"]:
        collector = TrialPreservationCollector.from_config(
            runtime_root / f"truth_editing_preservation_runtime_{tier}_v1.json",
            backend=inference_backend,
        )
        for repeat_index in contract["repeats"]["repeat_indices"]:
            receipt = collector.collect_base_repeat(
                bundle,
                repeat_plan_sha256=repeat_plan_sha,
                repeat_index=repeat_index,
            )
            receipt_path = (
                OUTPUT
                / "preservation-thresholds"
                / "base-repeats"
                / tier
                / f"repeat-{repeat_index}.json"
            )
            _write_new(receipt_path, receipt)
            receipts.append(receipt_path)

    calibration = OUTPUT / "preservation-thresholds" / "calibration.json"
    calibration.parent.mkdir(parents=True, exist_ok=True)
    build_preservation_threshold_calibration(
        calibration,
        calibration_id="qwen3-vl-8b-thinking-preservation-v2-r4",
        base_model_sha256=identity["base_model_sha256"],
        receipt_paths=receipts,
        minimum_repeats=5,
        quantile=0.95,
        absolute_margin=0.0001,
        relative_margin=0.25,
    )
    terminal = {
        "format": "truth_editing_preservation_capture_terminal_v1",
        "contract_sha256": _sha(contract),
        "capture_identity_sha256": identity["self_sha256"],
        "repeat_plan_sha256": repeat_plan_sha,
        "repeat_receipt_count": len(receipts),
        "calibration_sha256": json.loads(calibration.read_text())["self_sha256"],
    }
    terminal["self_sha256"] = _sha(terminal)
    _write_new(OUTPUT / "preservation-capture-terminal.json", terminal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
