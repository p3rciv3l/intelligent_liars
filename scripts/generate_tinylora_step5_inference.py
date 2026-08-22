#!/usr/bin/env python3
"""Generate complete, deterministic, unscored TinyLoRA Step 5 inventories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from intelligent_liars.step5_inference_inventory import (
    DecodingContract,
    InferenceContractError,
    QwenInferenceBackend,
    build_inference_requests,
    file_sha256,
    generate_inference_inventories,
    install_candidate_adapter,
    publish_inference_run,
    seed_inference,
    verified_model_identity,
)
from intelligent_liars.step5_multimodal_assets import (
    rebase_image_references,
    validate_staged_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-snapshot-root", type=Path, required=True)
    parser.add_argument("--model-snapshot-plan", type=Path, required=True)
    parser.add_argument("--image-bundle-root", type=Path, required=True)
    parser.add_argument(
        "--runtime-image-receipt",
        type=Path,
        required=True,
        help="verified JSON receipt for the exact prebuilt runtime image",
    )
    parser.add_argument("--model-state", choices=("base", "candidate"), required=True)
    parser.add_argument("--adapter-state", type=Path)
    parser.add_argument("--adapter-metadata", type=Path)
    parser.add_argument("--basis-state", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=20260822)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _software_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/intelligent_liars/step5_inference_inventory.py",
        PROJECT_ROOT / "src/intelligent_liars/model_cache.py",
        PROJECT_ROOT / "src/intelligent_liars/models.py",
        PROJECT_ROOT / "src/intelligent_liars/step5_multimodal_assets.py",
        PROJECT_ROOT / "src/intelligent_liars/standalone_models.py",
        PROJECT_ROOT / "src/intelligent_liars/tinylora_step5.py",
    ):
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(file_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_state_arguments(args: argparse.Namespace) -> None:
    candidate_values = (args.adapter_state, args.adapter_metadata)
    if args.model_state == "base" and any(value is not None for value in candidate_values):
        raise InferenceContractError("base inference must not receive adapter files")
    if args.model_state == "base" and args.basis_state is not None:
        raise InferenceContractError("base inference must not receive a TinyLoRA basis")
    if args.model_state == "candidate" and any(value is None for value in candidate_values):
        raise InferenceContractError(
            "candidate inference requires --adapter-state and --adapter-metadata"
        )


def _verified_vision_rows(
    *, plan_path: Path, image_bundle_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = json.loads(plan_path.read_text())
    specification = plan["outputs"]["preservation_development_vision"]
    original = plan_path.parent / specification["path"]
    rows = _read_jsonl(original)
    manifest = validate_staged_bundle(image_bundle_root)
    rebased = rebase_image_references(
        rows,
        bundle_root=image_bundle_root,
        manifest=manifest,
    )
    return rebased, {
        "format": manifest["format"],
        "content_sha256": manifest["content_sha256"],
        "manifest_file_sha256": file_sha256(image_bundle_root / "manifest.json"),
        "unique_images": manifest["totals"]["unique_images"],
        "bytes": manifest["totals"]["bytes"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_state_arguments(args)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(args.output_dir)
    plan_path = args.plan.resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    model_specification = plan["model"]
    model_cache_identity, _verified = verified_model_identity(
        snapshot_root=args.model_snapshot_root,
        snapshot_plan_path=args.model_snapshot_plan,
        frozen_model=model_specification,
    )
    vision_rows, image_receipt = _verified_vision_rows(
        plan_path=plan_path,
        image_bundle_root=args.image_bundle_root,
    )
    requests, source_receipt = build_inference_requests(
        plan_path,
        verified_vision_rows=vision_rows,
        image_asset_receipt=image_receipt,
    )
    source_receipt["model_cache"] = model_cache_identity
    try:
        runtime_receipt = json.loads(args.runtime_image_receipt.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise InferenceContractError("invalid runtime image receipt") from error
    if not isinstance(runtime_receipt, dict) or not runtime_receipt:
        raise InferenceContractError("runtime image receipt must be a nonempty object")
    source_receipt["runtime_image"] = {
        "receipt_sha256": file_sha256(args.runtime_image_receipt),
        "receipt": runtime_receipt,
    }

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise InferenceContractError("Step 5 production inference requires CUDA")
    seed_inference(args.seed)
    processor = AutoProcessor.from_pretrained(
        str(args.model_snapshot_root),
        local_files_only=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model_snapshot_root),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attention,
        device_map={"": args.device},
    )
    model.eval()
    if args.model_state == "base":
        model_identity = {
            "state": "base",
            **model_cache_identity,
        }
    else:
        model_identity = {
            **model_cache_identity,
            **install_candidate_adapter(
                model=model,
                processor=processor,
                model_id=str(model_specification["model_id"]),
                revision=str(model_specification["revision"]),
                plan=plan,
                plan_sha256=file_sha256(plan_path),
                adapter_state_path=args.adapter_state,
                adapter_metadata_path=args.adapter_metadata,
                basis_path=args.basis_state,
            ),
        }
    decoding = DecodingContract(seed=args.seed)
    payload = generate_inference_inventories(
        requests,
        backend=QwenInferenceBackend(model=model, processor=processor),
        decoding=decoding,
        source_receipt=source_receipt,
        model_identity=model_identity,
        software_sha256=_software_sha256(),
    )
    manifest = publish_inference_run(args.output_dir, payload)
    print(
        json.dumps(
            {
                "complete": manifest["complete"],
                "output_dir": str(args.output_dir),
                "records": manifest["completed_records"],
                "run_identity_sha256": manifest["run_identity_sha256"],
                "content_sha256": manifest["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
