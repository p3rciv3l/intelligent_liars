#!/usr/bin/env python3
"""Build the immutable local data/evaluation contract for TinyLoRA Step 5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from intelligent_liars.tinylora_pilot import file_sha256, stable_score
from intelligent_liars.tinylora_step5 import (
    audit_seal_evidence,
    enrich_behavior_alternatives,
    qualify_text_preservation_rows,
    source_training_admission,
    split_iid_development,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _ordered(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            stable_score(seed, str(row["record_id"])),
            str(row["record_id"]),
        ),
    )


def _text_preservation_row(
    row: dict[str, Any],
    *,
    split: str,
    source_path: str,
    category: str,
) -> dict[str, Any]:
    messages = row["payload"]["messages"]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"Text preservation row lacks final assistant turn: {row['record_id']}")
    output = {
        "format": "tinylora_step5_example_v1",
        "record_id": row["record_id"],
        "split_group_id": row["split_group_id"],
        "split": split,
        "kind": "preservation",
        "objective": "preservation_kl",
        "preservation_category": category,
        "messages": messages,
        "source": source_path,
    }
    qualification = row.get("qualification")
    if qualification is not None:
        output["qualification"] = qualification
    return output


def _vision_preservation_row(
    row: dict[str, Any],
    *,
    split: str,
    source_path: str,
    repository_root: Path,
    seed: int,
) -> dict[str, Any]:
    payload = row["payload"]
    questions = payload["questions"]["question"]
    answers = payload["questions"]["answer"]
    if not questions or len(questions) != len(answers):
        raise ValueError(f"Invalid PixMo question inventory: {row['record_id']}")
    index = int(stable_score(seed, str(row["record_id"]))[:8], 16) % len(questions)
    image = payload["image_snapshot"]
    relative_image = Path(str(image["local_path"]))
    image_path = repository_root / relative_image
    if not image_path.is_file() or file_sha256(image_path) != image["sha256"]:
        raise ValueError(f"Missing or mismatched PixMo image: {relative_image}")
    return {
        "format": "tinylora_step5_example_v1",
        "record_id": row["record_id"],
        "split_group_id": row["split_group_id"],
        "split": split,
        "kind": "preservation",
        "objective": "preservation_kl",
        "preservation_category": f"vision_{payload['config']}",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(relative_image)},
                    {"type": "text", "text": str(questions[index])},
                ],
            },
            {"role": "assistant", "content": str(answers[index])},
        ],
        "image_sha256": image["sha256"],
        "source": source_path,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "scenarios": len({row.get("scenario_id") for row in rows if row.get("scenario_id")}),
        "families": dict(sorted(Counter(row.get("family", "preservation") for row in rows).items())),
        "objectives": dict(sorted(Counter(row["objective"] for row in rows).items())),
        "preservation_categories": dict(
            sorted(
                Counter(
                    row["preservation_category"]
                    for row in rows
                    if row["kind"] == "preservation"
                ).items()
            )
        ),
    }


def _xstest_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    output = []
    for row in rows:
        if row["label"] not in {"safe", "unsafe"}:
            raise ValueError(f"Unsupported XSTest label: {row['label']}")
        output.append(
            {
                "format": "tinylora_step5_safety_eval_v1",
                "record_id": f"xstest.{int(row['id']):03d}",
                "split_group_id": f"xstest.{row['type']}",
                "split": "development_safety_refusal",
                "kind": "safety_evaluation",
                "objective": "safety_refusal_calibration",
                "prompt": row["prompt"],
                "expected_behavior": "comply" if row["label"] == "safe" else "refuse",
                "xstest_type": row["type"],
                "focus": row["focus"],
                "source": str(path),
            }
        )
    return output


def _source_entry_for_path(
    registry: dict[str, Any], repository_relative_path: str
) -> dict[str, Any]:
    matches = [
        source
        for source in registry.get("sources", [])
        if repository_relative_path in source.get("paths", [])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one registry entry for {repository_relative_path}"
        )
    return matches[0]


def _prime_category(row: dict[str, Any]) -> str:
    general_text_tasks = {
        "complex_json_output",
        "ifeval",
        "no_verification",
        "pydantic_adherance",
    }
    return (
        "general_text"
        if row["payload"].get("task_type") in general_text_tasks
        else "reasoning"
    )


def build_plan(
    *,
    repository_root: Path,
    pilot_v1_root: Path,
    preservation_root: Path,
    output_root: Path,
    seed: int,
    tokenizer: Any,
    tokenizer_sha256: str,
    tokenizer_config_sha256: str,
    max_length: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    pilot_manifest_path = pilot_v1_root / "manifest.json"
    pilot_manifest = json.loads(pilot_manifest_path.read_text())
    if pilot_manifest.get("format") != "tinylora_bounded_pilot_plan_v1":
        raise ValueError("Unsupported pilot-v1 manifest")

    train_v1 = read_jsonl(pilot_v1_root / pilot_manifest["outputs"]["train"]["path"])
    heldout_v1 = read_jsonl(
        pilot_v1_root / pilot_manifest["outputs"]["development"]["path"]
    )
    train_behavior = enrich_behavior_alternatives(
        row for row in train_v1 if row["kind"] == "behavior"
    )
    heldout_family = enrich_behavior_alternatives(
        row for row in heldout_v1 if row["kind"] == "behavior"
    )
    train_behavior, iid_development = split_iid_development(
        train_behavior,
        seed=seed,
        fraction=0.1,
    )
    heldout_family = [
        {**row, "split": "development_heldout_family"} for row in heldout_family
    ]

    source_registry_path = pilot_v1_root.parent / "source_registry.json"
    source_registry = json.loads(source_registry_path.read_text())
    prime_path = preservation_root / "prime_synthetic_2_verified_preservation.jsonl"
    tulu_path = preservation_root / "tulu_3_sft_preservation.jsonl"
    pixmo_path = preservation_root / "pixmo_docs_preservation.jsonl"
    xstest_path = preservation_root / "xstest_prompts.csv"
    registry_paths = {
        "prime": str(prime_path.relative_to(repository_root)),
        "tulu": str(tulu_path.relative_to(repository_root)),
        "pixmo_docs": str(pixmo_path.relative_to(repository_root)),
    }
    source_entries = {
        name: _source_entry_for_path(source_registry, path)
        for name, path in registry_paths.items()
    }
    source_admission = {
        name: {
            "source_id": source_entries[name]["source_id"],
            "admitted_to_training": source_training_admission(source_entries[name])[0],
            "reason": source_training_admission(source_entries[name])[1],
        }
        for name in sorted(source_entries)
    }
    if not source_admission["prime"]["admitted_to_training"]:
        raise ValueError("Prime preservation source is not admitted by the registry")
    if not source_admission["pixmo_docs"]["admitted_to_training"]:
        raise ValueError("PixMo-Docs preservation source is not admitted by the registry")
    if source_admission["tulu"]["admitted_to_training"]:
        raise ValueError("Tulu source was expected to remain quarantined pending review")

    prime, text_exclusions = qualify_text_preservation_rows(
        read_jsonl(prime_path),
        tokenizer=tokenizer,
        max_length=max_length,
        semantic_exclusions={
            "prime_synthetic_2_verified.default.train.000079539": (
                "semantic_quality_changes_problem_to_force_answer"
            )
        },
    )
    prime = _ordered(prime, seed=seed)
    pixmo = _ordered(read_jsonl(pixmo_path), seed=seed + 2)
    safety_refusal_development = _xstest_rows(xstest_path)
    prime_by_category = {
        category: [row for row in prime if _prime_category(row) == category]
        for category in ("reasoning", "general_text")
    }
    if any(len(rows) < 4 for rows in prime_by_category.values()) or len(pixmo) < 200:
        raise ValueError("Preservation snapshots are smaller than the frozen Step 5 plan")

    prime_splits: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for category, rows in prime_by_category.items():
        development_count = max(1, len(rows) // 5)
        prime_splits[category] = (
            rows[:-development_count],
            rows[-development_count:],
        )

    source_prefix = str(preservation_root.relative_to(repository_root))
    preservation_train = [
        *[
            _text_preservation_row(
                row,
                split="train",
                source_path=f"{source_prefix}/{prime_path.name}",
                category="reasoning",
            )
            for row in prime_splits["reasoning"][0]
        ],
        *[
            _text_preservation_row(
                row,
                split="train",
                source_path=f"{source_prefix}/{prime_path.name}",
                category="general_text",
            )
            for row in prime_splits["general_text"][0]
        ],
        *[
            _vision_preservation_row(
                row,
                split="train",
                source_path=f"{source_prefix}/{pixmo_path.name}",
                repository_root=repository_root,
                seed=seed,
            )
            for row in pixmo[:100]
        ],
    ]
    preservation_development_text = [
        *[
            _text_preservation_row(
                row,
                split="development_preservation_text",
                source_path=f"{source_prefix}/{prime_path.name}",
                category="reasoning",
            )
            for row in prime_splits["reasoning"][1]
        ],
        *[
            _text_preservation_row(
                row,
                split="development_preservation_text",
                source_path=f"{source_prefix}/{prime_path.name}",
                category="general_text",
            )
            for row in prime_splits["general_text"][1]
        ],
    ]
    preservation_development_vision = [
        _vision_preservation_row(
            row,
            split="development_preservation_vision",
            source_path=f"{source_prefix}/{pixmo_path.name}",
            repository_root=repository_root,
            seed=seed,
        )
        for row in pixmo[100:200]
    ]

    outputs = {
        "train_behavior": train_behavior,
        "development_iid": iid_development,
        "development_heldout_family": heldout_family,
        "preservation_train": preservation_train,
        "preservation_development_text": preservation_development_text,
        "preservation_development_vision": preservation_development_vision,
        "safety_refusal_development": safety_refusal_development,
    }
    output_manifest: dict[str, Any] = {}
    for name, rows in outputs.items():
        path = output_root / f"{name}.jsonl"
        sorted_rows = sorted(rows, key=lambda row: str(row["record_id"]))
        write_jsonl(path, sorted_rows)
        output_manifest[name] = {
            "path": path.name,
            "sha256": file_sha256(path),
            **_summarize(sorted_rows),
        }

    audit_path = pilot_v1_root / pilot_manifest["outputs"]["audit"]["path"]
    seal_evidence = audit_seal_evidence(
        audit_path,
        expected_sha256=pilot_manifest["outputs"]["audit"]["sha256"],
    )
    if not seal_evidence["hash_matches"]:
        raise ValueError("Sealed audit hash does not match the pilot manifest")

    manifest = {
        "format": "tinylora_step5_plan_v1",
        "large_run_enabled": False,
        "paid_execution_enabled": False,
        "seed": seed,
        "model": {
            "model_id": "Qwen/Qwen3-VL-8B-Thinking",
            "revision": "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b",
            "attention": "flash_attention_2",
            "vision_weights_frozen": True,
        },
        "inputs": {
            "pilot_v1_manifest_sha256": file_sha256(pilot_manifest_path),
            "prime_snapshot_sha256": file_sha256(prime_path),
            "tulu_snapshot_sha256": file_sha256(tulu_path),
            "pixmo_docs_snapshot_sha256": file_sha256(pixmo_path),
            "xstest_prompts_sha256": file_sha256(xstest_path),
            "xstest_revision": "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d",
            "source_registry_sha256": file_sha256(source_registry_path),
            "tokenizer_json_sha256": tokenizer_sha256,
            "tokenizer_config_sha256": tokenizer_config_sha256,
        },
        "sealed_audit": {
            "opened": False,
            "records": pilot_manifest["outputs"]["audit"]["records"],
            "sha256": pilot_manifest["outputs"]["audit"]["sha256"],
            "packaged": False,
            "evidence": seal_evidence,
        },
        "split_policy": (
            "Keep six variants and every scenario-series/template sibling together; "
            "preserve the pilot-v1 family holdout; hold out a deterministic 10% "
            "of series inside training families for IID development."
        ),
        "source_admission": source_admission,
        "preservation_curation": {
            "max_length": max_length,
            "tokenizer_model_id": "Qwen/Qwen3-VL-8B-Thinking",
            "tokenizer_revision": "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b",
            "qualified_prime_records": len(prime),
            "excluded_prime_records": len(text_exclusions),
            "exclusion_manifest_sha256": hashlib.sha256(
                (json.dumps(text_exclusions, sort_keys=True) + "\n").encode()
            ).hexdigest(),
            "exclusions": text_exclusions,
            "semantic_quality_policy": (
                "Source-level verification is necessary but not sufficient; known "
                "semantic defects are excluded before length qualification and split."
            ),
        },
        "source_notices": {
            "xstest": {
                "name": "XSTest",
                "upstream_url": "https://github.com/paul-rottger/xstest",
                "revision": "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d",
                "license": "CC-BY-4.0",
                "attribution": "Paul Rottger et al., XSTest",
            },
            "pixmo_docs": {
                "name": "Ai2 PixMo-Docs",
                "upstream_url": "https://huggingface.co/datasets/allenai/pixmo-docs",
                "license": "ODC-BY-1.0 with separate model-output terms",
                "responsible_use_notice": (
                    "Retain Ai2 responsible-use and attribution terms; local image "
                    "availability does not expand redistribution rights."
                ),
            },
        },
        "preservation_policy": {
            "batch_fraction": 0.25,
            "train_and_development_disjoint": True,
            "real_local_images": True,
            "required_categories": [
                "general_text",
                "reasoning",
                "safety_refusal",
                "vision_charts",
                "vision_diagrams",
                "vision_tables",
                "vision_other",
            ],
            "status": (
                "complete for the bounded screen; Tulu rows are quarantined and "
                "excluded pending subset-terms review"
            ),
        },
        "evaluation_batteries": {
            "structured_actions": (
                "paired true-state and counterfactual action rows in both IID and "
                "held-out-family development"
            ),
            "safety_refusal": (
                "XSTest 250 safe plus 200 unsafe contrasts, pinned at the recorded revision"
            ),
        },
        "evaluation": {
            "primary": "paired completion log-probability margins by scenario and family",
            "generation": "thinking disabled, deterministic, 128-token budget, parsed reports/actions",
            "uncertainty": "paired bootstrap by scenario and family",
            "probe_role": "bounded training regularizer and disjoint evaluation diagnostic only",
            "thresholds": "freeze after repeated unchanged-base evaluation",
        },
        "arms": [
            {
                "name": "tinylora_dim13",
                "adapter": "tinylora",
                "svd_rank": 3,
                "projection_dim": 13,
                "train_layers": [21],
            },
            {
                "name": "tinylora_dim63",
                "adapter": "tinylora",
                "svd_rank": 3,
                "projection_dim": 63,
                "train_layers": [21],
            },
            {
                "name": "lora_rank3_ceiling",
                "adapter": "ordinary_lora",
                "lora_rank": 3,
                "train_layers": [21],
            },
        ],
        "basis_contract": {
            "effective_coordinate_counts": [13, 63],
            "normalization": "unit_norm_columns_before_coordinate selection",
            "nesting": "the 13-coordinate basis is the exact prefix of the 63-coordinate basis",
            "capacity_ceiling": (
                "ordinary rank-3 LoRA is the representational ceiling because the "
                "TinyLoRA update has matrix rank at most 3"
            ),
        },
        "schedule": {
            "identical_row_order_across_arms": True,
            "zero_unexplained_skips": True,
            "tiny_overfit_required": True,
            "max_concurrent_single_gpu_workers": 3,
        },
        "outputs": output_manifest,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {**manifest, "manifest_sha256": file_sha256(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--pilot-v1-root", type=Path, required=True)
    parser.add_argument("--preservation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    model_id = "Qwen/Qwen3-VL-8B-Thinking"
    revision = "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b"
    tokenizer_path = Path(
        hf_hub_download(
            repo_id=model_id,
            filename="tokenizer.json",
            revision=revision,
        )
    )
    tokenizer_config_path = Path(
        hf_hub_download(
            repo_id=model_id,
            filename="tokenizer_config.json",
            revision=revision,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    manifest = build_plan(
        repository_root=args.repository_root.resolve(),
        pilot_v1_root=args.pilot_v1_root.resolve(),
        preservation_root=args.preservation_root.resolve(),
        output_root=args.output_root.resolve(),
        seed=args.seed,
        tokenizer=tokenizer,
        tokenizer_sha256=file_sha256(tokenizer_path),
        tokenizer_config_sha256=file_sha256(tokenizer_config_path),
        max_length=args.max_length,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
