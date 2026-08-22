#!/usr/bin/env python3
"""Build immutable family-group splits and bounded TinyLoRA pilot mixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from intelligent_liars.tinylora_pilot import (
    BEHAVIOR_OBJECTIVES,
    PILOT_SPLIT_COUNTS,
    assign_group_splits,
    file_sha256,
    load_exclusions,
    stable_score,
    summarize_rows,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def _preservation_rows(
    corpus_root: Path,
    *,
    exclusions: set[tuple[str, str]],
    seed: int,
    limit: int,
) -> list[dict[str, Any]]:
    source = corpus_root / "preservation" / "prime_synthetic_2_verified_snapshot.jsonl"
    source_key = str(source.relative_to(corpus_root))
    eligible = [
        row
        for row in _read_jsonl(source)
        if (source_key, str(row["record_id"])) not in exclusions
        and sum(
            len(str(message.get("content", "")))
            for message in row["payload"].get("messages", [])
        )
        <= 6_000
    ]
    eligible.sort(key=lambda row: stable_score(seed, str(row["record_id"])))
    selected = eligible[:limit]
    output: list[dict[str, Any]] = []
    for row in selected:
        output.append(
            {
                "format": "tinylora_pilot_example_v1",
                "record_id": row["record_id"],
                "split_group_id": row["split_group_id"],
                "split": "train",
                "kind": "preservation",
                "objective": "preservation_kl",
                "messages": row["payload"]["messages"],
                "source": source_key,
            }
        )
    return output


def build_plan(
    *, corpus_root: Path, holdout_root: Path, output_root: Path, seed: int
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    exclusions_path = holdout_root / "training_exclusions.json"
    exclusions = load_exclusions(exclusions_path)
    rendered_path = corpus_root / "synthetic" / "rendered_training_examples.jsonl"
    rendered = _read_jsonl(rendered_path)
    groups = {str(row["split_group_id"]) for row in rendered}
    assignments = assign_group_splits(groups, seed=seed)
    behavior_rows: list[dict[str, Any]] = []
    for row in rendered:
        if row["objective"] not in BEHAVIOR_OBJECTIVES:
            raise ValueError(f"Unsupported rendered objective: {row['objective']}")
        behavior_rows.append(
            {
                **row,
                "format": "tinylora_pilot_example_v1",
                "kind": "behavior",
                "split": assignments[str(row["split_group_id"])],
                "source": str(rendered_path.relative_to(corpus_root)),
            }
        )
    preservation = _preservation_rows(
        corpus_root,
        exclusions=exclusions,
        seed=seed,
        limit=256,
    )
    all_rows = behavior_rows + preservation
    outputs: dict[str, Path] = {}
    for split in ("train", "development", "audit"):
        rows = [row for row in all_rows if row["split"] == split]
        rows.sort(key=lambda row: str(row["record_id"]))
        path = output_root / f"{split}.jsonl"
        _write_jsonl(path, rows)
        outputs[split] = path
    group_rows = [
        {"split_group_id": group, "split": assignments[group]}
        for group in sorted(assignments)
    ]
    group_path = output_root / "group_assignments.jsonl"
    _write_jsonl(group_path, group_rows)
    manifest = {
        "format": "tinylora_bounded_pilot_plan_v1",
        "large_run_enabled": False,
        "seed": seed,
        "split_policy": "Whole family-level split_group_id; 37 train, 8 development, 8 audit.",
        "preservation_policy": (
            f"Up to 256 deterministic Prime SYNTHETIC-2 prompts in train for "
            f"frozen-base KL; {len(preservation)} passed the conservative 6,000-character bound."
        ),
        "direction_policy": "Only conditioned_false_direct_report receives positive layer-21 margin loss; behavior CE covers all six paired objectives.",
        "inputs": {
            "corpus_manifest_sha256": file_sha256(corpus_root / "manifest.json"),
            "rendered_examples_sha256": file_sha256(rendered_path),
            "holdout_manifest_sha256": file_sha256(holdout_root / "holdout_manifest.json"),
            "training_exclusions_sha256": file_sha256(exclusions_path),
        },
        "split_group_counts": PILOT_SPLIT_COUNTS,
        "outputs": {
            split: {
                "path": path.name,
                "sha256": file_sha256(path),
                **summarize_rows(_read_jsonl(path)),
            }
            for split, path in outputs.items()
        },
        "group_assignments": {
            "path": group_path.name,
            "sha256": file_sha256(group_path),
        },
        "pilot": {
            "ranks": [1, 2, 3],
            "projection_dim": 13,
            "train_layers": [21],
            "max_optimizer_steps": 200,
            "max_sequence_length": 2048,
            "attention": "flash_attention_2",
            "gradient_checkpointing": True,
            "objective_weights": {
                "behavior_ce": 1.0,
                "directional_margin": 0.25,
                "preservation_kl": 0.5,
            },
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--holdout-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    manifest = build_plan(
        corpus_root=args.corpus_root.resolve(),
        holdout_root=args.holdout_root.resolve(),
        output_root=args.output_root.resolve(),
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
