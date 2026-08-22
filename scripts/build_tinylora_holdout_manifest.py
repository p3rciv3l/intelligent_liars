#!/usr/bin/env python3
"""Freeze external holdouts as hashes and run retrospective decontamination."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from intelligent_liars.holdout_decontamination import (
    TextFingerprint,
    find_collisions,
    fingerprint_compiled_corpus,
    fingerprint_record,
    summarize_collisions,
    write_fingerprints,
)

HF_DATASETS = {
    "ai2_arc": {
        "dataset": "allenai/ai2_arc",
        "splits": [
            ("ARC-Challenge", "validation"),
            ("ARC-Challenge", "test"),
            ("ARC-Easy", "validation"),
            ("ARC-Easy", "test"),
        ],
    },
    "pku_deceptionbench": {
        "dataset": "PKU-Alignment/DeceptionBench",
        "splits": [("default", "test")],
    },
}

GIT_SOURCES = {
    "metr_public_tasks": {
        "url": "https://github.com/METR/public-tasks.git",
        "directory": "metr-public-tasks",
    },
    "olmes": {
        "url": "https://github.com/allenai/olmes.git",
        "directory": "olmes",
    },
    "pku_mm_deceptionbench": {
        "url": "https://github.com/PKU-Alignment/MM-DeceptionBench.git",
        "directory": "mm-deceptionbench",
    },
    "osworld": {
        "url": "https://github.com/xlang-ai/OSWorld.git",
        "directory": "osworld",
    },
}


def _json_get(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return value


def _hf_rows(dataset: str, config: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": 100,
            }
        )
        payload = _json_get(f"https://datasets-server.huggingface.co/rows?{query}")
        page = payload.get("rows")
        if not isinstance(page, list):
            raise ValueError(f"Dataset Viewer returned no rows for {dataset}/{config}/{split}")
        rows.extend(item["row"] for item in page if isinstance(item, dict))
        offset += len(page)
        total = int(payload.get("num_rows_total", offset))
        if not page or offset >= total:
            return rows


def _git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _fingerprint_rows(
    rows: list[Any], *, source_id: str, id_prefix: str
) -> list[TextFingerprint]:
    fingerprints: list[TextFingerprint] = []
    for index, row in enumerate(rows):
        record_id = (
            str(row.get("id", f"{id_prefix}:{index:06d}"))
            if isinstance(row, dict)
            else f"{id_prefix}:{index:06d}"
        )
        fingerprints.extend(
            fingerprint_record(row, source_id=source_id, record_id=record_id)
        )
    return fingerprints


def _metr_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/manifest.yaml")):
        payload = yaml.safe_load(path.read_text())
        tasks = payload.get("tasks", {}) if isinstance(payload, dict) else {}
        for task_id, task in tasks.items():
            records.append(
                {
                    "id": f"{path.parent.name}:{task_id}",
                    "suite": payload.get("meta", {}).get("name"),
                    "task": task,
                }
            )
    return records


def _mm_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((root / "dataset").glob("*.json")):
        payload = json.loads(path.read_text())
        for index, row in enumerate(payload):
            records.append({"id": f"{path.stem}:{index:04d}", **row})
    return records


def _osworld_records(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text())
        for path in sorted((root / "evaluation_examples" / "examples").glob("*/*.json"))
    ]


def _olmes_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    task_root = root / "oe_eval" / "tasks" / "oe_eval_tasks"
    for path in sorted(task_root.glob("*.py")):
        if path.name == "__init__.py":
            continue
        records.append({"id": path.stem, "task_definition": path.read_text()})
    return records


def _apollo_audit_records(project_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths = sorted(
        project_root.glob(
            "references/deception-detection/data/rollouts/ai_audit__*llama*.json"
        )
    )
    for path in paths:
        payload = json.loads(path.read_text())
        for index, row in enumerate(payload.get("rollouts", [])):
            records.append(
                {
                    "id": f"{path.stem}:{index:05d}",
                    "input_messages": row.get("input_messages"),
                    "metadata": row.get("metadata"),
                }
            )
    return records


def _truthspec_audit_records(project_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    names = ["cities_qa.csv", "companies_true_false_qa.csv"]
    for name in names:
        path = project_root / "references" / "truth_spec" / "data" / "geometry_of_truth" / name
        with path.open(newline="") as source:
            for index, row in enumerate(csv.DictReader(source)):
                records.append({"id": f"{path.stem}:{index:05d}", **row})
    return records


def build_manifest(
    *, project_root: Path, scratch_root: Path, corpus_root: Path, output_root: Path
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    fingerprints: list[TextFingerprint] = []
    sources: list[dict[str, Any]] = []

    for source_id, spec in HF_DATASETS.items():
        dataset = str(spec["dataset"])
        metadata = _json_get(f"https://huggingface.co/api/datasets/{dataset}")
        revision = str(metadata["sha"])
        row_count = 0
        for config, split in spec["splits"]:
            rows = _hf_rows(dataset, config, split)
            row_count += len(rows)
            fingerprints.extend(
                _fingerprint_rows(
                    rows,
                    source_id=source_id,
                    id_prefix=f"{config}:{split}",
                )
            )
        sources.append(
            {
                "source_id": source_id,
                "kind": "huggingface_dataset_viewer",
                "url": f"https://huggingface.co/datasets/{dataset}",
                "revision": revision,
                "selectors": spec["splits"],
                "record_count": row_count,
            }
        )

    git_extractors = {
        "metr_public_tasks": _metr_records,
        "olmes": _olmes_records,
        "pku_mm_deceptionbench": _mm_records,
        "osworld": _osworld_records,
    }
    for source_id, spec in GIT_SOURCES.items():
        root = scratch_root / str(spec["directory"])
        records = git_extractors[source_id](root)
        fingerprints.extend(
            _fingerprint_rows(records, source_id=source_id, id_prefix=source_id)
        )
        sources.append(
            {
                "source_id": source_id,
                "kind": "git_checkout",
                "url": spec["url"],
                "revision": _git_revision(root),
                "record_count": len(records),
            }
        )

    local_sources = {
        "apollo_ai_audit": _apollo_audit_records(project_root),
        "truthspec_disjoint_qa_audit": _truthspec_audit_records(project_root),
    }
    for source_id, records in local_sources.items():
        fingerprints.extend(
            _fingerprint_rows(records, source_id=source_id, id_prefix=source_id)
        )
        sources.append(
            {
                "source_id": source_id,
                "kind": "project_local_holdout",
                "revision": subprocess.check_output(
                    ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
                ).strip(),
                "record_count": len(records),
            }
        )

    fingerprint_path = output_root / "holdout_fingerprints.jsonl"
    write_fingerprints(fingerprint_path, fingerprints)
    manifest = {
        "format": "tinylora_holdout_manifest_v1",
        "privacy_boundary": "Stores normalized content hashes and SimHash fingerprints, not benchmark text.",
        "fingerprint_algorithm": {
            "normalization": "NFKC casefold alphanumeric tokens",
            "features": "unigrams plus adjacent bigrams",
            "exact": "sha256",
            "near": "64-bit SimHash; eight 8-bit candidate bands; Hamming <= 6; token ratio >= 0.75",
        },
        "sources": sorted(sources, key=lambda item: item["source_id"]),
        "fingerprint_count": len(fingerprints),
        "fingerprints_sha256": hashlib.sha256(fingerprint_path.read_bytes()).hexdigest(),
    }
    (output_root / "holdout_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    corpus_fingerprints = fingerprint_compiled_corpus(corpus_root)
    collisions = find_collisions(corpus_fingerprints, fingerprints)
    report = summarize_collisions(
        collisions,
        corpus_fingerprint_count=len(corpus_fingerprints),
        holdout_fingerprint_count=len(fingerprints),
    )
    report["corpus_root"] = str(corpus_root)
    report["holdout_manifest_sha256"] = hashlib.sha256(
        (output_root / "holdout_manifest.json").read_bytes()
    ).hexdigest()
    (output_root / "decontamination_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    exclusion_rows = [
        {
            "source": item["source"],
            "record_id": item["record_id"],
            "reason": "holdout_exact_or_near_collision",
        }
        for item in report["contaminated_records"]
    ]
    exclusion_manifest = {
        "format": "tinylora_training_exclusions_v1",
        "policy": "Every record implicated by the frozen exact/near holdout scan is excluded conservatively.",
        "decontamination_report_sha256": hashlib.sha256(
            (output_root / "decontamination_report.json").read_bytes()
        ).hexdigest(),
        "excluded_record_count": len(exclusion_rows),
        "excluded_records": exclusion_rows,
    }
    exclusion_path = output_root / "training_exclusions.json"
    exclusion_path.write_text(
        json.dumps(exclusion_manifest, indent=2, sort_keys=True) + "\n"
    )
    report["training_exclusions_sha256"] = hashlib.sha256(
        exclusion_path.read_bytes()
    ).hexdigest()
    report["valid_after_exclusions"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = build_manifest(
        project_root=args.project_root.resolve(),
        scratch_root=args.scratch_root.resolve(),
        corpus_root=args.corpus_root.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "valid",
                    "valid_after_exclusions",
                    "collision_count",
                    "contaminated_record_count",
                    "corpus_fingerprint_count",
                    "holdout_fingerprint_count",
                    "holdout_manifest_sha256",
                    "training_exclusions_sha256",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
