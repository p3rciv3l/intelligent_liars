#!/usr/bin/env python3
"""Fetch deterministic, auditable preservation snapshots from Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

from intelligent_liars.tinylora_corpus import deterministic_page_offsets


DATASET_SERVER = "https://datasets-server.huggingface.co"
PAGE_LENGTH = 100

TULU_SOURCE_QUOTAS = {
    "ai2-adapt-dev/oasst1_converted": 100,
    "ai2-adapt-dev/no_robots_converted": 100,
    "ai2-adapt-dev/tulu_v3.9_sciriff_10k": 100,
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated": 100,
    "ai2-adapt-dev/flan_v2_converted": 100,
    "ai2-adapt-dev/tulu_v3.9_table_gpt_5k": 100,
    "ai2-adapt-dev/tulu_v3.9_aya_100k": 100,
    "ai2-adapt-dev/numinamath_tir_math_decontaminated": 100,
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k": 100,
}

PRIME_TASK_QUOTAS = {
    "ascii_tree_formatting": 50,
    "complex_json_output": 100,
    "pydantic_adherance": 100,
    "prime_rl_code": 100,
    "verifiable_math": 100,
    "unscramble_sentence": 100,
    "code_output_prediction": 100,
    "ifeval": 100,
    "no_verification": 100,
    "reasoning_gym": 100,
}


def _get_json(
    endpoint: str,
    params: dict[str, Any],
    *,
    cache_path: Path | None = None,
    pause_after_success: float = 0.0,
) -> dict[str, Any]:
    if cache_path is not None and cache_path.is_file():
        payload = json.loads(cache_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Cached payload is not an object: {cache_path}")
        return payload
    retry_delays = [2.0, 5.0, 10.0, 20.0, 40.0, 60.0]
    for attempt in range(len(retry_delays) + 1):
        response = requests.get(
            f"{DATASET_SERVER}/{endpoint}", params=params, timeout=120
        )
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Dataset server returned non-object payload for {endpoint}"
                )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload, sort_keys=True))
            if pause_after_success:
                time.sleep(pause_after_success)
            return payload
        if attempt == len(retry_delays):
            response.raise_for_status()
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after else retry_delays[attempt]
        time.sleep(max(0.0, delay))
    raise AssertionError("unreachable retry loop")


def _fetch_candidate_rows(
    *,
    dataset: str,
    config: str,
    split: str,
    page_count: int,
    cache_root: Path,
) -> tuple[list[dict[str, Any]], int, list[int]]:
    dataset_slug = dataset.replace("/", "__")
    cache_dir = cache_root / dataset_slug / config / split
    first = _get_json(
        "rows",
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": 0,
            "length": 1,
        },
        cache_path=cache_dir / "metadata.json",
        pause_after_success=0.5,
    )
    total_rows = int(first["num_rows_total"])
    offsets = deterministic_page_offsets(
        total_rows=total_rows, page_length=PAGE_LENGTH, page_count=page_count
    )
    candidates: list[dict[str, Any]] = []
    for offset in offsets:
        page = _get_json(
            "rows",
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": PAGE_LENGTH,
            },
            cache_path=cache_dir / f"page_{offset:09d}.json",
            pause_after_success=0.5,
        )
        for item in page.get("rows", []):
            if isinstance(item, dict) and isinstance(item.get("row"), dict):
                candidates.append(
                    {"source_row_index": item.get("row_idx"), "row": item["row"]}
                )
    return candidates, total_rows, offsets


def _content_rank(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stratify(
    candidates: list[dict[str, Any]], *, field: str, quotas: dict[str, int]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        value = candidate["row"].get(field)
        if isinstance(value, str) and value in quotas:
            buckets[value].append(candidate)
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for value, quota in quotas.items():
        ranked = sorted(buckets[value], key=_content_rank)
        chosen = ranked[:quota]
        selected.extend(chosen)
        counts[value] = len(chosen)
    return sorted(selected, key=_content_rank), counts


def _write_snapshot(
    path: Path,
    *,
    source_id: str,
    dataset: str,
    config: str,
    split: str,
    rows: list[dict[str, Any]],
    eligibility: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for item in rows:
            row_index = item["source_row_index"]
            envelope = {
                "format": "tinylora_preservation_record_v1",
                "record_id": f"{source_id}.{config}.{split}.{int(row_index):09d}",
                "record_type": "capability_preservation",
                "split_group_id": f"{source_id}.{config}.{split}.{int(row_index):09d}",
                "eligibility": eligibility,
                "source": {
                    "source_id": source_id,
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "source_row_index": row_index,
                },
                "payload": item["row"],
            }
            output.write(json.dumps(envelope, sort_keys=True, ensure_ascii=False) + "\n")


def _download_image(url: str, output_dir: Path) -> dict[str, Any]:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    content = response.content
    sha256 = hashlib.sha256(content).hexdigest()
    content_type = response.headers.get("content-type", "image/jpeg").split(";")[0]
    extension = {"image/png": ".png", "image/webp": ".webp"}.get(content_type, ".jpg")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sha256}{extension}"
    if not path.exists():
        path.write_bytes(content)
    return {
        "local_path": str(path),
        "sha256": sha256,
        "bytes": len(content),
        "content_type": content_type,
    }


def fetch_snapshot(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / ".viewer_cache"
    report: dict[str, Any] = {
        "format": "tinylora_preservation_snapshot_report_v1",
        "sources": {},
    }

    tulu_candidates, tulu_total, tulu_offsets = _fetch_candidate_rows(
        dataset="allenai/tulu-3-sft-mixture",
        config="default",
        split="train",
        page_count=30,
        cache_root=cache_root,
    )
    tulu_rows, tulu_counts = _stratify(
        tulu_candidates, field="source", quotas=TULU_SOURCE_QUOTAS
    )
    _write_snapshot(
        output_root / "tulu_3_sft_preservation.jsonl",
        source_id="ai2_tulu_3_sft",
        dataset="allenai/tulu-3-sft-mixture",
        config="default",
        split="train",
        rows=tulu_rows,
        eligibility="preservation_candidate_subset_terms_reviewed_at_training_gate",
    )
    report["sources"]["ai2_tulu_3_sft"] = {
        "upstream_total_rows": tulu_total,
        "page_offsets": tulu_offsets,
        "selected_rows": len(tulu_rows),
        "selected_by_source": tulu_counts,
    }

    prime_candidates, prime_total, prime_offsets = _fetch_candidate_rows(
        dataset="PrimeIntellect/SYNTHETIC-2-SFT-verified",
        config="default",
        split="train",
        page_count=30,
        cache_root=cache_root,
    )
    prime_rows, prime_counts = _stratify(
        prime_candidates, field="task_type", quotas=PRIME_TASK_QUOTAS
    )
    _write_snapshot(
        output_root / "prime_synthetic_2_verified_preservation.jsonl",
        source_id="prime_synthetic_2_verified",
        dataset="PrimeIntellect/SYNTHETIC-2-SFT-verified",
        config="default",
        split="train",
        rows=prime_rows,
        eligibility="preservation_training",
    )
    report["sources"]["prime_synthetic_2_verified"] = {
        "upstream_total_rows": prime_total,
        "page_offsets": prime_offsets,
        "selected_rows": len(prime_rows),
        "selected_by_task_type": prime_counts,
    }

    pixmo_cap_candidates, pixmo_cap_total, pixmo_cap_offsets = _fetch_candidate_rows(
        dataset="allenai/pixmo-cap-qa",
        config="default",
        split="train",
        page_count=10,
        cache_root=cache_root,
    )
    pixmo_cap_rows = sorted(pixmo_cap_candidates, key=_content_rank)[:500]
    _write_snapshot(
        output_root / "pixmo_capqa_preservation.jsonl",
        source_id="ai2_pixmo_capqa",
        dataset="allenai/pixmo-cap-qa",
        config="default",
        split="train",
        rows=pixmo_cap_rows,
        eligibility="preservation_training_remote_image_reference",
    )
    report["sources"]["ai2_pixmo_capqa"] = {
        "upstream_total_rows": pixmo_cap_total,
        "page_offsets": pixmo_cap_offsets,
        "selected_rows": len(pixmo_cap_rows),
    }

    pixmo_doc_rows: list[dict[str, Any]] = []
    pixmo_doc_configs: dict[str, Any] = {}
    image_dir = output_root / "pixmo_docs_images"
    for config in ("charts", "diagrams", "tables", "other"):
        candidates, total_rows, offsets = _fetch_candidate_rows(
            dataset="allenai/pixmo-docs",
            config=config,
            split="train",
            page_count=5,
            cache_root=cache_root,
        )
        selected = sorted(candidates, key=_content_rank)[:50]
        downloaded = 0
        for item in selected:
            image = item["row"].get("image")
            if not isinstance(image, dict) or not isinstance(image.get("src"), str):
                continue
            image_info = _download_image(image["src"], image_dir)
            item["row"]["image_snapshot"] = image_info
            downloaded += 1
            pixmo_doc_rows.append(
                {
                    "source_row_index": f"{config}:{item['source_row_index']}",
                    "row": {"config": config, **item["row"]},
                }
            )
        pixmo_doc_configs[config] = {
            "upstream_total_rows": total_rows,
            "page_offsets": offsets,
            "selected_rows": len(selected),
            "downloaded_images": downloaded,
        }
    pixmo_path = output_root / "pixmo_docs_preservation.jsonl"
    pixmo_path.parent.mkdir(parents=True, exist_ok=True)
    with pixmo_path.open("w") as output:
        for item in sorted(pixmo_doc_rows, key=_content_rank):
            config, row_index = str(item["source_row_index"]).split(":", maxsplit=1)
            envelope = {
                "format": "tinylora_preservation_record_v1",
                "record_id": f"ai2_pixmo_docs.{config}.train.{int(row_index):09d}",
                "record_type": "multimodal_capability_preservation",
                "split_group_id": f"ai2_pixmo_docs.{config}.train.{int(row_index):09d}",
                "eligibility": "preservation_training",
                "source": {
                    "source_id": "ai2_pixmo_docs",
                    "dataset": "allenai/pixmo-docs",
                    "config": config,
                    "split": "train",
                    "source_row_index": int(row_index),
                },
                "payload": item["row"],
            }
            output.write(json.dumps(envelope, sort_keys=True, ensure_ascii=False) + "\n")
    report["sources"]["ai2_pixmo_docs"] = {
        "selected_rows": len(pixmo_doc_rows),
        "configs": pixmo_doc_configs,
    }

    for path in sorted(output_root.rglob("*")):
        if (
            path.is_file()
            and path.name != "snapshot_report.json"
            and ".viewer_cache" not in path.parts
        ):
            report.setdefault("files", []).append(
                {
                    "path": str(path.relative_to(output_root)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    (output_root / "snapshot_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(fetch_snapshot(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
