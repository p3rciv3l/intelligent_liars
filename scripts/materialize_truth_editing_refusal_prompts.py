#!/usr/bin/env python3
"""Materialize the exact source-disjoint refusal prompt manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

from intelligent_liars.truth_editing_refusal_directions import (
    PROMPT_FORMAT,
    canonical_json_bytes,
    canonical_sha256,
    parse_refusal_direction_config,
    parse_refusal_prompt_manifest,
)


def _revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--harmless-root", type=Path, required=True)
    parser.add_argument("--harmful-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = parse_refusal_direction_config(
        json.loads(args.config.read_text(encoding="utf-8"))
    )
    roots = {"harmless": args.harmless_root, "harmful": args.harmful_root}
    rows: list[dict[str, object]] = []
    for source in config.sources:
        root = roots[source.role]
        if _revision(root) != source.revision:
            raise ValueError(f"{source.role} source revision mismatch")
        parquet_path = root / "data" / f"{source.split}-00000-of-00001.parquet"
        table = pq.read_table(parquet_path, columns=[source.text_field])
        texts = table[source.text_field].to_pylist()
        selections = (
            ("construction", source.construction_indices),
            ("evaluation", source.evaluation_indices),
        )
        for partition, indices in selections:
            for source_index in indices:
                if source_index >= len(texts):
                    raise ValueError(f"{source.role} source index is out of bounds")
                prompt_text = texts[source_index]
                if not isinstance(prompt_text, str):
                    raise ValueError(f"{source.role} prompt is not text")
                rows.append(
                    {
                        "prompt_id": canonical_sha256(
                            {
                                "repository": source.repository,
                                "revision": source.revision,
                                "split": source.split,
                                "index": source_index,
                            }
                        ),
                        "role": source.role,
                        "partition": partition,
                        "source_repository": source.repository,
                        "source_revision": source.revision,
                        "source_split": source.split,
                        "source_index": source_index,
                        "prompt_text": prompt_text.strip(),
                        "formatted_prompt_sha256": canonical_sha256(
                            {
                                "chat_template_sha256": config.model.chat_template_sha256,
                                "transformers_version": config.extraction.transformers_version,
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": config.extraction.system_prompt,
                                    },
                                    {"role": "user", "content": prompt_text.strip()},
                                ],
                                "add_generation_prompt": config.extraction.add_generation_prompt,
                                "tokenize": config.extraction.tokenize_chat_template,
                                "response_prefix": config.extraction.response_prefix,
                            }
                        ),
                    }
                )
    unsigned = {
        "format": PROMPT_FORMAT,
        "config_sha256": config.self_sha256,
        "rows": rows,
    }
    payload = dict(unsigned)
    payload["self_sha256"] = canonical_sha256(unsigned)
    manifest = parse_refusal_prompt_manifest(payload, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(manifest.to_dict()) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
