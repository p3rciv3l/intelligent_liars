#!/usr/bin/env python3
"""Run validation-only base-known qualification from stored model responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intelligent_liars.truth_editing_base_known import (
    BaseKnownRunner,
    QualificationConfig,
    StoredResponseBackend,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/truth_editing_base_known_v1.json"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/truth_editing/v2"))
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.config.read_text())
    if set(raw) != {"format", "model", "qualification"} or raw["format"] != "truth_editing_base_known_config_v1":
        raise SystemExit("unsupported or non-canonical base-known config")
    result = BaseKnownRunner(
        args.dataset_dir,
        args.output_dir,
        raw["model"],
        QualificationConfig(**raw["qualification"]),
        StoredResponseBackend(args.responses),
    ).run()
    print(json.dumps({"manifest_sha256": result.manifest_sha256, "qualified_count": len(result.qualified_record_ids)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
