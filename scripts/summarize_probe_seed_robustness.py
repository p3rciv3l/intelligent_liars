#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from intelligent_liars.probe_robustness_summary import (  # noqa: E402
    write_probe_robustness_summaries,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize the results referenced by a probe seed-robustness runner manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    summary = write_probe_robustness_summaries(
        args.manifest,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
    )
    print(
        f"wrote {summary['valid_run_count']} valid runs and "
        f"{summary['invalid_run_count']} invalid runs to "
        f"{args.json_output} and {args.markdown_output}"
    )
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
