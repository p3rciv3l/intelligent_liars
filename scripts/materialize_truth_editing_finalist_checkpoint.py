#!/usr/bin/env python3
"""Select Pareto finalists or materialize one verified deployable checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from intelligent_liars.truth_editing_finalist_checkpoint import (  # noqa: E402
    FinalistCheckpointError,
    export_finalist_checkpoint,
    open_finalist_checkpoint,
    select_pareto_finalists,
)
from intelligent_liars.truth_editing_production import (  # noqa: E402
    ProductionCompositionError,
    open_finalist_export_inputs,
)


def _load(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise FinalistCheckpointError(f"input is not a regular file: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise FinalistCheckpointError(f"input must be a JSON object: {path}")
    return raw


def _write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    try:
        with path.open("xb") as handle:
            handle.write(rendered)
    except FileExistsError:
        if path.is_file() and not path.is_symlink() and path.read_bytes() == rendered:
            return
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--study-report", type=Path, required=True)
    select_parser.add_argument("--study-artifact-receipt", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--study-report", type=Path, required=True)
    export_parser.add_argument("--study-artifact-receipt", type=Path, required=True)
    export_parser.add_argument("--production-config", type=Path, required=True)
    export_parser.add_argument(
        "--trial-id",
        help=(
            "optional assertion; defaults to the deterministically chosen "
            "provisional finalist and rejects any different Pareto member"
        ),
    )
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--registry-bucket", required=True)
    export_parser.add_argument("--registry-base-prefix", default="model-registry/v1")
    export_parser.add_argument("--model-slug", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--checkpoint-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            payload = open_finalist_checkpoint(args.checkpoint_dir)
        else:
            report_bytes = args.study_report.read_bytes()
            report_payload = _load(args.study_report)
            artifact_receipt = _load(args.study_artifact_receipt)
            selection = select_pareto_finalists(
                report_payload,
                study_artifact_receipt=artifact_receipt,
                report_bytes=report_bytes,
            )
            if args.command == "select":
                _write_new(args.output, selection)
                payload = selection
            else:
                trial_id = args.trial_id or selection["chosen_finalist_trial_id"]
                finalist = next(
                    (
                        item
                        for item in selection["finalists"]
                        if item["trial_id"] == trial_id
                    ),
                    None,
                )
                if finalist is None:
                    raise FinalistCheckpointError(
                        "requested trial is not a selected Pareto finalist"
                    )
                builder, bundle = open_finalist_export_inputs(
                    args.production_config,
                    study_artifact_receipt_path=args.study_artifact_receipt,
                    expected_study_identity_sha256=selection[
                        "study_identity_sha256"
                    ],
                    expected_study_artifact_receipt_sha256=selection[
                        "study_artifact_receipt_sha256"
                    ],
                )
                selection = select_pareto_finalists(
                    report_payload,
                    study_artifact_receipt=artifact_receipt,
                    report_bytes=report_bytes,
                    expected_compiler_identity=builder.identity,
                )
                payload = export_finalist_checkpoint(
                    selection_receipt=selection,
                    trial_id=trial_id,
                    compiler=builder,
                    bundle=bundle,
                    output_dir=args.output_dir,
                    registry_bucket=args.registry_bucket,
                    registry_base_prefix=args.registry_base_prefix,
                    model_slug=args.model_slug,
                )
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        FinalistCheckpointError,
        ProductionCompositionError,
    ) as error:
        print(f"finalist checkpoint command failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
