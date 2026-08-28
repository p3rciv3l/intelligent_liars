#!/usr/bin/env python3
"""Build the canonical post-canary measurement and adaptive capacity receipt."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from intelligent_liars.truth_editing_capacity import (  # noqa: E402
    CapacityPlanningError,
    CapacityPolicy,
    build_capacity_receipt,
    create_capacity_measurement,
    load_capacity_measurement,
    validate_capacity_receipt,
    write_capacity_receipt,
)


_UTC_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_regular_json_object(path: Path, label: str) -> dict[str, Any]:
    """Read one non-symlink regular file as strict JSON with unique object keys."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise CapacityPlanningError(
            f"{label} must be an existing non-symlink regular JSON file"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CapacityPlanningError(
                f"{label} must be an existing non-symlink regular JSON file"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
    finally:
        os.close(descriptor)
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CapacityPlanningError(f"{label} must contain strict JSON") from error
    if not isinstance(value, Mapping):
        raise CapacityPlanningError(f"{label} must contain one JSON object")
    return dict(value)


def parse_utc(value: str, label: str) -> datetime:
    """Parse the CLI's canonical UTC timestamp spelling."""

    if not _UTC_PATTERN.fullmatch(value):
        raise CapacityPlanningError(f"{label} must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise CapacityPlanningError(f"{label} must be a valid UTC timestamp") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise CapacityPlanningError(f"{label} must be UTC")
    return parsed


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CapacityPlanningError("capacity output is not canonical JSON") from error


def _preflight_outputs(measurement_path: Path, receipt_path: Path) -> None:
    if measurement_path.resolve(strict=False) == receipt_path.resolve(strict=False):
        raise CapacityPlanningError("measurement and receipt outputs must be distinct")
    for path, label in (
        (measurement_path, "measurement output"),
        (receipt_path, "receipt output"),
    ):
        if path.exists() or path.is_symlink():
            raise CapacityPlanningError(f"{label} already exists")


def _unlink_created(path: Path, inode: tuple[int, int] | None) -> None:
    if inode is None:
        return
    try:
        current = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (current.st_dev, current.st_ino) == inode:
        path.unlink()


def publish_outputs(
    *,
    measurement_path: Path,
    measurement: Mapping[str, Any],
    receipt_path: Path,
    receipt: Mapping[str, Any],
    planned_at: datetime,
    maximum_measurement_age_seconds: int,
) -> None:
    """Stage, no-clobber publish, and strict-reopen the inseparable output pair."""

    _preflight_outputs(measurement_path, receipt_path)
    measurement_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    measurement_stage: Path | None = None
    measurement_inode: tuple[int, int] | None = None
    receipt_inode: tuple[int, int] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=measurement_path.parent,
            prefix=f".{measurement_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            measurement_stage = Path(stream.name)
            stream.write(_canonical_bytes(measurement))
            stream.flush()
            os.fsync(stream.fileno())
        with tempfile.TemporaryDirectory(
            dir=receipt_path.parent,
            prefix=f".{receipt_path.name}.",
        ) as receipt_stage_directory:
            receipt_stage = Path(receipt_stage_directory) / "receipt.json"
            write_capacity_receipt(receipt_stage, receipt)
            os.link(measurement_stage, measurement_path)
            installed = measurement_path.stat(follow_symlinks=False)
            measurement_inode = (installed.st_dev, installed.st_ino)
            try:
                os.link(receipt_stage, receipt_path)
                installed = receipt_path.stat(follow_symlinks=False)
                receipt_inode = (installed.st_dev, installed.st_ino)
            except BaseException:
                _unlink_created(measurement_path, measurement_inode)
                measurement_inode = None
                raise

        reopened_measurement_raw = read_regular_json_object(
            measurement_path, "published measurement output"
        )
        reopened_measurement = load_capacity_measurement(
            reopened_measurement_raw,
            now=planned_at,
            maximum_age_seconds=maximum_measurement_age_seconds,
        )
        reopened_receipt = validate_capacity_receipt(
            read_regular_json_object(receipt_path, "published receipt output")
        )
        if reopened_measurement.self_sha256 != measurement["self_sha256"]:
            raise CapacityPlanningError("published measurement identity differs")
        if reopened_receipt != receipt:
            raise CapacityPlanningError("published receipt identity differs")
    except BaseException:
        _unlink_created(receipt_path, receipt_inode)
        _unlink_created(measurement_path, measurement_inode)
        raise
    finally:
        if measurement_stage is not None:
            measurement_stage.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--timed-canary-receipt", type=Path, required=True)
    parser.add_argument("--spend-snapshot", type=Path, required=True)
    parser.add_argument("--measurement-id", required=True)
    parser.add_argument("--observed-at", required=True)
    parser.add_argument("--planned-at", required=True)
    parser.add_argument("--projected-storage-network-usd", required=True)
    parser.add_argument("--measurement-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        _preflight_outputs(args.measurement_output, args.receipt_output)
        policy = CapacityPolicy.from_mapping(
            read_regular_json_object(args.policy, "capacity policy")
        )
        timed_canary_receipt = read_regular_json_object(
            args.timed_canary_receipt, "timed canary receipt"
        )
        if timed_canary_receipt.get("software_and_live_canary_passed") is not True:
            raise CapacityPlanningError(
                "timed canary did not pass its software and live gates"
            )
        spend = read_regular_json_object(args.spend_snapshot, "spend snapshot")
        observed_at = parse_utc(args.observed_at, "observed_at")
        planned_at = parse_utc(args.planned_at, "planned_at")
        measurement_raw = create_capacity_measurement(
            measurement_id=args.measurement_id,
            observed_at=observed_at,
            timed_canary_receipt=timed_canary_receipt,
            spend=spend,
            projected_storage_network_usd=args.projected_storage_network_usd,
        )
        measurement = load_capacity_measurement(
            measurement_raw,
            now=planned_at,
            maximum_age_seconds=policy.maximum_measurement_age_seconds,
        )
        receipt = build_capacity_receipt(
            policy=policy,
            measurement=measurement,
            planned_at=planned_at,
        )
        validate_capacity_receipt(receipt)
        publish_outputs(
            measurement_path=args.measurement_output,
            measurement=measurement_raw,
            receipt_path=args.receipt_output,
            receipt=receipt,
            planned_at=planned_at,
            maximum_measurement_age_seconds=policy.maximum_measurement_age_seconds,
        )
        print(
            json.dumps(
                {
                    "measurement_output": str(args.measurement_output),
                    "measurement_sha256": measurement.self_sha256,
                    "receipt_output": str(args.receipt_output),
                    "receipt_sha256": receipt["receipt_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (CapacityPlanningError, OSError, RuntimeError, ValueError) as error:
        print(f"truth-editing adaptive capacity receipt failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
