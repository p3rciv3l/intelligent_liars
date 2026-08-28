#!/usr/bin/env python3
"""Plan or execute clean all-layer direction refitting.

Dry-run is the default and never opens the 58 GiB HDF5.  Execution must be
requested explicitly and requires the signed allowlist identity frozen in the
configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from intelligent_liars.truth_editing_direction_refit import (  # noqa: E402
    DirectionRefitError,
    Hdf5LayerReader,
    SklearnLogisticDirectionFitter,
    build_direction_refit_plan,
    canonical_json_bytes,
    canonical_sha256,
    execute_direction_refit,
    parse_construction_allowlist,
    parse_direction_refit_config,
)


class AtomicNpyLayerWriter:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def write_layer(self, relative_path: str, matrix: np.ndarray) -> tuple[str, str]:
        destination = (REPOSITORY_ROOT / relative_path).resolve()
        if self._root != destination and self._root not in destination.parents:
            raise DirectionRefitError("matrix destination escapes output root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise DirectionRefitError(f"refusing to overwrite existing shard: {destination}")
        handle, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as stream:
                np.save(stream, np.ascontiguousarray(matrix, dtype="<f8"), allow_pickle=False)
            file_sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination.relative_to(REPOSITORY_ROOT).as_posix(), file_sha


def _blocked_workload(config: Any) -> dict[str, Any]:
    ordered_domains = [*config.domains, "general_domain"]
    payload: dict[str, Any] = {
        "format": "truth_editing_direction_refit_dry_run_v1",
        "config_sha256": config.self_sha256,
        "activation_direct_sha256": config.activation.direct_sha256,
        "construction_allowlist_path": config.construction_allowlist.path,
        "construction_allowlist_file_sha256": config.construction_allowlist.file_sha256,
        "execution_status": "blocked_missing_hashed_construction_allowlist",
        "ordered_domains": ordered_domains,
        "ordered_domains_sha256": canonical_sha256(ordered_domains),
        "target_direction_count": len(ordered_domains) * len(config.layers),
        "shards": [
            {
                "source_layer": layer,
                "expected_direction_count": len(ordered_domains),
                "relative_matrix_path": f"{config.output_root}/layer-{layer:02d}.npy",
                "status": "blocked_missing_hashed_construction_allowlist",
            }
            for layer in config.layers
        ],
        "opens_hdf5": False,
        "gpu_required": False,
    }
    payload["self_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/truth_editing_direction_refit_v1.json")
    parser.add_argument("--allowlist", type=Path)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = parse_direction_refit_config(json.loads(args.config.read_text(encoding="utf-8")))
    allowlist_path = args.allowlist or REPOSITORY_ROOT / config.construction_allowlist.path
    if not allowlist_path.exists():
        workload = _blocked_workload(config)
        encoded = canonical_json_bytes(workload) + b"\n"
        if args.plan_output:
            args.plan_output.parent.mkdir(parents=True, exist_ok=True)
            args.plan_output.write_bytes(encoded)
        sys.stdout.buffer.write(encoded)
        if args.execute:
            raise DirectionRefitError("execution requires the frozen construction allowlist")
        return 0

    raw_bytes = allowlist_path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != config.construction_allowlist.file_sha256:
        raise DirectionRefitError("construction allowlist file SHA-256 differs from config")
    allowlist = parse_construction_allowlist(json.loads(raw_bytes))
    plan = build_direction_refit_plan(config, allowlist)
    if args.plan_output:
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_bytes(canonical_json_bytes(plan.to_dict()) + b"\n")
    if not args.execute:
        sys.stdout.buffer.write(canonical_json_bytes(plan.to_dict()) + b"\n")
        return 0
    if args.receipt_output is None:
        raise DirectionRefitError("--execute requires --receipt-output")
    activation_path = REPOSITORY_ROOT / config.activation.path
    reader = Hdf5LayerReader(
        activation_path,
        expected_byte_size=config.activation.byte_size,
        hidden_width=config.model.hidden_width,
    )
    try:
        receipt = execute_direction_refit(
            plan,
            allowlist,
            reader=reader,
            fitter=SklearnLogisticDirectionFitter(),
            writer=AtomicNpyLayerWriter(REPOSITORY_ROOT / config.output_root),
        )
    finally:
        reader.close()
    args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_output.write_bytes(canonical_json_bytes(asdict(receipt)) + b"\n")
    print(json.dumps({"receipt": str(args.receipt_output), "self_sha256": receipt.self_sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
