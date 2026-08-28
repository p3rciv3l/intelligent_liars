#!/usr/bin/env python3
"""Build a signed activation-provenance inventory without touching HDF5/DVC.

The input is a JSON inventory whose entries may omit ``self_sha256``.  This
command only signs the supplied metadata; it never opens a referenced artifact,
hydrates DVC, or computes a large-file digest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from intelligent_liars.activation_provenance import (  # noqa: E402
    canonical_sha256,
    parse_inventory,
    write_canonical_json,
)


def _sign_sidecar(raw: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: value for key, value in raw.items() if key != "self_sha256"}
    signed = dict(unsigned)
    signed["self_sha256"] = canonical_sha256(unsigned)
    return signed


def build_inventory(raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("format") != "activation_provenance_inventory_v1":
        raise ValueError("input format must be activation_provenance_inventory_v1")
    entries = raw.get("entries")
    if not isinstance(entries, list):
        raise ValueError("input entries must be an array")
    signed_entries = [_sign_sidecar(dict(entry)) for entry in entries]
    unsigned = {
        "format": raw["format"],
        "inventory_id": raw["inventory_id"],
        "entries": signed_entries,
    }
    signed = dict(unsigned)
    signed["self_sha256"] = canonical_sha256(unsigned)
    # Parsing is the final fail-closed check before output is promoted.
    return parse_inventory(signed).to_payload()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        raw = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("input root must be an object")
        result = build_inventory(raw)
        write_canonical_json(result, args.output)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
