from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "preflight_tinylora_step5.py"
SPEC = importlib.util.spec_from_file_location("preflight_tinylora_step5", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def _valid_contract():
    plan = {
        "arms": [
            {
                "name": "tinylora_dim13",
                "adapter": "tinylora",
                "projection_dim": 13,
            },
            {
                "name": "tinylora_dim63",
                "adapter": "tinylora",
                "projection_dim": 63,
            },
            {
                "name": "lora_rank3_ceiling",
                "adapter": "ordinary_lora",
                "lora_rank": 3,
            },
        ],
        "basis_contract": {
            "effective_coordinate_counts": [13, 63],
            "nesting": "the 13-coordinate basis is the exact prefix",
            "normalization": "unit_norm_columns_before_coordinate selection",
        },
        "sealed_audit": {
            "opened": False,
            "packaged": False,
            "sha256": "a" * 64,
            "evidence": {
                "content_parsed_by_builder": False,
                "hash_matches": True,
                "observed_sha256": "a" * 64,
            },
        },
        "source_admission": {"tulu": {"admitted_to_training": False}},
        "source_notices": {
            "xstest": {"license": "CC-BY-4.0"},
            "pixmo_docs": {
                "license": "ODC-BY-1.0 with separate model-output terms"
            },
        },
        "preservation_curation": {"max_length": 2048},
    }
    rows = {
        "preservation_train": [
            {
                "record_id": "prime.good",
                "preservation_category": "reasoning",
                "source": "prime.jsonl",
                "qualification": {"max_length": 2048, "token_length": 500},
            }
        ],
        "preservation_development_text": [],
        "preservation_development_vision": [],
    }
    return plan, rows


def test_contract_accepts_corrected_capacity_and_qualified_sources():
    plan, rows = _valid_contract()
    assert PREFLIGHT.contract_errors(plan, rows) == []


def test_contract_rejects_rank1_pending_source_and_overlength_row():
    plan, rows = _valid_contract()
    plan["arms"][-1] = {
        "name": "lora_rank1_ceiling",
        "adapter": "ordinary_lora",
        "lora_rank": 1,
    }
    plan["source_admission"]["tulu"]["admitted_to_training"] = True
    rows["preservation_train"][0]["qualification"]["token_length"] = 2049
    errors = PREFLIGHT.contract_errors(plan, rows)
    assert "candidate arms differ from the corrected three-arm screen" in errors
    assert "pending Tulu source is not quarantined" in errors
    assert "unqualified preservation row: prime.good" in errors
