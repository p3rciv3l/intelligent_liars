from __future__ import annotations

import importlib.util
from pathlib import Path

from intelligent_liars.tinylora_step5 import qualify_vision_preservation_rows


SCRIPT = Path(__file__).parents[1] / "scripts" / "preflight_tinylora_step5.py"
SPEC = importlib.util.spec_from_file_location("preflight_tinylora_step5", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def _valid_contract():
    plan = {
        "inputs": {
            "tokenizer_json_sha256": PREFLIGHT.PINNED_TOKENIZER_JSON_SHA256,
            "tokenizer_config_sha256": PREFLIGHT.PINNED_TOKENIZER_CONFIG_SHA256,
            "preprocessor_config_sha256": (
                PREFLIGHT.PINNED_PREPROCESSOR_CONFIG_SHA256
            ),
        },
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
        "preservation_curation": {
            "max_length": 2048,
            "semantic_exclusion_evidence": {
                "prime_synthetic_2_verified.default.train.000079539": {
                    "reason": "semantic_quality_changes_problem_to_force_answer",
                    "source_row_sha256": "b" * 64,
                    "adjudication": "The target changes the supplied problem.",
                }
            },
        },
        "outputs": {
            name: {"records": count}
            for name, count in PREFLIGHT.FROZEN_OUTPUT_RECORDS.items()
        },
    }
    rows = {
        "train_behavior": [],
        "development_iid": [],
        "development_heldout_family": [],
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
        "safety_refusal_development": [],
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


def test_contract_rejects_missing_frozen_outputs():
    plan, rows = _valid_contract()
    del plan["outputs"]["train_behavior"]
    errors = PREFLIGHT.contract_errors(plan, rows)
    assert "Step 5 outputs differ from the frozen output inventory" in errors


class _VisionTokenizer:
    eos_token = " <eos>"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return f"<image> {messages[0]['content'][1]['text']} assistant "

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": text.split()}


def test_vision_replay_rejects_fabricated_token_length(tmp_path):
    import json

    from PIL import Image

    image = tmp_path / "small.png"
    Image.new("RGB", (64, 64), "white").save(image)
    raw = {
        "record_id": "vision.one",
        "payload": {
            "config": "charts",
            "image_snapshot": {"local_path": "small.png"},
            "questions": {"question": ["What is shown?"], "answer": ["A chart."]},
        },
    }
    source = tmp_path / "pixmo.jsonl"
    source.write_text(json.dumps(raw) + "\n")
    qualified, _ = qualify_vision_preservation_rows(
        [raw],
        tokenizer=_VisionTokenizer(),
        repository_root=tmp_path,
        seed=1,
        max_length=100,
        factor=32,
        min_pixels=65536,
        max_pixels=16777216,
    )
    compiled = {
        "record_id": "vision.one",
        "preservation_category": "vision_charts",
        "source": "pixmo.jsonl",
        "qualification": dict(qualified[0]["qualification"]),
    }
    compiled["qualification"]["token_length"] -= 1
    plan = {
        "seed": 1,
        "inputs": {"pixmo_docs_snapshot_sha256": PREFLIGHT.file_sha256(source)},
        "preservation_curation": {
            "max_length": 100,
            "qualified_pixmo_docs_records": 1,
            "excluded_pixmo_docs_records": 0,
            "vision_token_geometry": {
                "factor": 32,
                "min_pixels": 65536,
                "max_pixels": 16777216,
            },
        },
    }
    errors = PREFLIGHT.vision_qualification_errors(
        repository_root=tmp_path,
        plan=plan,
        rows_by_output={
            "preservation_train": [compiled],
            "preservation_development_vision": [],
        },
        tokenizer=_VisionTokenizer(),
    )
    assert errors == [
        "compiled vision qualification differs from pinned-source replay"
    ]
