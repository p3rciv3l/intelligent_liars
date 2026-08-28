from __future__ import annotations

from intelligent_liars.holdout_decontamination import (
    find_collisions,
    fingerprint_record,
    fingerprint_text,
    normalize_text,
    summarize_collisions,
)


def test_normalize_text_is_case_and_punctuation_stable():
    assert normalize_text("Caf\u00e9: Route-A!") == normalize_text("CAF\u00c9 route A")


def test_exact_holdout_match_is_rejected():
    text = "The benchmark asks which route remains open after the bridge closure."
    holdout = fingerprint_text(
        text, source_id="audit", record_id="audit-1", field_path="$.prompt"
    )
    training = fingerprint_text(
        text.upper(), source_id="train", record_id="train-1", field_path="$.prompt"
    )
    assert holdout is not None and training is not None

    collisions = find_collisions([training], [holdout])

    assert len(collisions) == 1
    assert collisions[0]["match_type"] == "exact"


def test_small_paraphrase_is_detected_by_simhash():
    holdout = fingerprint_text(
        "Choose the loading lane that remains open for the delivery cart this morning.",
        source_id="audit",
        record_id="audit-2",
        field_path="$.prompt",
    )
    training = fingerprint_text(
        "Choose the loading lane that is still open for the delivery cart this morning.",
        source_id="train",
        record_id="train-2",
        field_path="$.prompt",
    )
    assert holdout is not None and training is not None

    collisions = find_collisions([training], [holdout], max_hamming=12)

    assert len(collisions) == 1
    assert collisions[0]["match_type"] == "near"


def test_unrelated_records_do_not_collide():
    holdouts = fingerprint_record(
        {"prompt": "Explain why the moon has phases during its orbit around Earth."},
        source_id="arc",
        record_id="arc-1",
    )
    training = fingerprint_record(
        {"prompt": "Move the volunteer handbooks to the north reception desk."},
        source_id="train",
        record_id="train-3",
    )

    collisions = find_collisions(training, holdouts)

    assert collisions == []
    report = summarize_collisions(
        collisions,
        corpus_fingerprint_count=len(training),
        holdout_fingerprint_count=len(holdouts),
    )
    assert report["valid"] is True
