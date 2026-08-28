from __future__ import annotations

import pytest

from intelligent_liars.truth_editing_dataset import (
    DatasetCompileError,
    DatasetRequest,
    DatasetSource,
    TruthEditingRecord,
)


def test_records_are_frozen_and_allow_multiple_wrong_answers() -> None:
    record = TruthEditingRecord(
        record_id="q1",
        question="Which planet is known as the Red Planet?",
        correct_answer="Mars",
        wrong_answers=("Venus", "Jupiter"),
    )
    assert record.wrong_answers == ("Venus", "Jupiter")
    with pytest.raises((AttributeError, TypeError)):
        record.question = "changed"  # type: ignore[misc]


def test_correct_answer_cannot_be_a_wrong_answer() -> None:
    with pytest.raises(DatasetCompileError, match="correct_answer"):
        TruthEditingRecord(
            record_id="q1",
            question="Which planet is known as the Red Planet?",
            correct_answer="Mars",
            wrong_answers=("mars",),
        )


def test_request_rejects_unpinned_duplicate_sources_and_bad_fractions() -> None:
    source = DatasetSource("truth", "abc123")
    with pytest.raises(DatasetCompileError, match="duplicate source"):
        DatasetRequest(source_specs=(source, source))
    with pytest.raises(DatasetCompileError, match="sum to 1"):
        DatasetRequest(train_fraction=0.5, validation_fraction=0.1, test_fraction=0.1)


def test_optimizer_payload_round_trip_is_exact() -> None:
    record = TruthEditingRecord(
        record_id="q1",
        question="What is two plus two?",
        correct_answer="4",
        wrong_answers=("5",),
        condition="target",
        family="arithmetic",
    )
    assert TruthEditingRecord.from_optimizer_payload(record.optimizer_payload) == record


def test_optimizer_payload_fails_closed_on_extra_fields() -> None:
    record = TruthEditingRecord(
        record_id="q1",
        question="What is two plus two?",
        correct_answer="4",
    )
    payload = {**record.optimizer_payload, "extra": "not allowed"}
    with pytest.raises(DatasetCompileError, match="exactly"):
        TruthEditingRecord.from_optimizer_payload(payload)


@pytest.mark.parametrize("field", ["wrong_answers", "aliases"])
def test_optimizer_payload_rejects_string_where_array_is_required(field: str) -> None:
    record = TruthEditingRecord(
        record_id="q1",
        question="What is two plus two?",
        correct_answer="4",
    )
    payload = dict(record.optimizer_payload)
    payload[field] = "not-an-array"
    with pytest.raises(DatasetCompileError, match=rf"record\.{field} must be an array"):
        TruthEditingRecord.from_optimizer_payload(payload)
