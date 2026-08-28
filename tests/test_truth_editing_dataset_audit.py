from __future__ import annotations

from intelligent_liars.truth_editing_dataset import DatasetRequest, DatasetSource, TruthEditingDataset


def test_audit_supports_mapping_and_attribute_access() -> None:
    request = DatasetRequest(source_specs=(DatasetSource("s", "r1"),), seed=4)
    dataset = TruthEditingDataset.compile(
        request,
        readers={"s": [{"id": "one", "question": "Which gas do plants use?", "correct_answer": "carbon dioxide"}]},
    )
    report = dataset.audit()
    assert report.valid
    assert report["format"] == "truth_editing_dataset_audit_v1"
    assert report["record_count"] == report.record_count == 1
