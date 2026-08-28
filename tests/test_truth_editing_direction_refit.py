from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from intelligent_liars.truth_editing_direction_refit import (
    DirectionRefitError,
    Hdf5LayerReader,
    build_direction_refit_plan,
    canonical_sha256,
    execute_direction_refit,
    parse_construction_allowlist,
    parse_direction_refit_config,
    parse_direction_refit_receipt,
)


DOMAINS = tuple(f"domain_{index:02d}" for index in range(17))


def _signed(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    result["self_sha256"] = canonical_sha256(result)
    return result


def _config() -> dict:
    return _signed(
        {
            "format": "truth_editing_direction_refit_config_v1",
            "config_id": "test-refit",
            "model": {
                "repository": "Qwen/Qwen3-VL-8B-Thinking",
                "revision": "9" * 40,
                "model_sha256": "a" * 64,
                "decoder_layer_count": 36,
                "hidden_width": 2,
            },
            "activation": {
                "path": "artifacts/activations/all-text.h5",
                "byte_size": 123,
                "direct_sha256": "b" * 64,
                "dvc_md5": "c" * 32,
                "evidence_status": "verified_metadata",
                "sidecar_sha256": "d" * 64,
                "example_pooling": "arithmetic_mean_over_half_open_answer_token_span_v1",
            },
            "construction_allowlist": {
                "path": "datasets/direction-construction.json",
                "file_sha256": "e" * 64,
            },
            "domains": list(DOMAINS),
            "layers": list(range(36)),
            "fit": {
                "estimator": "sklearn.linear_model.LogisticRegression",
                "solver": "liblinear",
                "class_weight": "balanced",
                "regularization_c": 1.0,
                "max_iter": 1000,
                "random_seed": 0,
                "normalization": "unit_l2_with_intercept_rescaled",
                "sign_convention": "sklearn_logistic_coef_positive_points_honest_to_deceptive",
            },
            "output_root": "artifacts/directions/refit-v1",
            "source_code_revision": "f" * 40,
        }
    )


def _allowlist() -> dict:
    rows = []
    for domain_index, domain in enumerate(DOMAINS):
        for group_index in range(2):
            group_id = f"{domain}.group-{group_index}"
            for label in (0, 1):
                row_id = f"{domain}.g{group_index}.label{label}"
                rows.append(
                    {
                        "row_id": row_id,
                        "group_id": group_id,
                        "domain": domain,
                        "hdf5_task": domain,
                        "hdf5_row_index": group_index * 2 + label,
                        "label": label,
                        "selector": "direction_construction",
                    }
                )
    return _signed(
        {
            "format": "truth_editing_construction_row_allowlist_v1",
            "allowlist_id": "clean-construction",
            "activation_direct_sha256": "b" * 64,
            "dataset_manifest_sha256": "1" * 64,
            "construction_group_manifest_sha256": "2" * 64,
            "rows": rows,
        }
    )


def _bind_file_hash(config: dict, allowlist: dict) -> dict:
    result = copy.deepcopy(config)
    unsigned = copy.deepcopy(allowlist)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    result["construction_allowlist"]["file_sha256"] = hashlib.sha256(
        encoded
    ).hexdigest()
    result.pop("self_sha256")
    return _signed(result)


def test_plan_covers_every_domain_and_layer_with_resumable_layer_shards() -> None:
    raw_allowlist = _allowlist()
    config = parse_direction_refit_config(_bind_file_hash(_config(), raw_allowlist))
    allowlist = parse_construction_allowlist(raw_allowlist)

    initial = build_direction_refit_plan(config, allowlist)
    plan = build_direction_refit_plan(
        config, allowlist, completed_shards={0: initial.shards[0].shard_id}
    )

    assert plan.target_direction_count == 18 * 36
    assert plan.ordered_domains[-1] == "general_domain"
    assert len(plan.shards) == 36
    assert plan.shards[0].status == "complete"
    assert plan.shards[1].status == "pending"
    assert plan.shards[1].expected_direction_count == 18
    assert plan.to_dict()["self_sha256"] == plan.self_sha256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda raw: raw["rows"][0].update(selector="optimizer_validation"),
            "selector",
        ),
        (lambda raw: raw["rows"][0].update(group_id=""), "group_id"),
        (
            lambda raw: raw["rows"][1].update(
                hdf5_task=raw["rows"][0]["hdf5_task"],
                hdf5_row_index=raw["rows"][0]["hdf5_row_index"],
            ),
            "overlap",
        ),
        (lambda raw: raw["rows"][0].update(label=1), "single-class group"),
    ],
)
def test_allowlist_refuses_leakage_or_ambiguous_construction_rows(
    mutation, message
) -> None:
    raw = _allowlist()
    mutation(raw)
    raw.pop("self_sha256")
    raw = _signed(raw)
    with pytest.raises(DirectionRefitError, match=message):
        parse_construction_allowlist(raw)


def test_plan_refuses_unverified_or_mismatched_hdf5_identity() -> None:
    raw_allowlist = _allowlist()
    raw_config = _bind_file_hash(_config(), raw_allowlist)
    raw_config["activation"]["evidence_status"] = "unknown"
    raw_config.pop("self_sha256")
    with pytest.raises(DirectionRefitError, match="verified"):
        parse_direction_refit_config(_signed(raw_config))

    raw_config = _bind_file_hash(_config(), raw_allowlist)
    raw_config["activation"]["direct_sha256"] = "3" * 64
    raw_config.pop("self_sha256")
    config = parse_direction_refit_config(_signed(raw_config))
    with pytest.raises(DirectionRefitError, match="activation identity"):
        build_direction_refit_plan(config, parse_construction_allowlist(raw_allowlist))

    raw_config = _bind_file_hash(_config(), raw_allowlist)
    raw_config["activation"]["example_pooling"] = "last_token"
    raw_config.pop("self_sha256")
    with pytest.raises(DirectionRefitError, match="example_pooling"):
        parse_direction_refit_config(_signed(raw_config))


class _Reader:
    def __init__(self) -> None:
        self.layers: list[int] = []

    def read_layer(self, layer: int, rows) -> dict[str, np.ndarray]:
        self.layers.append(layer)
        return {
            row.row_id: np.asarray(
                [float(row.label), float(layer + 1)], dtype=np.float64
            )
            for row in rows
        }


class _Fitter:
    def fit(
        self, features: np.ndarray, labels: np.ndarray, fit
    ) -> tuple[np.ndarray, float, float]:
        assert set(labels.tolist()) == {0, 1}
        return np.asarray([4.0, 0.0]), 0.25, 4.0


class _Writer:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write_layer(self, relative_path: str, matrix: np.ndarray) -> tuple[str, str]:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix, allow_pickle=False)
        return relative_path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_execution_streams_each_layer_once_and_emits_exact_receipts(
    tmp_path: Path,
) -> None:
    raw_allowlist = _allowlist()
    config = parse_direction_refit_config(_bind_file_hash(_config(), raw_allowlist))
    allowlist = parse_construction_allowlist(raw_allowlist)
    plan = build_direction_refit_plan(config, allowlist)
    reader = _Reader()

    receipt = execute_direction_refit(
        plan,
        allowlist,
        reader=reader,
        fitter=_Fitter(),
        writer=_Writer(tmp_path),
    )

    assert reader.layers == list(range(36))
    assert receipt.completed_direction_count == 18 * 36
    assert len(receipt.layer_receipts) == 36
    first = receipt.layer_receipts[0]
    assert first.matrix_shape == (18, 2)
    assert first.matrix_dtype == "<f8"
    assert len(first.direction_receipts) == 18
    assert first.direction_receipts[-1].domain == "general_domain"
    assert first.direction_receipts[0].model_revision == "9" * 40
    assert first.direction_receipts[0].activation_direct_sha256 == "b" * 64
    assert (
        first.direction_receipts[0].construction_row_allowlist_sha256
        == allowlist.self_sha256
    )
    assert parse_direction_refit_receipt(receipt.to_dict()) == receipt

    tampered = receipt.to_dict()
    tampered["layer_receipts"][0]["direction_receipts"][0]["domain"] = "changed"
    with pytest.raises(DirectionRefitError, match="self hash"):
        parse_direction_refit_receipt(tampered)


def _write_hdf5_fixture(
    path: Path,
    *,
    splits: list[int],
    example_labels: list[int],
    token_labels: list[int],
    activations: list[list[float]],
) -> int:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "qwen_answer_token_activations_v2"
        metadata = handle.create_group("metadata/task")
        metadata.attrs["aggregation"] = "token_rows/no_pooling"
        metadata.create_dataset("example_splits", data=np.asarray(splits, dtype=np.int64))
        metadata.create_dataset(
            "example_labels", data=np.asarray(example_labels, dtype=np.int64)
        )
        # Deliberately disagree with example_labels at example-addressed indices.
        # A reader that indexes this token-level dataset by example index fails.
        metadata.create_dataset("labels", data=np.asarray(token_labels, dtype=np.int64))
        layer = handle.create_group("layer_0")
        layer.create_dataset("task", data=np.asarray(activations, dtype=np.float16))
    return path.stat().st_size


def _construction_row(row_id: str, index: int, label: int) -> object:
    from intelligent_liars.truth_editing_direction_refit import ConstructionRow

    return ConstructionRow(
        row_id=row_id,
        group_id="paired-group",
        domain="domain_00",
        hdf5_task="task",
        hdf5_row_index=index,
        label=label,
        selector="direction_construction",
    )


def test_hdf5_reader_binds_example_labels_and_mean_pools_each_exact_token_span(
    tmp_path: Path,
) -> None:
    path = tmp_path / "activations.h5"
    size = _write_hdf5_fixture(
        path,
        splits=[0, 2, 5, 6],
        example_labels=[0, 1, 0],
        token_labels=[1, 1, 0, 0, 0, 1],
        activations=[[1, 10], [3, 30], [100, 2], [200, 4], [300, 6], [9, 90]],
    )
    reader = Hdf5LayerReader(path, expected_byte_size=size, hidden_width=2)
    try:
        values = reader.read_layer(
            0,
            (
                _construction_row("first", 0, 0),
                _construction_row("middle", 1, 1),
                _construction_row("last", 2, 0),
            ),
        )
    finally:
        reader.close()

    np.testing.assert_array_equal(values["first"], np.asarray([2.0, 20.0]))
    np.testing.assert_array_equal(values["middle"], np.asarray([200.0, 4.0]))
    # Single-token spans have exact parity with their sole stored activation.
    np.testing.assert_array_equal(values["last"], np.asarray([9.0, 90.0]))


@pytest.mark.parametrize(
    ("splits", "message"),
    [
        ([0, 0, 1], "empty or inverted"),
        ([0, 2, 1], "empty or inverted"),
        ([0, 1, 3], "outside activation rows"),
    ],
)
def test_hdf5_reader_fails_closed_on_invalid_example_token_spans(
    tmp_path: Path, splits: list[int], message: str
) -> None:
    path = tmp_path / "invalid.h5"
    size = _write_hdf5_fixture(
        path,
        splits=splits,
        example_labels=[0, 1],
        token_labels=[0, 1],
        activations=[[1, 2], [3, 4]],
    )
    reader = Hdf5LayerReader(path, expected_byte_size=size, hidden_width=2)
    try:
        with pytest.raises(DirectionRefitError, match=message):
            reader.read_layer(0, (_construction_row("selected", 0, 0),))
    finally:
        reader.close()


def test_resume_never_reads_completed_layer() -> None:
    raw_allowlist = _allowlist()
    config = parse_direction_refit_config(_bind_file_hash(_config(), raw_allowlist))
    allowlist = parse_construction_allowlist(raw_allowlist)
    initial = build_direction_refit_plan(config, allowlist)
    resumed = build_direction_refit_plan(
        config, allowlist, completed_shards={0: initial.shards[0].shard_id}
    )
    reader = _Reader()

    # Fail before writing layer 1, after proving the completed layer was skipped.
    class StopFitter:
        def fit(self, features, labels, fit):
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        execute_direction_refit(
            resumed,
            allowlist,
            reader=reader,
            fitter=StopFitter(),
            writer=object(),  # type: ignore[arg-type]
        )
    assert reader.layers == [1]


@pytest.mark.parametrize("mode", ["nonfinite", "wrong_shape", "zero", "rank"])
def test_execution_refuses_invalid_vectors(tmp_path: Path, mode: str) -> None:
    raw_allowlist = _allowlist()
    raw_config = _bind_file_hash(_config(), raw_allowlist)
    if mode == "rank":
        raw_config["model"]["hidden_width"] = 1
        raw_config.pop("self_sha256")
        raw_config = _signed(raw_config)
    config = parse_direction_refit_config(raw_config)
    allowlist = parse_construction_allowlist(raw_allowlist)
    plan = build_direction_refit_plan(config, allowlist)

    class BadFitter:
        def fit(self, features, labels, fit):
            if mode == "nonfinite":
                return np.asarray([np.nan, 0.0]), 0.0, 1.0
            if mode == "wrong_shape":
                return np.asarray([1.0, 0.0, 0.0]), 0.0, 1.0
            if mode == "zero":
                return np.zeros(2), 0.0, 0.0
            return np.asarray([1.0]), 0.0, 1.0

    with pytest.raises(DirectionRefitError, match="nonfinite|shape|norm|rank"):
        execute_direction_refit(
            plan,
            allowlist,
            reader=_Reader(),
            fitter=BadFitter(),
            writer=_Writer(tmp_path),
        )
