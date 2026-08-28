"""Leakage-safe, resumable refitting of all-layer truth directions.

The public seam has three operations: parse frozen inputs, build a dry-run
plan, and execute that plan through injected layer-reader/fitter/writer
adapters.  The implementation never discovers rows from HDF5.  A signed
allowlist is the sole selector, and every pending layer is read exactly once.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np


CONFIG_FORMAT = "truth_editing_direction_refit_config_v1"
ALLOWLIST_FORMAT = "truth_editing_construction_row_allowlist_v1"
PLAN_FORMAT = "truth_editing_direction_refit_plan_v1"
RECEIPT_FORMAT = "truth_editing_direction_refit_receipt_v1"
LAYER_RECEIPT_FORMAT = "truth_editing_direction_refit_layer_receipt_v1"
DIRECTION_RECEIPT_FORMAT = "truth_editing_refit_direction_v1"
ALLOWED_SELECTOR = "direction_construction"
GENERAL_DOMAIN = "general_domain"
EXAMPLE_POOLING = "arithmetic_mean_over_half_open_answer_token_span_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MD5 = re.compile(r"^[0-9a-f]{32}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class DirectionRefitError(ValueError):
    """A direction-refit input or output fails a frozen invariant."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DirectionRefitError("value is not canonical finite JSON") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DirectionRefitError(f"{name} must be an object")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise DirectionRefitError(
            f"{name} fields differ; missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DirectionRefitError(f"{name} must be a nonempty trimmed string")
    return value


def _hash(value: Any, name: str, pattern: re.Pattern[str] = _SHA256) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise DirectionRefitError(f"{name} has an invalid hash")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DirectionRefitError(f"{name} must be an integer >= {minimum}")
    return value


def _float(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DirectionRefitError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise DirectionRefitError(
            f"{name} must be finite" + (" and positive" if positive else "")
        )
    return result


def _verify_self_hash(raw: Mapping[str, Any], name: str) -> str:
    claimed = _hash(raw.get("self_sha256"), f"{name}.self_sha256")
    unsigned = dict(raw)
    del unsigned["self_sha256"]
    if canonical_sha256(unsigned) != claimed:
        raise DirectionRefitError(f"{name} self hash mismatch")
    return claimed


@dataclass(frozen=True)
class RefitModelIdentity:
    repository: str
    revision: str
    model_sha256: str
    decoder_layer_count: int
    hidden_width: int


@dataclass(frozen=True)
class FrozenActivationIdentity:
    path: str
    byte_size: int
    direct_sha256: str
    dvc_md5: str
    evidence_status: Literal["verified_metadata", "proven"]
    sidecar_sha256: str
    example_pooling: Literal[
        "arithmetic_mean_over_half_open_answer_token_span_v1"
    ]


@dataclass(frozen=True)
class ConstructionAllowlistIdentity:
    path: str
    file_sha256: str


@dataclass(frozen=True)
class ProbeFitConfig:
    estimator: str
    solver: str
    class_weight: str
    regularization_c: float
    max_iter: int
    random_seed: int
    normalization: str
    sign_convention: str


@dataclass(frozen=True)
class DirectionRefitConfig:
    format: Literal["truth_editing_direction_refit_config_v1"]
    config_id: str
    model: RefitModelIdentity
    activation: FrozenActivationIdentity
    construction_allowlist: ConstructionAllowlistIdentity
    domains: tuple[str, ...]
    layers: tuple[int, ...]
    fit: ProbeFitConfig
    output_root: str
    source_code_revision: str
    self_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(asdict(self)))


@dataclass(frozen=True)
class ConstructionRow:
    row_id: str
    group_id: str
    domain: str
    hdf5_task: str
    hdf5_row_index: int
    label: Literal[0, 1]
    selector: Literal["direction_construction"]


@dataclass(frozen=True)
class ConstructionAllowlist:
    format: Literal["truth_editing_construction_row_allowlist_v1"]
    allowlist_id: str
    activation_direct_sha256: str
    dataset_manifest_sha256: str
    construction_group_manifest_sha256: str
    rows: tuple[ConstructionRow, ...]
    self_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(asdict(self)))

    @property
    def file_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


@dataclass(frozen=True)
class RefitShardPlan:
    shard_id: str
    source_layer: int
    relative_matrix_path: str
    expected_direction_count: int
    status: Literal["pending", "complete"]
    resume_receipt_sha256: str | None


@dataclass(frozen=True)
class DirectionRefitPlan:
    format: Literal["truth_editing_direction_refit_plan_v1"]
    config_sha256: str
    activation_direct_sha256: str
    construction_row_allowlist_sha256: str
    ordered_domains: tuple[str, ...]
    ordered_domains_sha256: str
    target_direction_count: int
    shards: tuple[RefitShardPlan, ...]
    self_sha256: str
    config: DirectionRefitConfig

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        del raw["config"]
        return json.loads(canonical_json_bytes(raw))


@dataclass(frozen=True)
class DirectionArtifactReceipt:
    format: Literal["truth_editing_refit_direction_v1"]
    direction_id: str
    domain: str
    source_layer: int
    width: int
    rank: int
    artifact_path: str
    artifact_file_sha256: str
    artifact_row: int
    vector_sha256: str
    original_norm: float
    rescaled_intercept: float
    sign_convention: str
    model_repository: str
    model_revision: str
    model_sha256: str
    activation_direct_sha256: str
    activation_dvc_md5: str
    activation_sidecar_sha256: str
    construction_selector: Literal["direction_construction"]
    construction_row_allowlist_sha256: str
    ordered_row_ids_sha256: str
    construction_group_manifest_sha256: str
    dataset_manifest_sha256: str
    source_code_revision: str
    probe_config_sha256: str
    reconstruction_plan_sha256: str
    shard_id: str
    finite: bool
    unit_norm: bool
    qualified_rank: int
    self_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(asdict(self)))


@dataclass(frozen=True)
class LayerRefitReceipt:
    format: Literal["truth_editing_direction_refit_layer_receipt_v1"]
    shard_id: str
    source_layer: int
    artifact_path: str
    artifact_file_sha256: str
    matrix_shape: tuple[int, int]
    matrix_dtype: str
    ordered_domains_sha256: str
    direction_receipts: tuple[DirectionArtifactReceipt, ...]
    self_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(asdict(self)))


@dataclass(frozen=True)
class DirectionRefitReceipt:
    format: Literal["truth_editing_direction_refit_receipt_v1"]
    plan_sha256: str
    completed_direction_count: int
    layer_receipts: tuple[LayerRefitReceipt, ...]
    self_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(asdict(self)))


class LayerActivationReader(Protocol):
    def read_layer(
        self, layer: int, rows: tuple[ConstructionRow, ...]
    ) -> Mapping[str, np.ndarray]: ...


class DirectionFitter(Protocol):
    def fit(
        self, features: np.ndarray, labels: np.ndarray, fit: ProbeFitConfig
    ) -> tuple[np.ndarray, float, float]: ...


class LayerMatrixWriter(Protocol):
    def write_layer(
        self, relative_path: str, matrix: np.ndarray
    ) -> tuple[str, str]: ...


class Hdf5LayerReader:
    """Read exact allowlisted examples and mean-pool their answer-token spans.

    ``hdf5_row_index`` is an example address, never a token-row address.  For
    example ``i``, the class comes from ``example_labels[i]`` and its feature is
    the arithmetic mean of activation rows in the half-open interval
    ``[example_splits[i], example_splits[i + 1])``.  The adapter validates the
    example-level metadata independently of the token-level ``labels`` dataset.
    """

    def __init__(self, path: Path, *, expected_byte_size: int, hidden_width: int) -> None:
        import h5py

        if path.stat().st_size != expected_byte_size:
            raise DirectionRefitError(
                "activation HDF5 byte size differs from frozen identity"
            )
        self._handle = h5py.File(path, "r")
        self._width = hidden_width
        if self._handle.attrs.get("format") != "qwen_answer_token_activations_v2":
            self._handle.close()
            raise DirectionRefitError("activation HDF5 format is not supported")

    def close(self) -> None:
        self._handle.close()

    def read_layer(
        self, layer: int, rows: tuple[ConstructionRow, ...]
    ) -> Mapping[str, np.ndarray]:
        selected: dict[str, list[ConstructionRow]] = {}
        for row in rows:
            selected.setdefault(row.hdf5_task, []).append(row)

        result: dict[str, np.ndarray] = {}
        for task, task_rows in sorted(selected.items()):
            metadata_path = f"metadata/{task}"
            dataset_path = f"layer_{layer}/{task}"
            if metadata_path not in self._handle or dataset_path not in self._handle:
                raise DirectionRefitError(
                    f"HDF5 is missing {task!r} for layer {layer}"
                )
            metadata = self._handle[metadata_path]
            if "example_splits" not in metadata or "example_labels" not in metadata:
                raise DirectionRefitError(
                    f"HDF5 example metadata is incomplete for {task!r}"
                )
            if metadata.attrs.get("aggregation") != "token_rows/no_pooling":
                raise DirectionRefitError(
                    f"HDF5 aggregation contract differs for {task!r}"
                )
            splits = np.asarray(metadata["example_splits"][:], dtype=np.int64)
            example_labels = np.asarray(
                metadata["example_labels"][:], dtype=np.int64
            )
            dataset = self._handle[dataset_path]
            if dataset.ndim != 2 or int(dataset.shape[1]) != self._width:
                raise DirectionRefitError(f"HDF5 shape mismatch for {dataset_path}")
            if splits.ndim != 1 or example_labels.ndim != 1:
                raise DirectionRefitError(
                    f"HDF5 example metadata shape mismatch for {task!r}"
                )
            if splits.shape[0] != example_labels.shape[0] + 1:
                raise DirectionRefitError(
                    f"HDF5 example split count mismatch for {task!r}"
                )
            if splits.shape[0] > 1 and np.any(np.diff(splits) <= 0):
                raise DirectionRefitError(
                    f"HDF5 contains an empty or inverted example span for {task!r}"
                )
            if splits.shape[0] == 0 or int(splits[-1]) != int(dataset.shape[0]):
                raise DirectionRefitError(
                    f"HDF5 example splits end outside activation rows for {task!r}"
                )

            spans: list[tuple[int, int, ConstructionRow]] = []
            for row in task_rows:
                index = row.hdf5_row_index
                if index < 0 or index >= example_labels.shape[0]:
                    raise DirectionRefitError(
                        f"allowlisted HDF5 example is out of range: {row.row_id}"
                    )
                if int(example_labels[index]) != row.label:
                    raise DirectionRefitError(
                        f"allowlisted label differs from HDF5 example label: {row.row_id}"
                    )
                start, stop = int(splits[index]), int(splits[index + 1])
                if start < 0 or start >= stop:
                    raise DirectionRefitError(
                        f"empty or inverted activation span for {row.row_id}"
                    )
                if stop > int(dataset.shape[0]):
                    raise DirectionRefitError(
                        f"activation span is outside activation rows for {row.row_id}"
                    )
                spans.append((start, stop, row))

            # Batch adjacent allowlisted examples into one HDF5 slice.  Blocks
            # never bridge a non-allowlisted example, and pooling still uses
            # each example's exact subspan.  Dense construction sets therefore
            # scan a task once per layer instead of issuing one read per row.
            ordered_spans = sorted(spans, key=lambda item: (item[0], item[1]))
            block: list[tuple[int, int, ConstructionRow]] = []
            for span in ordered_spans:
                if block and span[0] != block[-1][1]:
                    self._pool_span_block(dataset, block, result)
                    block = []
                block.append(span)
            if block:
                self._pool_span_block(dataset, block, result)
        return result

    @staticmethod
    def _pool_span_block(
        dataset: Any,
        spans: list[tuple[int, int, ConstructionRow]],
        result: dict[str, np.ndarray],
    ) -> None:
        block_start = spans[0][0]
        block_stop = spans[-1][1]
        activation_rows = np.asarray(
            dataset[block_start:block_stop], dtype=np.float64
        )
        for start, stop, row in spans:
            local_start, local_stop = start - block_start, stop - block_start
            result[row.row_id] = np.asarray(
                np.mean(
                    activation_rows[local_start:local_stop],
                    axis=0,
                    dtype=np.float64,
                ),
                dtype=np.float64,
            )


def parse_direction_refit_config(value: Any) -> DirectionRefitConfig:
    raw = _mapping(value, "config")
    _exact(
        raw,
        {
            "format",
            "config_id",
            "model",
            "activation",
            "construction_allowlist",
            "domains",
            "layers",
            "fit",
            "output_root",
            "source_code_revision",
            "self_sha256",
        },
        "config",
    )
    if raw["format"] != CONFIG_FORMAT:
        raise DirectionRefitError("unsupported config format")
    model_raw = _mapping(raw["model"], "config.model")
    _exact(
        model_raw,
        {
            "repository",
            "revision",
            "model_sha256",
            "decoder_layer_count",
            "hidden_width",
        },
        "config.model",
    )
    model = RefitModelIdentity(
        repository=_text(model_raw["repository"], "config.model.repository"),
        revision=_hash(model_raw["revision"], "config.model.revision", _REVISION),
        model_sha256=_hash(model_raw["model_sha256"], "config.model.model_sha256"),
        decoder_layer_count=_integer(
            model_raw["decoder_layer_count"],
            "config.model.decoder_layer_count",
            minimum=1,
        ),
        hidden_width=_integer(
            model_raw["hidden_width"], "config.model.hidden_width", minimum=1
        ),
    )
    activation_raw = _mapping(raw["activation"], "config.activation")
    _exact(
        activation_raw,
        {
            "path",
            "byte_size",
            "direct_sha256",
            "dvc_md5",
            "evidence_status",
            "sidecar_sha256",
            "example_pooling",
        },
        "config.activation",
    )
    evidence = activation_raw["evidence_status"]
    if evidence not in {"verified_metadata", "proven"}:
        raise DirectionRefitError("activation identity must be verified")
    activation = FrozenActivationIdentity(
        path=_safe_relative_path(activation_raw["path"], "config.activation.path"),
        byte_size=_integer(
            activation_raw["byte_size"], "config.activation.byte_size", minimum=1
        ),
        direct_sha256=_hash(
            activation_raw["direct_sha256"], "config.activation.direct_sha256"
        ),
        dvc_md5=_hash(activation_raw["dvc_md5"], "config.activation.dvc_md5", _MD5),
        evidence_status=evidence,
        sidecar_sha256=_hash(
            activation_raw["sidecar_sha256"], "config.activation.sidecar_sha256"
        ),
        example_pooling=cast(
            Literal["arithmetic_mean_over_half_open_answer_token_span_v1"],
            _required_example_pooling(activation_raw["example_pooling"]),
        ),
    )
    allow_raw = _mapping(raw["construction_allowlist"], "config.construction_allowlist")
    _exact(allow_raw, {"path", "file_sha256"}, "config.construction_allowlist")
    allow_identity = ConstructionAllowlistIdentity(
        path=_safe_relative_path(
            allow_raw["path"], "config.construction_allowlist.path"
        ),
        file_sha256=_hash(
            allow_raw["file_sha256"], "config.construction_allowlist.file_sha256"
        ),
    )
    domains = _unique_text_tuple(raw["domains"], "config.domains")
    if len(domains) != 17 or GENERAL_DOMAIN in domains:
        raise DirectionRefitError(
            "config.domains must contain exactly 17 specific domains"
        )
    layers = _unique_int_tuple(raw["layers"], "config.layers")
    if layers != tuple(range(model.decoder_layer_count)):
        raise DirectionRefitError(
            "config.layers must cover every decoder layer in order"
        )
    fit_raw = _mapping(raw["fit"], "config.fit")
    _exact(
        fit_raw,
        {
            "estimator",
            "solver",
            "class_weight",
            "regularization_c",
            "max_iter",
            "random_seed",
            "normalization",
            "sign_convention",
        },
        "config.fit",
    )
    expected_fit = {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "solver": "liblinear",
        "class_weight": "balanced",
        "normalization": "unit_l2_with_intercept_rescaled",
        "sign_convention": "sklearn_logistic_coef_positive_points_honest_to_deceptive",
    }
    for key, expected in expected_fit.items():
        if fit_raw[key] != expected:
            raise DirectionRefitError(f"config.fit.{key} must equal {expected!r}")
    fit = ProbeFitConfig(
        estimator=fit_raw["estimator"],
        solver=fit_raw["solver"],
        class_weight=fit_raw["class_weight"],
        regularization_c=_float(
            fit_raw["regularization_c"], "config.fit.regularization_c", positive=True
        ),
        max_iter=_integer(fit_raw["max_iter"], "config.fit.max_iter", minimum=1),
        random_seed=_integer(fit_raw["random_seed"], "config.fit.random_seed"),
        normalization=fit_raw["normalization"],
        sign_convention=fit_raw["sign_convention"],
    )
    return DirectionRefitConfig(
        format=cast(Literal["truth_editing_direction_refit_config_v1"], CONFIG_FORMAT),
        config_id=_text(raw["config_id"], "config.config_id"),
        model=model,
        activation=activation,
        construction_allowlist=allow_identity,
        domains=domains,
        layers=layers,
        fit=fit,
        output_root=_safe_relative_path(raw["output_root"], "config.output_root"),
        source_code_revision=_hash(
            raw["source_code_revision"], "config.source_code_revision", _REVISION
        ),
        self_sha256=_verify_self_hash(raw, "config"),
    )


def parse_construction_allowlist(value: Any) -> ConstructionAllowlist:
    raw = _mapping(value, "allowlist")
    _exact(
        raw,
        {
            "format",
            "allowlist_id",
            "activation_direct_sha256",
            "dataset_manifest_sha256",
            "construction_group_manifest_sha256",
            "rows",
            "self_sha256",
        },
        "allowlist",
    )
    if raw["format"] != ALLOWLIST_FORMAT:
        raise DirectionRefitError("unsupported allowlist format")
    row_values = raw["rows"]
    if (
        isinstance(row_values, (str, bytes))
        or not isinstance(row_values, Sequence)
        or not row_values
    ):
        raise DirectionRefitError("allowlist.rows must be a nonempty array")
    rows: list[ConstructionRow] = []
    seen_ids: set[str] = set()
    seen_addresses: set[tuple[str, int]] = set()
    labels_by_group: dict[str, set[int]] = {}
    for index, value_row in enumerate(row_values):
        row_raw = _mapping(value_row, f"allowlist.rows[{index}]")
        _exact(
            row_raw,
            {
                "row_id",
                "group_id",
                "domain",
                "hdf5_task",
                "hdf5_row_index",
                "label",
                "selector",
            },
            f"allowlist.rows[{index}]",
        )
        selector = row_raw["selector"]
        if selector != ALLOWED_SELECTOR:
            raise DirectionRefitError(
                f"allowlist selector must be {ALLOWED_SELECTOR}; validation/test/quarantine/judge/audit selectors are forbidden"
            )
        label = row_raw["label"]
        if isinstance(label, bool) or label not in {0, 1}:
            raise DirectionRefitError("allowlist row label must be 0 or 1")
        row = ConstructionRow(
            row_id=_text(row_raw["row_id"], f"allowlist.rows[{index}].row_id"),
            group_id=_text(row_raw["group_id"], f"allowlist.rows[{index}].group_id"),
            domain=_text(row_raw["domain"], f"allowlist.rows[{index}].domain"),
            hdf5_task=_text(row_raw["hdf5_task"], f"allowlist.rows[{index}].hdf5_task"),
            hdf5_row_index=_integer(
                row_raw["hdf5_row_index"], f"allowlist.rows[{index}].hdf5_row_index"
            ),
            label=label,
            selector=cast(Literal["direction_construction"], ALLOWED_SELECTOR),
        )
        if (
            row.row_id in seen_ids
            or (row.hdf5_task, row.hdf5_row_index) in seen_addresses
        ):
            raise DirectionRefitError("allowlist row overlap is forbidden")
        seen_ids.add(row.row_id)
        seen_addresses.add((row.hdf5_task, row.hdf5_row_index))
        labels_by_group.setdefault(row.group_id, set()).add(row.label)
        rows.append(row)
    if any(labels != {0, 1} for labels in labels_by_group.values()):
        raise DirectionRefitError("single-class groups are forbidden")
    return ConstructionAllowlist(
        format=cast(
            Literal["truth_editing_construction_row_allowlist_v1"], ALLOWLIST_FORMAT
        ),
        allowlist_id=_text(raw["allowlist_id"], "allowlist.allowlist_id"),
        activation_direct_sha256=_hash(
            raw["activation_direct_sha256"], "allowlist.activation_direct_sha256"
        ),
        dataset_manifest_sha256=_hash(
            raw["dataset_manifest_sha256"], "allowlist.dataset_manifest_sha256"
        ),
        construction_group_manifest_sha256=_hash(
            raw["construction_group_manifest_sha256"],
            "allowlist.construction_group_manifest_sha256",
        ),
        rows=tuple(rows),
        self_sha256=_verify_self_hash(raw, "allowlist"),
    )


def build_direction_refit_plan(
    config: DirectionRefitConfig,
    allowlist: ConstructionAllowlist,
    *,
    completed_shards: Mapping[int, str] | None = None,
) -> DirectionRefitPlan:
    if allowlist.activation_direct_sha256 != config.activation.direct_sha256:
        raise DirectionRefitError(
            "allowlist activation identity does not match frozen HDF5"
        )
    if allowlist.file_sha256 != config.construction_allowlist.file_sha256:
        raise DirectionRefitError("construction allowlist file identity mismatch")
    observed_domains = {row.domain for row in allowlist.rows}
    if observed_domains != set(config.domains):
        raise DirectionRefitError(
            "allowlist domains do not exactly match configured domains"
        )
    for domain in config.domains:
        labels = {row.label for row in allowlist.rows if row.domain == domain}
        if labels != {0, 1}:
            raise DirectionRefitError(f"domain {domain!r} is single-class")
    ordered_domains = (*config.domains, GENERAL_DOMAIN)
    ordered_domains_sha = canonical_sha256(list(ordered_domains))
    completed = completed_shards or {}
    unknown_layers = set(completed) - set(config.layers)
    if unknown_layers:
        raise DirectionRefitError(
            f"completed shard layers are unknown: {sorted(unknown_layers)}"
        )
    shards: list[RefitShardPlan] = []
    for layer in config.layers:
        unsigned = {
            "format": "truth_editing_direction_refit_shard_v1",
            "config_sha256": config.self_sha256,
            "allowlist_sha256": allowlist.self_sha256,
            "ordered_domains_sha256": ordered_domains_sha,
            "source_layer": layer,
        }
        shard_id = canonical_sha256(unsigned)
        marker = completed.get(layer)
        if marker is not None and marker != shard_id:
            raise DirectionRefitError(
                f"completed shard receipt mismatch for layer {layer}"
            )
        shards.append(
            RefitShardPlan(
                shard_id=shard_id,
                source_layer=layer,
                relative_matrix_path=f"{config.output_root}/layer-{layer:02d}.npy",
                expected_direction_count=len(ordered_domains),
                status="complete" if marker is not None else "pending",
                resume_receipt_sha256=None if marker is None else shard_id,
            )
        )
    unsigned_plan = {
        "format": PLAN_FORMAT,
        "config_sha256": config.self_sha256,
        "activation_direct_sha256": config.activation.direct_sha256,
        "construction_row_allowlist_sha256": allowlist.self_sha256,
        "ordered_domains": list(ordered_domains),
        "ordered_domains_sha256": ordered_domains_sha,
        "target_direction_count": len(ordered_domains) * len(config.layers),
        "shards": [asdict(shard) for shard in shards],
    }
    return DirectionRefitPlan(
        format=cast(Literal["truth_editing_direction_refit_plan_v1"], PLAN_FORMAT),
        config_sha256=config.self_sha256,
        activation_direct_sha256=config.activation.direct_sha256,
        construction_row_allowlist_sha256=allowlist.self_sha256,
        ordered_domains=ordered_domains,
        ordered_domains_sha256=ordered_domains_sha,
        target_direction_count=len(ordered_domains) * len(config.layers),
        shards=tuple(shards),
        self_sha256=canonical_sha256(unsigned_plan),
        config=config,
    )


def execute_direction_refit(
    plan: DirectionRefitPlan,
    allowlist: ConstructionAllowlist,
    *,
    reader: LayerActivationReader,
    fitter: DirectionFitter,
    writer: LayerMatrixWriter,
) -> DirectionRefitReceipt:
    if plan.construction_row_allowlist_sha256 != allowlist.self_sha256:
        raise DirectionRefitError("execution allowlist identity differs from plan")
    config = plan.config
    layer_receipts: list[LayerRefitReceipt] = []
    rows = allowlist.rows
    for shard in plan.shards:
        if shard.status == "complete":
            continue
        values = reader.read_layer(shard.source_layer, rows)
        if set(values) != {row.row_id for row in rows}:
            raise DirectionRefitError(
                f"layer {shard.source_layer} row identity coverage mismatch"
            )
        vectors: list[np.ndarray] = []
        fit_outputs: list[tuple[str, tuple[ConstructionRow, ...], float, float]] = []
        for domain in plan.ordered_domains:
            domain_rows = (
                rows
                if domain == GENERAL_DOMAIN
                else tuple(row for row in rows if row.domain == domain)
            )
            features = np.stack(
                [
                    np.asarray(values[row.row_id], dtype=np.float64)
                    for row in domain_rows
                ]
            )
            labels = np.asarray([row.label for row in domain_rows], dtype=np.int64)
            _validate_feature_matrix(
                features, labels, config.model.hidden_width, domain
            )
            vector, intercept, original_norm = fitter.fit(features, labels, config.fit)
            normalized, rescaled_intercept, norm = _normalize_direction(
                vector, intercept, original_norm, config.model.hidden_width
            )
            vectors.append(normalized)
            fit_outputs.append((domain, domain_rows, rescaled_intercept, norm))
        matrix = np.ascontiguousarray(np.stack(vectors), dtype="<f8")
        if matrix.shape != (len(plan.ordered_domains), config.model.hidden_width):
            raise DirectionRefitError("layer matrix shape mismatch")
        artifact_path, file_sha = writer.write_layer(shard.relative_matrix_path, matrix)
        _hash(file_sha, "artifact file sha256")
        direction_receipts: list[DirectionArtifactReceipt] = []
        for row_index, (
            domain,
            domain_rows,
            rescaled_intercept,
            original_norm,
        ) in enumerate(fit_outputs):
            canonical_vector = np.ascontiguousarray(matrix[row_index], dtype="<f8")
            vector_prefix = canonical_json_bytes(
                {
                    "format": "truth_editing_vector_f64le_v1",
                    "shape": list(canonical_vector.shape),
                }
            )
            vector_sha = hashlib.sha256(
                vector_prefix + b"\0" + canonical_vector.tobytes(order="C")
            ).hexdigest()
            unsigned = {
                "format": DIRECTION_RECEIPT_FORMAT,
                "direction_id": f"refit-v1-{_slug(domain)}-layer-{shard.source_layer:02d}",
                "domain": domain,
                "source_layer": shard.source_layer,
                "width": config.model.hidden_width,
                "rank": 1,
                "artifact_path": f"{artifact_path}#row/{row_index}",
                "artifact_file_sha256": file_sha,
                "artifact_row": row_index,
                "vector_sha256": vector_sha,
                "original_norm": original_norm,
                "rescaled_intercept": rescaled_intercept,
                "sign_convention": config.fit.sign_convention,
                "model_repository": config.model.repository,
                "model_revision": config.model.revision,
                "model_sha256": config.model.model_sha256,
                "activation_direct_sha256": config.activation.direct_sha256,
                "activation_dvc_md5": config.activation.dvc_md5,
                "activation_sidecar_sha256": config.activation.sidecar_sha256,
                "construction_selector": ALLOWED_SELECTOR,
                "construction_row_allowlist_sha256": allowlist.self_sha256,
                "ordered_row_ids_sha256": canonical_sha256(
                    [row.row_id for row in domain_rows]
                ),
                "construction_group_manifest_sha256": allowlist.construction_group_manifest_sha256,
                "dataset_manifest_sha256": allowlist.dataset_manifest_sha256,
                "source_code_revision": config.source_code_revision,
                "probe_config_sha256": canonical_sha256(asdict(config.fit)),
                "reconstruction_plan_sha256": plan.self_sha256,
                "shard_id": shard.shard_id,
                "finite": True,
                "unit_norm": True,
                "qualified_rank": 1,
            }
            direction_receipts.append(
                DirectionArtifactReceipt(
                    format=cast(
                        Literal["truth_editing_refit_direction_v1"],
                        DIRECTION_RECEIPT_FORMAT,
                    ),
                    direction_id=cast(str, unsigned["direction_id"]),
                    domain=domain,
                    source_layer=shard.source_layer,
                    width=config.model.hidden_width,
                    rank=1,
                    artifact_path=cast(str, unsigned["artifact_path"]),
                    artifact_file_sha256=file_sha,
                    artifact_row=row_index,
                    vector_sha256=vector_sha,
                    original_norm=original_norm,
                    rescaled_intercept=rescaled_intercept,
                    sign_convention=config.fit.sign_convention,
                    model_repository=config.model.repository,
                    model_revision=config.model.revision,
                    model_sha256=config.model.model_sha256,
                    activation_direct_sha256=config.activation.direct_sha256,
                    activation_dvc_md5=config.activation.dvc_md5,
                    activation_sidecar_sha256=config.activation.sidecar_sha256,
                    construction_selector="direction_construction",
                    construction_row_allowlist_sha256=allowlist.self_sha256,
                    ordered_row_ids_sha256=cast(
                        str, unsigned["ordered_row_ids_sha256"]
                    ),
                    construction_group_manifest_sha256=allowlist.construction_group_manifest_sha256,
                    dataset_manifest_sha256=allowlist.dataset_manifest_sha256,
                    source_code_revision=config.source_code_revision,
                    probe_config_sha256=canonical_sha256(asdict(config.fit)),
                    reconstruction_plan_sha256=plan.self_sha256,
                    shard_id=shard.shard_id,
                    finite=True,
                    unit_norm=True,
                    qualified_rank=1,
                    self_sha256=canonical_sha256(unsigned),
                )
            )
        unsigned_layer = {
            "format": LAYER_RECEIPT_FORMAT,
            "shard_id": shard.shard_id,
            "source_layer": shard.source_layer,
            "artifact_path": artifact_path,
            "artifact_file_sha256": file_sha,
            "matrix_shape": list(matrix.shape),
            "matrix_dtype": matrix.dtype.str,
            "ordered_domains_sha256": plan.ordered_domains_sha256,
            "direction_receipts": [asdict(receipt) for receipt in direction_receipts],
        }
        layer_receipts.append(
            LayerRefitReceipt(
                format=cast(
                    Literal["truth_editing_direction_refit_layer_receipt_v1"],
                    LAYER_RECEIPT_FORMAT,
                ),
                shard_id=shard.shard_id,
                source_layer=shard.source_layer,
                artifact_path=artifact_path,
                artifact_file_sha256=file_sha,
                matrix_shape=matrix.shape,
                matrix_dtype=matrix.dtype.str,
                ordered_domains_sha256=plan.ordered_domains_sha256,
                direction_receipts=tuple(direction_receipts),
                self_sha256=canonical_sha256(unsigned_layer),
            )
        )
    unsigned_receipt = {
        "format": RECEIPT_FORMAT,
        "plan_sha256": plan.self_sha256,
        "completed_direction_count": sum(
            len(item.direction_receipts) for item in layer_receipts
        ),
        "layer_receipts": [asdict(item) for item in layer_receipts],
    }
    completed_count = sum(len(item.direction_receipts) for item in layer_receipts)
    return DirectionRefitReceipt(
        format=cast(
            Literal["truth_editing_direction_refit_receipt_v1"], RECEIPT_FORMAT
        ),
        plan_sha256=plan.self_sha256,
        completed_direction_count=completed_count,
        layer_receipts=tuple(layer_receipts),
        self_sha256=canonical_sha256(unsigned_receipt),
    )


def parse_direction_refit_receipt(value: Any) -> DirectionRefitReceipt:
    """Rehydrate and recursively verify a persisted aggregate receipt."""

    raw = _mapping(value, "receipt")
    _exact(
        raw,
        {
            "format",
            "plan_sha256",
            "completed_direction_count",
            "layer_receipts",
            "self_sha256",
        },
        "receipt",
    )
    if raw["format"] != RECEIPT_FORMAT:
        raise DirectionRefitError("unsupported direction refit receipt format")
    layer_values = raw["layer_receipts"]
    if isinstance(layer_values, (str, bytes)) or not isinstance(layer_values, Sequence):
        raise DirectionRefitError("receipt.layer_receipts must be an array")
    layers = tuple(
        _parse_layer_receipt(item, index) for index, item in enumerate(layer_values)
    )
    layer_indices = tuple(item.source_layer for item in layers)
    if layer_indices != tuple(sorted(set(layer_indices))):
        raise DirectionRefitError("receipt layer indices must be sorted and unique")
    completed = _integer(
        raw["completed_direction_count"], "receipt.completed_direction_count"
    )
    if completed != sum(len(item.direction_receipts) for item in layers):
        raise DirectionRefitError("receipt completed direction count mismatch")
    return DirectionRefitReceipt(
        format="truth_editing_direction_refit_receipt_v1",
        plan_sha256=_hash(raw["plan_sha256"], "receipt.plan_sha256"),
        completed_direction_count=completed,
        layer_receipts=layers,
        self_sha256=_verify_self_hash(raw, "receipt"),
    )


def _parse_layer_receipt(value: Any, index: int) -> LayerRefitReceipt:
    name = f"receipt.layer_receipts[{index}]"
    raw = _mapping(value, name)
    _exact(
        raw,
        {
            "format",
            "shard_id",
            "source_layer",
            "artifact_path",
            "artifact_file_sha256",
            "matrix_shape",
            "matrix_dtype",
            "ordered_domains_sha256",
            "direction_receipts",
            "self_sha256",
        },
        name,
    )
    if raw["format"] != LAYER_RECEIPT_FORMAT:
        raise DirectionRefitError(f"{name}.format is unsupported")
    shape_raw = raw["matrix_shape"]
    if (
        isinstance(shape_raw, (str, bytes))
        or not isinstance(shape_raw, Sequence)
        or len(shape_raw) != 2
    ):
        raise DirectionRefitError(f"{name}.matrix_shape must have two dimensions")
    shape = (
        _integer(shape_raw[0], f"{name}.matrix_shape[0]", minimum=1),
        _integer(shape_raw[1], f"{name}.matrix_shape[1]", minimum=1),
    )
    cells_raw = raw["direction_receipts"]
    if isinstance(cells_raw, (str, bytes)) or not isinstance(cells_raw, Sequence):
        raise DirectionRefitError(f"{name}.direction_receipts must be an array")
    cells = tuple(
        _parse_direction_receipt(item, name, cell_index)
        for cell_index, item in enumerate(cells_raw)
    )
    layer = _integer(raw["source_layer"], f"{name}.source_layer")
    artifact_path = _text(raw["artifact_path"], f"{name}.artifact_path")
    artifact_sha = _hash(raw["artifact_file_sha256"], f"{name}.artifact_file_sha256")
    if len(cells) != shape[0]:
        raise DirectionRefitError(f"{name} row count differs from direction receipts")
    for row_index, cell in enumerate(cells):
        if (
            cell.source_layer != layer
            or cell.artifact_file_sha256 != artifact_sha
            or cell.artifact_row != row_index
            or cell.artifact_path != f"{artifact_path}#row/{row_index}"
            or cell.width != shape[1]
        ):
            raise DirectionRefitError(f"{name} cell does not bind its layer matrix")
    matrix_dtype = _text(raw["matrix_dtype"], f"{name}.matrix_dtype")
    if matrix_dtype != "<f8":
        raise DirectionRefitError(f"{name}.matrix_dtype must be <f8")
    return LayerRefitReceipt(
        format="truth_editing_direction_refit_layer_receipt_v1",
        shard_id=_hash(raw["shard_id"], f"{name}.shard_id"),
        source_layer=layer,
        artifact_path=artifact_path,
        artifact_file_sha256=artifact_sha,
        matrix_shape=shape,
        matrix_dtype=matrix_dtype,
        ordered_domains_sha256=_hash(
            raw["ordered_domains_sha256"], f"{name}.ordered_domains_sha256"
        ),
        direction_receipts=cells,
        self_sha256=_verify_self_hash(raw, name),
    )


def _parse_direction_receipt(
    value: Any, parent: str, index: int
) -> DirectionArtifactReceipt:
    name = f"{parent}.direction_receipts[{index}]"
    raw = _mapping(value, name)
    fields = set(DirectionArtifactReceipt.__dataclass_fields__)
    _exact(raw, fields, name)
    if raw["format"] != DIRECTION_RECEIPT_FORMAT:
        raise DirectionRefitError(f"{name}.format is unsupported")
    finite = raw["finite"]
    unit_norm = raw["unit_norm"]
    if finite is not True or unit_norm is not True:
        raise DirectionRefitError(f"{name} must attest finite unit-norm output")
    rank = _integer(raw["rank"], f"{name}.rank", minimum=1)
    qualified_rank = _integer(
        raw["qualified_rank"], f"{name}.qualified_rank", minimum=1
    )
    if rank != 1 or qualified_rank != 1:
        raise DirectionRefitError(f"{name} rank must equal one")
    if raw["construction_selector"] != ALLOWED_SELECTOR:
        raise DirectionRefitError(f"{name}.construction_selector is forbidden")
    original_norm = _float(raw["original_norm"], f"{name}.original_norm", positive=True)
    return DirectionArtifactReceipt(
        format="truth_editing_refit_direction_v1",
        direction_id=_text(raw["direction_id"], f"{name}.direction_id"),
        domain=_text(raw["domain"], f"{name}.domain"),
        source_layer=_integer(raw["source_layer"], f"{name}.source_layer"),
        width=_integer(raw["width"], f"{name}.width", minimum=1),
        rank=rank,
        artifact_path=_text(raw["artifact_path"], f"{name}.artifact_path"),
        artifact_file_sha256=_hash(
            raw["artifact_file_sha256"], f"{name}.artifact_file_sha256"
        ),
        artifact_row=_integer(raw["artifact_row"], f"{name}.artifact_row"),
        vector_sha256=_hash(raw["vector_sha256"], f"{name}.vector_sha256"),
        original_norm=original_norm,
        rescaled_intercept=_float(
            raw["rescaled_intercept"], f"{name}.rescaled_intercept"
        ),
        sign_convention=_text(raw["sign_convention"], f"{name}.sign_convention"),
        model_repository=_text(raw["model_repository"], f"{name}.model_repository"),
        model_revision=_hash(
            raw["model_revision"], f"{name}.model_revision", _REVISION
        ),
        model_sha256=_hash(raw["model_sha256"], f"{name}.model_sha256"),
        activation_direct_sha256=_hash(
            raw["activation_direct_sha256"], f"{name}.activation_direct_sha256"
        ),
        activation_dvc_md5=_hash(
            raw["activation_dvc_md5"], f"{name}.activation_dvc_md5", _MD5
        ),
        activation_sidecar_sha256=_hash(
            raw["activation_sidecar_sha256"], f"{name}.activation_sidecar_sha256"
        ),
        construction_selector=cast(
            Literal["direction_construction"],
            _text(raw["construction_selector"], f"{name}.construction_selector"),
        ),
        construction_row_allowlist_sha256=_hash(
            raw["construction_row_allowlist_sha256"],
            f"{name}.construction_row_allowlist_sha256",
        ),
        ordered_row_ids_sha256=_hash(
            raw["ordered_row_ids_sha256"], f"{name}.ordered_row_ids_sha256"
        ),
        construction_group_manifest_sha256=_hash(
            raw["construction_group_manifest_sha256"],
            f"{name}.construction_group_manifest_sha256",
        ),
        dataset_manifest_sha256=_hash(
            raw["dataset_manifest_sha256"], f"{name}.dataset_manifest_sha256"
        ),
        source_code_revision=_hash(
            raw["source_code_revision"], f"{name}.source_code_revision", _REVISION
        ),
        probe_config_sha256=_hash(
            raw["probe_config_sha256"], f"{name}.probe_config_sha256"
        ),
        reconstruction_plan_sha256=_hash(
            raw["reconstruction_plan_sha256"], f"{name}.reconstruction_plan_sha256"
        ),
        shard_id=_hash(raw["shard_id"], f"{name}.shard_id"),
        finite=True,
        unit_norm=True,
        qualified_rank=qualified_rank,
        self_sha256=_verify_self_hash(raw, name),
    )


class SklearnLogisticDirectionFitter:
    """Production CPU adapter; imported lazily so dry runs need no sklearn."""

    def fit(
        self, features: np.ndarray, labels: np.ndarray, fit: ProbeFitConfig
    ) -> tuple[np.ndarray, float, float]:
        from sklearn.linear_model import LogisticRegression

        classifier = LogisticRegression(
            C=fit.regularization_c,
            class_weight=fit.class_weight,
            max_iter=fit.max_iter,
            random_state=fit.random_seed,
            solver=fit.solver,
        )
        classifier.fit(features, labels)
        vector = np.asarray(classifier.coef_[0], dtype=np.float64)
        return vector, float(classifier.intercept_[0]), float(np.linalg.norm(vector))


def _validate_feature_matrix(
    features: np.ndarray, labels: np.ndarray, width: int, domain: str
) -> None:
    if features.ndim != 2 or features.shape != (labels.shape[0], width):
        raise DirectionRefitError(f"{domain} feature shape mismatch")
    if features.shape[0] < 2 or np.unique(labels).tolist() != [0, 1]:
        raise DirectionRefitError(f"{domain} is single-class")
    if not np.isfinite(features).all():
        raise DirectionRefitError(f"{domain} has nonfinite features")
    if np.linalg.matrix_rank(features) < 1:
        raise DirectionRefitError(f"{domain} feature rank is zero")


def _normalize_direction(
    vector: np.ndarray, intercept: float, claimed_norm: float, width: int
) -> tuple[np.ndarray, float, float]:
    raw = np.asarray(vector, dtype=np.float64)
    if raw.shape != (width,):
        raise DirectionRefitError("direction vector shape mismatch")
    if (
        not np.isfinite(raw).all()
        or not math.isfinite(intercept)
        or not math.isfinite(claimed_norm)
    ):
        raise DirectionRefitError("direction output contains nonfinite values")
    norm = float(np.linalg.norm(raw))
    if (
        norm <= 0.0
        or claimed_norm <= 0.0
        or not math.isclose(norm, claimed_norm, rel_tol=1e-10, abs_tol=1e-12)
    ):
        raise DirectionRefitError("direction norm receipt mismatch or zero norm")
    normalized = np.ascontiguousarray(raw / norm, dtype="<f8")
    if np.linalg.matrix_rank(normalized.reshape(1, -1)) != 1:
        raise DirectionRefitError("direction rank must equal one")
    return normalized, float(intercept / norm), norm


def _safe_relative_path(value: Any, name: str) -> str:
    result = _text(value, name)
    parts = result.split("/")
    if result.startswith("/") or ".." in parts or "." in parts:
        raise DirectionRefitError(f"{name} must be a contained relative path")
    return result


def _unique_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DirectionRefitError(f"{name} must be an array")
    result = tuple(_text(item, f"{name}[]") for item in value)
    if not result or len(result) != len(set(result)):
        raise DirectionRefitError(f"{name} must be nonempty and unique")
    return result


def _unique_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DirectionRefitError(f"{name} must be an array")
    result = tuple(_integer(item, f"{name}[]") for item in value)
    if not result or len(result) != len(set(result)):
        raise DirectionRefitError(f"{name} must be nonempty and unique")
    return result


def _required_example_pooling(value: Any) -> str:
    if value != EXAMPLE_POOLING:
        raise DirectionRefitError(
            f"config.activation.example_pooling must equal {EXAMPLE_POOLING!r}"
        )
    return EXAMPLE_POOLING


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


__all__ = [
    "ConstructionAllowlist",
    "DirectionArtifactReceipt",
    "DirectionRefitConfig",
    "DirectionRefitError",
    "DirectionRefitPlan",
    "DirectionRefitReceipt",
    "EXAMPLE_POOLING",
    "Hdf5LayerReader",
    "LayerRefitReceipt",
    "SklearnLogisticDirectionFitter",
    "build_direction_refit_plan",
    "canonical_json_bytes",
    "canonical_sha256",
    "execute_direction_refit",
    "parse_construction_allowlist",
    "parse_direction_refit_config",
    "parse_direction_refit_receipt",
]
