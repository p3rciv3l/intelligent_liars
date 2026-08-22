"""Fit source-disjoint, grouped probes from the legacy labeled activation cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from intelligent_liars.probes import DIRECTION_SIGN_CONVENTION


FIT_CONFIG_FORMAT = "intelligent_liars_step5_grouped_probe_fit_config_v1"
FIT_REPORT_FORMAT = "intelligent_liars_step5_grouped_probe_fit_report_v1"
PROBE_ARTIFACT_FORMAT = "intelligent_liars_grouped_linear_probe_v1"
IDENTITY_FORMAT = "intelligent_liars_legacy_probe_identity_registry_v1"
LABEL_CONVENTION = "HONEST=0, DECEPTIVE=1"


def _identifier_component(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "-" for character in value)


def _stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _portable_path(path: str | Path, *, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _dvc_pointer_identity(path: Path) -> dict[str, Any] | None:
    pointer = path.with_name(f"{path.name}.dvc")
    if not pointer.is_file():
        return None
    import yaml

    payload = yaml.safe_load(pointer.read_text())
    output = payload["outs"][0]
    if output.get("path") != path.name or output.get("hash") != "md5":
        raise ValueError(f"Unexpected DVC pointer contract for {path}")
    if int(output["size"]) != path.stat().st_size:
        raise ValueError(f"DVC pointer size does not match local artifact: {path}")
    return {
        "pointer_path": str(pointer.resolve()),
        "pointer_sha256": hashlib.sha256(pointer.read_bytes()).hexdigest(),
        "content_md5": str(output["md5"]),
        "size_bytes": int(output["size"]),
    }


def _decode_json(value: str | bytes | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return decoded if isinstance(decoded, Mapping) else {}


def _nested_source_index(metadata: Mapping[str, Any]) -> int | str | None:
    current: Any = metadata
    for _ in range(3):
        if not isinstance(current, Mapping):
            return None
        value = current.get("source_index")
        if isinstance(value, (int, str)) and not isinstance(value, bool):
            return value
        current = current.get("source_metadata")
    return None


@dataclass(frozen=True)
class LegacyExampleIdentity:
    example_id: str
    task_id: str
    source_group_id: str
    example_group_id: str
    template_group_id: str


def derive_legacy_example_identity(
    *,
    task: str,
    source_index: int,
    output_index: int,
    raw_metadata_json: str | bytes | None,
    source_dataset: str | bytes | None,
) -> LegacyExampleIdentity:
    """Derive stable IDs without using labels or model activations.

    The task is the outer source boundary.  Source indices bind paired outputs;
    explicit upstream pair/template identifiers take precedence when available.
    """

    task_id = _identifier_component(task)
    metadata = _decode_json(raw_metadata_json)
    if isinstance(source_dataset, bytes):
        source_dataset = source_dataset.decode("utf-8")
    if source_dataset:
        source_parts = Path(source_dataset).parts
        data_positions = [i for i, part in enumerate(source_parts) if part == "data"]
        source_name = "/".join(
            source_parts[data_positions[-1] + 1 :]
            if data_positions
            else source_parts[-2:]
        )
    else:
        source_name = task
    source_group_id = f"legacy/source/{_identifier_component(source_name)}"
    csv_name = metadata.get("csv")
    row = metadata.get("row")
    pair_index = metadata.get("pair_index")
    nested_source = _nested_source_index(metadata)
    if isinstance(csv_name, str) and isinstance(row, int) and not isinstance(row, bool):
        example_suffix = f"row:{row}"
    elif isinstance(pair_index, (int, str)) and not isinstance(pair_index, bool):
        example_suffix = f"pair:{_identifier_component(str(pair_index))}"
    elif nested_source is not None:
        example_suffix = f"source:{_identifier_component(str(nested_source))}"
    else:
        example_suffix = f"source:{source_index}"
    example_group_id = f"{source_group_id}/{example_suffix}"
    example_id = f"{example_group_id}/task:{task_id}/output:{output_index}"
    if isinstance(pair_index, (int, str)) and not isinstance(pair_index, bool):
        template_suffix = f"pair:{_identifier_component(str(pair_index))}"
    elif nested_source is not None:
        template_suffix = f"source:{_identifier_component(str(nested_source))}"
    else:
        template_suffix = f"source:{source_index}"
    return LegacyExampleIdentity(
        example_id=example_id,
        task_id=task_id,
        source_group_id=source_group_id,
        example_group_id=example_group_id,
        template_group_id=f"{source_group_id}/{template_suffix}",
    )


@dataclass(frozen=True)
class OuterSplit:
    regularizer_indices: np.ndarray
    evaluator_indices: np.ndarray
    regularizer_source_groups: tuple[str, ...]
    evaluator_source_groups: tuple[str, ...]


def build_outer_split(
    identities: Sequence[LegacyExampleIdentity],
    *,
    evaluator_task_ids: set[str],
) -> OuterSplit:
    available_tasks = {identity.task_id for identity in identities}
    missing = sorted(evaluator_task_ids - available_tasks)
    if missing:
        raise ValueError(f"Configured evaluator source groups are absent: {missing}")
    regularizer = np.asarray(
        [i for i, identity in enumerate(identities) if identity.task_id not in evaluator_task_ids],
        dtype=np.int64,
    )
    evaluator = np.asarray(
        [i for i, identity in enumerate(identities) if identity.task_id in evaluator_task_ids],
        dtype=np.int64,
    )
    if not len(regularizer) or not len(evaluator):
        raise ValueError("Outer split must contain both regularizer and evaluator examples")
    regularizer_groups = {identities[int(i)].source_group_id for i in regularizer}
    evaluator_groups = {identities[int(i)].source_group_id for i in evaluator}
    for field in ("source_group_id", "example_group_id", "template_group_id"):
        regularizer_ids = {getattr(identities[int(i)], field) for i in regularizer}
        evaluator_ids = {getattr(identities[int(i)], field) for i in evaluator}
        overlap = regularizer_ids & evaluator_ids
        if overlap:
            raise ValueError(
                f"cross-ensemble {field} overlap ({len(overlap)} identities)"
            )
    return OuterSplit(
        regularizer_indices=regularizer,
        evaluator_indices=evaluator,
        regularizer_source_groups=tuple(sorted(regularizer_groups)),
        evaluator_source_groups=tuple(sorted(evaluator_groups)),
    )


@dataclass(frozen=True)
class CrossfitFold:
    fold: int
    heldout_source_group_ids: tuple[str, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray


def build_crossfit_plan(
    identities: Sequence[LegacyExampleIdentity], indices: np.ndarray, *, fold_count: int
) -> tuple[CrossfitFold, ...]:
    groups = sorted({identities[int(i)].source_group_id for i in indices})
    if fold_count < 2 or len(groups) < fold_count:
        raise ValueError("Grouped cross-fit requires at least one source group per fold")
    ordered = sorted(groups, key=lambda group: hashlib.sha256(group.encode()).hexdigest())
    folds = []
    for fold in range(fold_count):
        heldout = tuple(sorted(ordered[fold::fold_count]))
        test = np.asarray(
            [int(i) for i in indices if identities[int(i)].source_group_id in heldout],
            dtype=np.int64,
        )
        train = np.asarray(
            [int(i) for i in indices if identities[int(i)].source_group_id not in heldout],
            dtype=np.int64,
        )
        train_templates = {identities[int(i)].template_group_id for i in train}
        test_templates = {identities[int(i)].template_group_id for i in test}
        if train_templates & test_templates:
            raise ValueError("cross-fit template_group_id overlap")
        folds.append(CrossfitFold(fold, heldout, train, test))
    return tuple(folds)


@dataclass(frozen=True)
class LinearProbeFit:
    direction: np.ndarray
    raw_space_coefficient: np.ndarray
    raw_space_intercept: float
    metrics: dict[str, float]


def fit_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    regularization_c: float,
) -> LinearProbeFit:
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features and labels must be aligned rank-2/rank-1 arrays")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Probe fitting requires both labels 0 and 1")
    if not math.isfinite(regularization_c) or regularization_c <= 0:
        raise ValueError("regularization_c must be positive and finite")
    mean = np.asarray(features.mean(axis=0), dtype=np.float64)
    scale = np.asarray(features.std(axis=0), dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    standardized = (features - mean) / scale
    classifier = LogisticRegression(
        C=regularization_c,
        class_weight="balanced",
        max_iter=1000,
        random_state=0,
        solver="liblinear",
    )
    classifier.fit(standardized, labels)
    standardized_coefficient = np.asarray(classifier.coef_[0], dtype=np.float64)
    raw_coefficient = standardized_coefficient / scale
    norm = float(np.linalg.norm(raw_coefficient))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Fitted probe direction is non-finite or zero")
    raw_intercept = float(classifier.intercept_[0] - np.dot(mean, raw_coefficient))
    scores = features @ raw_coefficient + raw_intercept
    predictions = (scores >= 0).astype(np.int8)
    return LinearProbeFit(
        direction=raw_coefficient / norm,
        raw_space_coefficient=raw_coefficient,
        raw_space_intercept=raw_intercept,
        metrics={
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "roc_auc": float(roc_auc_score(labels, scores)),
        },
    )


def task_balanced_indices(
    identities: Sequence[LegacyExampleIdentity],
    labels: np.ndarray,
    eligible_indices: np.ndarray,
    *,
    per_task_class_cap: int,
    seed: int,
) -> np.ndarray:
    if per_task_class_cap < 1:
        raise ValueError("per_task_class_cap must be positive")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    groups = sorted({identities[int(i)].task_id for i in eligible_indices})
    for group in groups:
        for label in (0, 1):
            candidates = np.asarray(
                [
                    int(i)
                    for i in eligible_indices
                    if identities[int(i)].task_id == group and int(labels[int(i)]) == label
                ],
                dtype=np.int64,
            )
            if not len(candidates):
                continue
            rng.shuffle(candidates)
            selected.extend(candidates[:per_task_class_cap].tolist())
    result = np.asarray(sorted(selected), dtype=np.int64)
    if set(np.unique(labels[result]).tolist()) != {0, 1}:
        raise ValueError("Balanced sample does not contain both labels")
    return result


def _macro_task_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    identities: Sequence[LegacyExampleIdentity],
    indices: np.ndarray,
) -> float:
    values = []
    for task in sorted({identities[int(i)].task_id for i in indices}):
        mask = np.asarray([identities[int(i)].task_id == task for i in indices])
        task_labels = labels[indices][mask]
        if len(np.unique(task_labels)) == 2:
            values.append(float(roc_auc_score(task_labels, scores[mask])))
    if not values:
        raise ValueError("No two-class task was available for macro AUC")
    return float(np.mean(values))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        Path(temporary).unlink()
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _raw_identity_metadata_by_task(
    raw_activation_cache: Path, tasks: Sequence[str]
) -> dict[str, tuple[list[str | bytes | None], list[str | bytes | None]]]:
    import h5py

    result: dict[
        str, tuple[list[str | bytes | None], list[str | bytes | None]]
    ] = {}
    with h5py.File(raw_activation_cache, "r") as handle:
        metadata = handle.get("metadata")
        for task in tasks:
            if metadata is None or task not in metadata or "example_metadata_json" not in metadata[task]:
                result[task] = ([], [])
            else:
                group = metadata[task]
                metadata_rows = list(group["example_metadata_json"][:])
                source_rows = (
                    list(group["source_datasets"][:])
                    if "source_datasets" in group
                    else []
                )
                result[task] = (metadata_rows, source_rows)
    return result


def load_legacy_identity_and_labels(
    pooled_cache: Path, raw_activation_cache: Path
) -> tuple[list[LegacyExampleIdentity], np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    import h5py

    with h5py.File(pooled_cache, "r") as handle:
        if handle.attrs.get("format") != "qwen_answer_token_pooled_features_v1":
            raise ValueError("Unexpected pooled feature cache format")
        tasks = json.loads(str(handle.attrs["selected_tasks_json"]))
        raw_identity_metadata = _raw_identity_metadata_by_task(
            raw_activation_cache, tasks
        )
        identities: list[LegacyExampleIdentity] = []
        labels: list[int] = []
        task_local_to_global: dict[str, np.ndarray] = {}
        for task in tasks:
            group = handle["metadata"][task]
            task_labels = np.asarray(group["example_labels"][:], dtype=np.int8)
            source_indices = np.asarray(group["example_source_indices"][:], dtype=np.int64)
            output_indices = np.asarray(group["example_output_indices"][:], dtype=np.int64)
            metadata_rows, source_rows = raw_identity_metadata.get(task, ([], []))
            if metadata_rows and len(metadata_rows) != len(task_labels):
                raise ValueError(f"Raw metadata row count differs for {task}")
            if source_rows and len(source_rows) != len(task_labels):
                raise ValueError(f"Raw source-dataset row count differs for {task}")
            start = len(identities)
            for row in range(len(task_labels)):
                identities.append(
                    derive_legacy_example_identity(
                        task=task,
                        source_index=int(source_indices[row]),
                        output_index=int(output_indices[row]),
                        raw_metadata_json=metadata_rows[row] if metadata_rows else None,
                        source_dataset=source_rows[row] if source_rows else None,
                    )
                )
            labels.extend(int(value) for value in task_labels)
            task_local_to_global[task] = np.arange(start, len(identities), dtype=np.int64)
        provenance = {
            "pooled_cache_path": str(pooled_cache.resolve()),
            "pooled_cache_size_bytes": pooled_cache.stat().st_size,
            "pooled_cache_format": str(handle.attrs["format"]),
            "pooling": str(handle.attrs["pooling"]),
            "label_convention": str(handle.attrs["label_convention"]),
            "hidden_dim": int(handle.attrs["hidden_dim"]),
            "selected_layers": [int(value) for value in handle.attrs["selected_layers"]],
            "raw_activation_cache_path": str(raw_activation_cache.resolve()),
            "raw_activation_cache_size_bytes": raw_activation_cache.stat().st_size,
            "pooled_cache_dvc_identity": _dvc_pointer_identity(pooled_cache),
            "raw_activation_cache_dvc_identity": _dvc_pointer_identity(raw_activation_cache),
        }
    return identities, np.asarray(labels, dtype=np.int8), task_local_to_global, provenance


def load_features(
    pooled_cache: Path,
    raw_activation_cache: Path,
    *,
    layer: int,
    token_pooling: str,
    task_local_to_global: Mapping[str, np.ndarray],
    selected_indices: np.ndarray,
) -> np.ndarray:
    import h5py

    cache_path = (
        pooled_cache
        if token_pooling == "mean_answer_tokens_per_example"
        else raw_activation_cache
    )
    if token_pooling not in {
        "mean_answer_tokens_per_example",
        "last_answer_token_per_example",
    }:
        raise ValueError(f"Unsupported token pooling: {token_pooling}")
    with h5py.File(cache_path, "r") as handle:
        layer_group = handle[f"layer_{layer}"]
        hidden_dim = int(
            handle.attrs.get("hidden_dim", layer_group[next(iter(layer_group))].shape[1])
        )
        features: np.ndarray = np.empty(
            (len(selected_indices), hidden_dim), dtype=np.float32
        )
        output_rows = {
            int(global_index): row
            for row, global_index in enumerate(selected_indices)
        }
        selected_set = set(output_rows)
        for task, global_indices in task_local_to_global.items():
            local_rows = [
                row
                for row, value in enumerate(global_indices)
                if int(value) in selected_set
            ]
            if not local_rows:
                continue
            if token_pooling == "mean_answer_tokens_per_example":
                if int(layer_group[task].shape[0]) != len(global_indices):
                    raise ValueError(
                        f"Feature row count differs for {task} at layer {layer}"
                    )
                feature_rows = local_rows
            else:
                splits = np.asarray(
                    handle["metadata"][task]["example_splits"][:], dtype=np.int64
                )
                feature_rows = [int(splits[row + 1] - 1) for row in local_rows]
                if any(row < 0 for row in feature_rows):
                    raise ValueError(f"Empty answer-token example in {task}")
            values = np.asarray(layer_group[task][feature_rows], dtype=np.float32)
            for value_row, local_row in enumerate(local_rows):
                features[output_rows[int(global_indices[local_row])]] = values[value_row]
    return features


def _probe_payload(
    *,
    probe_id: str,
    layer: int,
    pooling: str,
    sign: str,
    fit: LinearProbeFit,
    train_indices: np.ndarray,
    evaluation: Mapping[str, Any],
    identities: Sequence[LegacyExampleIdentity],
    step5_plan_manifest_sha256: str,
) -> dict[str, Any]:
    body = {
        "format": PROBE_ARTIFACT_FORMAT,
        "probe_id": probe_id,
        "layer": layer,
        "token_pooling": pooling,
        "direction_sign_convention": sign,
        "label_convention": LABEL_CONVENTION,
        "step5_plan_manifest_sha256": step5_plan_manifest_sha256,
        "final_direction": {"direction_vector": fit.direction.tolist()},
        "classifier": {
            "raw_space_coefficient": fit.raw_space_coefficient.tolist(),
            "raw_space_intercept": fit.raw_space_intercept,
            "training_metrics": fit.metrics,
        },
        "training": {
            "example_count": int(len(train_indices)),
            "source_group_ids": sorted({identities[int(i)].source_group_id for i in train_indices}),
            "template_group_count": len({identities[int(i)].template_group_id for i in train_indices}),
        },
        "evaluation": dict(evaluation),
    }
    return {**body, "receipt_sha256": _stable_json_sha256(body)}


def run_grouped_probe_fit(
    *,
    pooled_cache: Path,
    raw_activation_cache: Path,
    config_path: Path,
    step5_plan_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Calibrate on regularizer sources, then fit source-cross-fit evaluator probes."""

    config = json.loads(config_path.read_text())
    if config.get("format") != FIT_CONFIG_FORMAT:
        raise ValueError(f"Config format must be {FIT_CONFIG_FORMAT}")
    step5_plan_sha256 = hashlib.sha256(step5_plan_path.read_bytes()).hexdigest()
    if config.get("step5_plan_manifest_sha256") != step5_plan_sha256:
        raise ValueError(
            "Current Step 5 plan manifest does not match the hash frozen in the probe fit config"
        )
    calibration = config["calibration"]
    if config["qualification"].get("direction_sign_convention") != DIRECTION_SIGN_CONVENTION:
        raise ValueError(
            "Probe direction sign convention must match the repository canonical contract"
        )
    pooling_candidates = list(calibration["candidate_token_pooling"])
    identities, labels, task_map, provenance = load_legacy_identity_and_labels(
        pooled_cache, raw_activation_cache
    )
    project_root = config_path.resolve().parents[1]
    for field in ("pooled_cache_path", "raw_activation_cache_path"):
        provenance[field] = _portable_path(provenance[field], project_root=project_root)
    for field in ("pooled_cache_dvc_identity", "raw_activation_cache_dvc_identity"):
        if provenance[field] is not None:
            provenance[field]["pointer_path"] = _portable_path(
                provenance[field]["pointer_path"], project_root=project_root
            )
    supported_poolings = {
        "mean_answer_tokens_per_example",
        "last_answer_token_per_example",
    }
    if set(pooling_candidates) != supported_poolings or len(pooling_candidates) != 2:
        raise ValueError(
            "Pooling calibration must compare the frozen mean- and last-answer-token candidates"
        )
    if provenance["label_convention"] != LABEL_CONVENTION:
        raise ValueError(
            f"Legacy cache label convention must be {LABEL_CONVENTION!r}"
        )
    outer = build_outer_split(
        identities,
        evaluator_task_ids={
            _identifier_component(value) for value in config["evaluator_task_ids"]
        },
    )
    cap = int(calibration["general_task_class_cap"])
    seed = int(calibration["random_seed"])
    regularizer_sample = task_balanced_indices(
        identities, labels, outer.regularizer_indices, per_task_class_cap=cap, seed=seed
    )
    evaluator_sample = task_balanced_indices(
        identities, labels, outer.evaluator_indices, per_task_class_cap=cap, seed=seed + 1
    )
    analysis_indices = np.asarray(
        sorted(set(regularizer_sample.tolist()) | set(evaluator_sample.tolist())),
        dtype=np.int64,
    )
    feature_row = {int(global_index): row for row, global_index in enumerate(analysis_indices)}

    def rows(indices: np.ndarray) -> np.ndarray:
        return np.asarray([feature_row[int(index)] for index in indices], dtype=np.int64)

    regularizer_groups = list(outer.regularizer_source_groups)
    fold_count = int(calibration["cross_validation_folds"])
    if fold_count < 2 or fold_count > len(regularizer_groups):
        raise ValueError("Invalid regularizer calibration fold count")
    ordered_groups = sorted(
        regularizer_groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
    )
    calibration_group_folds = [
        set(ordered_groups[fold::fold_count]) for fold in range(fold_count)
    ]
    calibration_rows = []
    for token_pooling in pooling_candidates:
        for layer in calibration["candidate_layers"]:
            layer = int(layer)
            features = load_features(
                pooled_cache,
                raw_activation_cache,
                layer=layer,
                token_pooling=token_pooling,
                task_local_to_global=task_map,
                selected_indices=analysis_indices,
            )
            fold_aucs = []
            for heldout_groups in calibration_group_folds:
                train = np.asarray(
                    [i for i in regularizer_sample if identities[int(i)].source_group_id not in heldout_groups],
                    dtype=np.int64,
                )
                test = np.asarray(
                    [i for i in regularizer_sample if identities[int(i)].source_group_id in heldout_groups],
                    dtype=np.int64,
                )
                if not len(train) or not len(test) or len(np.unique(labels[test])) < 2:
                    continue
                fit = fit_linear_probe(features[rows(train)], labels[train], regularization_c=float(calibration["regularization_c"]))
                scores = features[rows(test)] @ fit.raw_space_coefficient + fit.raw_space_intercept
                fold_aucs.append(_macro_task_auc(scores, labels, identities, test))
            if not fold_aucs:
                raise ValueError(
                    f"No valid calibration folds for layer {layer} / {token_pooling}"
                )
            calibration_rows.append(
                {"layer": layer, "token_pooling": token_pooling, "fold_macro_task_roc_auc": fold_aucs, "mean_macro_task_roc_auc": float(np.mean(fold_aucs))}
            )
    selected = max(calibration_rows, key=lambda row: (row["mean_macro_task_roc_auc"], -row["layer"]))
    selected_layer = int(selected["layer"])
    selected_pooling = str(selected["token_pooling"])
    # Reload only the winning layer. Retaining every candidate layer would turn
    # a CPU qualification into a multi-gigabyte memory spike for no benefit.
    features = load_features(
        pooled_cache,
        raw_activation_cache,
        layer=selected_layer,
        token_pooling=selected_pooling,
        task_local_to_global=task_map,
        selected_indices=analysis_indices,
    )
    regularizer_fit = fit_linear_probe(
        features[rows(regularizer_sample)], labels[regularizer_sample], regularization_c=float(calibration["regularization_c"])
    )
    sign = str(config["qualification"]["direction_sign_convention"])
    probe_dir = output_root / "probes"
    regularizer_id = "legacy-grouped-regularizer"
    regularizer_path = probe_dir / f"{regularizer_id}.json"
    _atomic_json(
        regularizer_path,
        _probe_payload(
            probe_id=regularizer_id,
            layer=selected_layer,
            pooling=selected_pooling,
            sign=sign,
            fit=regularizer_fit,
            train_indices=regularizer_sample,
            evaluation={"kind": "regularizer_calibration_only", "selected_layer_score": selected["mean_macro_task_roc_auc"]},
            identities=identities,
            step5_plan_manifest_sha256=step5_plan_sha256,
        ),
    )
    registry_probes = [
        {
            "probe_id": regularizer_id,
            "ensemble": "regularizer",
            "artifact_path": os.path.relpath(regularizer_path, output_root),
            "artifact_direction_path": ["final_direction", "direction_vector"],
            "source_group_ids": list(outer.regularizer_source_groups),
            "example_ids": [identities[int(i)].example_id for i in regularizer_sample],
            "layer": selected_layer,
            "token_pooling": selected_pooling,
            "direction_sign_convention": sign,
        }
    ]
    crossfit_rows = []
    for fold in build_crossfit_plan(
        identities,
        evaluator_sample,
        fold_count=int(config["evaluator_crossfit_folds"]),
    ):
        fit = fit_linear_probe(
            features[rows(fold.train_indices)], labels[fold.train_indices], regularization_c=float(calibration["regularization_c"])
        )
        scores = features[rows(fold.test_indices)] @ fit.raw_space_coefficient + fit.raw_space_intercept
        heldout_metrics = {
            "balanced_accuracy": float(balanced_accuracy_score(labels[fold.test_indices], scores >= 0)),
            "roc_auc": float(roc_auc_score(labels[fold.test_indices], scores)),
        }
        probe_id = f"legacy-grouped-evaluator-{fold.fold:02d}"
        probe_path = probe_dir / f"{probe_id}.json"
        _atomic_json(
            probe_path,
            _probe_payload(
                probe_id=probe_id,
                layer=selected_layer,
                pooling=selected_pooling,
                sign=sign,
                fit=fit,
                train_indices=fold.train_indices,
                evaluation={"kind": "grouped_cross_fit", "heldout_source_group_ids": list(fold.heldout_source_group_ids), "heldout_example_count": int(len(fold.test_indices)), "heldout_metrics": heldout_metrics},
                identities=identities,
                step5_plan_manifest_sha256=step5_plan_sha256,
            ),
        )
        crossfit_rows.append({"probe_id": probe_id, "heldout_source_group_ids": list(fold.heldout_source_group_ids), **heldout_metrics})
        registry_probes.append(
            {
                "probe_id": probe_id,
                "ensemble": "evaluator",
                "artifact_path": os.path.relpath(probe_path, output_root),
                "artifact_direction_path": ["final_direction", "direction_vector"],
                "source_group_ids": sorted({identities[int(i)].source_group_id for i in fold.train_indices}),
                "example_ids": [identities[int(i)].example_id for i in fold.train_indices],
                "layer": selected_layer,
                "token_pooling": selected_pooling,
                "direction_sign_convention": sign,
            }
        )
    example_ids = sorted(identity.example_id for identity in identities)
    example_group_ids = sorted({identity.example_group_id for identity in identities})
    template_group_ids = sorted({identity.template_group_id for identity in identities})
    identity_body = {
        "format": IDENTITY_FORMAT,
        "policy": {
            "outer_split_unit": "source_group_id",
            "crossfit_split_unit": "source_group_id",
            "source_group_derivation": "normalized upstream task identifier",
            "example_group_derivation": "normalized task plus upstream example_source_index",
            "template_group_derivation": "explicit pair_index, then nested source_index, then example_source_index",
            "identity_derivation_uses_labels": False,
        },
        "inventory": {
            "example_count": len(example_ids),
            "unique_example_count": len(set(example_ids)),
            "example_group_count": len(example_group_ids),
            "template_group_count": len(template_group_ids),
            "source_group_ids": sorted({identity.source_group_id for identity in identities}),
        },
        "receipts": {
            "example_ids_sha256": _stable_json_sha256(example_ids),
            "example_group_ids_sha256": _stable_json_sha256(example_group_ids),
            "template_group_ids_sha256": _stable_json_sha256(template_group_ids),
        },
    }
    identity_registry = {**identity_body, "receipt_sha256": _stable_json_sha256(identity_body)}
    _atomic_json(output_root / "legacy_identity_registry.json", identity_registry)
    registry = {
        "format": "intelligent_liars_step5_probe_registry_v1",
        "qualification": {
            "layer": selected_layer,
            "token_pooling": selected_pooling,
            "direction_sign_convention": sign,
            "orthogonal_controls_per_probe": int(config["qualification"]["orthogonal_controls_per_probe"]),
        },
        "probes": registry_probes,
    }
    _atomic_json(output_root / "probe_registry.json", registry)
    report_body = {
        "format": FIT_REPORT_FORMAT,
        "status": "fit_complete_qualification_pending",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "step5_plan": {
            "path": _portable_path(step5_plan_path, project_root=project_root),
            "sha256": step5_plan_sha256,
        },
        "provenance": provenance,
        "split": {
            "regularizer_source_group_ids": list(outer.regularizer_source_groups),
            "evaluator_source_group_ids": list(outer.evaluator_source_groups),
            "regularizer_sample_examples": int(len(regularizer_sample)),
            "evaluator_sample_examples": int(len(evaluator_sample)),
        },
        "calibration": {"scope": "outer_regularizer_sources_only", "rows": calibration_rows, "selected": selected},
        "evaluator_crossfit": crossfit_rows,
    }
    report = {**report_body, "receipt_sha256": _stable_json_sha256(report_body)}
    _atomic_json(output_root / "fit_report.json", report)
    return report
