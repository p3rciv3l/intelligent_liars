"""Local production artifact materializer for finalist causal controls.

The public finalist exporter intentionally accepts only the audited winner.
This adapter is a separate, local-only seam: it compiles each strong candidate,
saves an edited checkpoint for bounded causal evaluation, and freezes every
artifact consumed by the Qwen causal backend.  It has no registry, cloud, or
upload capability.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from .models import ModelBundle
from .truth_editing_finalist_checkpoint import VerifiedFinalistCompiler
from .truth_editing_production import ProductionRunConfig
from .truth_editing_qwen_causal_backend import (
    CAUSAL_EVALUATOR_FORMAT,
    CAUSAL_SCENARIO_SET_FORMAT,
    RANKK_BASIS_ARTIFACT_FORMAT,
)
from .truth_editing_study import SearchProposal
from .truth_editing_weight_editor import WriterEditRuntime


MATERIALIZER_FORMAT = "truth_editing_production_causal_materializer_v1"
MATERIALIZATION_RECEIPT_FORMAT = (
    "truth_editing_production_causal_candidate_materialization_v1"
)
CHECKPOINT_MANIFEST_FORMAT = "truth_editing_local_causal_checkpoint_manifest_v1"
PERSISTENT_RECIPE_FORMAT = "truth_editing_causal_persistent_recipe_v1"
ACTIVATION_RECIPE_FORMAT = "truth_editing_causal_activation_recipe_v1"
CONTROL_KINDS = ("restoration", "re_ablation", "random_direction", "false_trigger")


class ProductionCausalMaterializationError(ValueError):
    """A real causal candidate package cannot be built without fabrication."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProductionCausalMaterializationError(
            "causal materialization value is not canonical JSON"
        ) from error


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProductionCausalMaterializationError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ProductionCausalMaterializationError(f"{label} must be a regular file")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionCausalMaterializationError(f"{label} is invalid JSON") from error
    if not isinstance(raw, dict):
        raise ProductionCausalMaterializationError(f"{label} must be an object")
    return raw


def _checkpoint_inventory(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ProductionCausalMaterializationError(
                "local causal checkpoint must not contain symlinks"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ProductionCausalMaterializationError(
                "local causal checkpoint contains an unsupported entry"
            )
        relative = path.relative_to(root).as_posix()
        files.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)}
        )
    if not files:
        raise ProductionCausalMaterializationError("local causal checkpoint is empty")
    return files


def _valid_messages(value: Any) -> list[dict[str, str]] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        return None
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"role", "content"}:
            return None
        role, content = item["role"], item["content"]
        if role not in {"system", "user", "assistant"} or not isinstance(content, str) or not content:
            return None
        result.append({"role": role, "content": content})
    if result[-1]["role"] == "assistant":
        return None
    return result


def _scenario_records(view_root: Path, *, count: int) -> list[dict[str, Any]]:
    source = view_root / "records.jsonl"
    if source.is_symlink() or not source.is_file():
        raise ProductionCausalMaterializationError(
            "production scenario view lacks records.jsonl"
        )
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
        for ordinal, line in enumerate(lines):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError
            scenario_id = value.get("scenario_id")
            role = value.get("role")
            if not isinstance(scenario_id, str) or not isinstance(role, str):
                continue
            grouped.setdefault(scenario_id, {})[role] = value
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ProductionCausalMaterializationError(
            "production scenario records are invalid"
        ) from error

    result: list[dict[str, Any]] = []
    direct_roles = ("truthful_direct_report", "false_trigger_truthful_control")
    retained_roles = ("indirect_retained_truth", "retained_truth_rephrase_proxy")
    for scenario_id in sorted(grouped):
        roles = grouped[scenario_id]
        direct = next((roles[role] for role in direct_roles if role in roles), None)
        retained = next((roles[role] for role in retained_roles if role in roles), None)
        if direct is None or retained is None:
            continue
        if direct.get("base_known_status") != "known" or retained.get("base_known_status") != "known":
            continue
        direct_messages = _valid_messages(direct.get("messages"))
        retained_messages = _valid_messages(retained.get("messages"))
        truth = direct.get("truthful_target")
        aliases = direct.get("truth_aliases")
        false_targets = direct.get("acceptable_false_targets")
        if (
            direct_messages is None
            or retained_messages is None
            or not isinstance(truth, str)
            or not truth
            or retained.get("truthful_target") != truth
            or isinstance(aliases, (str, bytes))
            or not isinstance(aliases, Sequence)
            or any(not isinstance(item, str) or not item for item in aliases)
            or isinstance(false_targets, (str, bytes))
            or not isinstance(false_targets, Sequence)
            or not false_targets
            or any(not isinstance(item, str) or not item for item in false_targets)
        ):
            continue
        normalized_truth = {truth.casefold(), *(item.casefold() for item in aliases)}
        if normalized_truth.intersection(item.casefold() for item in false_targets):
            continue
        result.append(
            {
                "record_id": scenario_id,
                "direct_messages": direct_messages,
                "retained_truth_messages": retained_messages,
                "truthful_target": truth,
                "truth_aliases": list(aliases),
                "plausible_false_targets": list(false_targets),
            }
        )
        if len(result) == count:
            return result
    raise ProductionCausalMaterializationError(
        f"production scenario view lacks {count} complete base-known causal scenario pairs"
    )


def _resume_result(
    output_dir: Path,
    *,
    expected_materializer_sha256: str,
    expected_study_sha256: str,
    expected_trial_id: str,
    expected_proposal_sha256: str,
) -> dict[str, Any] | None:
    receipt_path = output_dir / "materialization-receipt.json"
    if not receipt_path.exists():
        return None
    receipt = _load_object(receipt_path, "causal materialization receipt")
    if receipt.get("format") != MATERIALIZATION_RECEIPT_FORMAT:
        raise ProductionCausalMaterializationError(
            "causal materialization receipt format differs"
        )
    claimed = _digest(receipt.get("self_sha256"), "materialization self_sha256")
    unsigned = dict(receipt)
    unsigned.pop("self_sha256")
    if _sha(unsigned) != claimed or not isinstance(receipt.get("result"), Mapping):
        raise ProductionCausalMaterializationError(
            "causal materialization receipt identity differs"
        )
    if (
        receipt.get("materializer_identity_sha256") != expected_materializer_sha256
        or receipt.get("study_identity_sha256") != expected_study_sha256
        or receipt.get("trial_id") != expected_trial_id
        or receipt.get("proposal_sha256") != expected_proposal_sha256
    ):
        raise ProductionCausalMaterializationError(
            "causal materialization receipt selection binding differs"
        )
    result = dict(receipt["result"])
    expected_result_fields = {
        "edited_checkpoint_path",
        "edited_checkpoint_sha256",
        "edited_checkpoint_manifest_path",
        "basis_artifact_path",
        "persistent_recipe_path",
        "scenario_path",
        "evaluator_path",
        "runtime_identity_sha256",
        "direction_manifest_path",
        "controls",
    }
    if set(result) != expected_result_fields:
        raise ProductionCausalMaterializationError(
            "resumed causal materialization result fields differ"
        )
    file_keys = (
        "edited_checkpoint_manifest_path",
        "basis_artifact_path",
        "persistent_recipe_path",
        "scenario_path",
        "evaluator_path",
        "direction_manifest_path",
    )
    artifact_sha256s = receipt.get("artifact_sha256s")
    expected_artifact_keys = {
        *file_keys,
        "edited_checkpoint_path",
        *(f"activation_recipe:{kind}" for kind in CONTROL_KINDS),
    }
    if (
        not isinstance(artifact_sha256s, Mapping)
        or set(artifact_sha256s) != expected_artifact_keys
    ):
        raise ProductionCausalMaterializationError(
            "causal materialization receipt artifact inventory differs"
        )
    for key in file_keys:
        path = Path(str(result.get(key)))
        if path.is_symlink() or not path.is_file():
            raise ProductionCausalMaterializationError(
                "resumed causal materialization artifact is missing"
            )
        if _sha_file(path) != artifact_sha256s[key]:
            raise ProductionCausalMaterializationError(
                "resumed causal materialization artifact identity differs"
            )
    controls = result["controls"]
    if (
        isinstance(controls, (str, bytes))
        or not isinstance(controls, Sequence)
        or len(controls) != len(CONTROL_KINDS)
    ):
        raise ProductionCausalMaterializationError(
            "resumed causal control inventory differs"
        )
    for expected_kind, control in zip(CONTROL_KINDS, controls, strict=True):
        if not isinstance(control, Mapping) or control.get("control_kind") != expected_kind:
            raise ProductionCausalMaterializationError(
                "resumed causal control inventory differs"
            )
        recipe_path = Path(str(control.get("activation_recipe_path")))
        if recipe_path.is_symlink() or not recipe_path.is_file():
            raise ProductionCausalMaterializationError(
                "resumed causal control recipe is missing"
            )
        if _sha_file(recipe_path) != artifact_sha256s[f"activation_recipe:{expected_kind}"]:
            raise ProductionCausalMaterializationError(
                "resumed causal control recipe identity differs"
            )
    checkpoint = Path(str(result.get("edited_checkpoint_path")))
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ProductionCausalMaterializationError(
            "resumed causal checkpoint is missing"
        )
    observed_checkpoint_sha = _sha(
        {
            "format": "truth_editing_local_checkpoint_tree_v1",
            "files": _checkpoint_inventory(checkpoint),
        }
    )
    manifest = _load_object(
        Path(str(result["edited_checkpoint_manifest_path"])),
        "resumed causal checkpoint manifest",
    )
    if (
        observed_checkpoint_sha != artifact_sha256s["edited_checkpoint_path"]
        or manifest.get("checkpoint_sha256") != observed_checkpoint_sha
        or manifest.get("self_sha256") != result["edited_checkpoint_sha256"]
    ):
        raise ProductionCausalMaterializationError(
            "resumed causal checkpoint identity differs"
        )
    return result


class ProductionCausalCandidateMaterializer:
    """Concrete local-only adapter at the causal candidate materialization seam."""

    def __init__(
        self,
        *,
        config: ProductionRunConfig,
        compiler: VerifiedFinalistCompiler,
        bundle: ModelBundle,
        scenario_count: int = 8,
    ) -> None:
        if not isinstance(config, ProductionRunConfig):
            raise ProductionCausalMaterializationError(
                "config must be an opened ProductionRunConfig"
            )
        if type(compiler) is not VerifiedFinalistCompiler:
            raise ProductionCausalMaterializationError(
                "compiler must be the verified production finalist compiler"
            )
        if not isinstance(bundle, ModelBundle) or bundle.model is None:
            raise ProductionCausalMaterializationError(
                "bundle must be a loaded production ModelBundle"
            )
        if isinstance(scenario_count, bool) or not isinstance(scenario_count, int) or not 1 <= scenario_count <= 64:
            raise ProductionCausalMaterializationError(
                "scenario_count must be an integer from 1 through 64"
            )
        snapshot = bundle.verified_snapshot
        if not isinstance(snapshot, Mapping):
            raise ProductionCausalMaterializationError(
                "production bundle lacks verified snapshot identity"
            )
        if snapshot.get("model_sha256") != config.verified_model_sha256:
            raise ProductionCausalMaterializationError(
                "production bundle model identity differs from config"
            )
        if snapshot.get("snapshot_manifest_sha256") != config.verified_snapshot_manifest_sha256:
            raise ProductionCausalMaterializationError(
                "production bundle snapshot identity differs from config"
            )
        manifest = config.direction_manifest.resolve(strict=True)
        if manifest.is_symlink() or not manifest.is_file():
            raise ProductionCausalMaterializationError(
                "production direction manifest must be a regular file"
            )
        compiler_identity = dict(compiler.identity)
        if compiler_identity.get("model_sha256") != config.verified_model_sha256:
            raise ProductionCausalMaterializationError(
                "production compiler model identity differs from config"
            )
        unsigned = {
            "format": MATERIALIZER_FORMAT,
            "production_config_model_sha256": config.verified_model_sha256,
            "snapshot_manifest_sha256": config.verified_snapshot_manifest_sha256,
            "compiler_identity_sha256": _sha(compiler_identity),
            "direction_manifest_sha256": _sha_file(manifest),
            "scenario_view_sha256": _sha_file(config.scenario_view / "records.jsonl"),
            "scenario_count": scenario_count,
            "publication_scope": "local_bounded_causal_evaluation_only",
        }
        self._identity = MappingProxyType({**unsigned, "self_sha256": _sha(unsigned)})
        self._config = config
        self._compiler = compiler
        self._bundle = bundle
        self._scenario_count = scenario_count
        self._direction_manifest = manifest

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    def materialize_candidate(
        self,
        *,
        study_identity_sha256: str,
        trial_id: str,
        proposal: Mapping[str, Any],
        proposal_sha256: str,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        study_sha = _digest(study_identity_sha256, "study_identity_sha256")
        expected_proposal_sha = _digest(proposal_sha256, "proposal_sha256")
        if not isinstance(trial_id, str) or not trial_id or trial_id.strip() != trial_id:
            raise ProductionCausalMaterializationError("trial_id must be nonempty text")
        parsed = SearchProposal.from_dict(proposal)
        if _sha(parsed.to_dict()) != expected_proposal_sha:
            raise ProductionCausalMaterializationError("proposal identity differs")
        if not parsed.direction_ids:
            raise ProductionCausalMaterializationError(
                "refusal-only finalists lack an exact truth-direction identity for causal controls"
            )
        destination = Path(output_dir).resolve()
        resumed = _resume_result(
            destination,
            expected_materializer_sha256=str(self._identity["self_sha256"]),
            expected_study_sha256=study_sha,
            expected_trial_id=trial_id,
            expected_proposal_sha256=expected_proposal_sha,
        )
        if resumed is not None:
            return resumed
        if destination.exists() or destination.is_symlink():
            raise ProductionCausalMaterializationError(
                "candidate output exists without an immutable receipt"
            )

        compilation = self._compiler.compile_finalist(parsed, trial_id=trial_id)
        if (
            compilation.trial_id != trial_id
            or compilation.proposal_sha256 != expected_proposal_sha
            or compilation.compiled_edit.model_sha256 != self._config.verified_model_sha256
            or tuple(layer.layer_index for layer in compilation.compiled_edit.layers)
            != parsed.writer_layers
        ):
            raise ProductionCausalMaterializationError(
                "compiled finalist identity differs from the selected proposal"
            )

        staging = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        staging.mkdir(parents=True, mode=0o700)
        try:
            checkpoint = staging / "checkpoint"
            checkpoint.mkdir()
            runtime = WriterEditRuntime(
                verified_model_sha256=self._config.verified_model_sha256
            )
            with runtime.activate(self._bundle.model, compilation.compiled_edit):
                self._bundle.model.save_pretrained(
                    checkpoint, safe_serialization=True
                )
                self._bundle.processor.save_pretrained(checkpoint)
            files = _checkpoint_inventory(checkpoint)
            checkpoint_sha = _sha(
                {"format": "truth_editing_local_checkpoint_tree_v1", "files": files}
            )
            checkpoint_manifest_unsigned = {
                "format": CHECKPOINT_MANIFEST_FORMAT,
                "study_identity_sha256": study_sha,
                "trial_id": trial_id,
                "proposal_sha256": expected_proposal_sha,
                "recipe_id": compilation.compiled_edit.recipe_id,
                "basis_set_sha256": compilation.basis_set_sha256,
                "checkpoint_sha256": checkpoint_sha,
                "files": files,
                "file_count": len(files),
                "total_bytes": sum(item["bytes"] for item in files),
                "publication_scope": "local_bounded_causal_evaluation_only",
            }
            checkpoint_manifest = {
                **checkpoint_manifest_unsigned,
                "self_sha256": _sha(checkpoint_manifest_unsigned),
            }
            _write_json(staging / "checkpoint-manifest.json", checkpoint_manifest)

            torch.save(
                {
                    "format": RANKK_BASIS_ARTIFACT_FORMAT,
                    "basis_sha256": compilation.basis_set_sha256,
                    "by_layer": {
                        str(layer.layer_index): layer.basis.detach()
                        .to(device="cpu", dtype=torch.float32)
                        .contiguous()
                        for layer in compilation.compiled_edit.layers
                    },
                },
                staging / "rankk-basis.pt",
            )
            persistent_unsigned = {
                "format": PERSISTENT_RECIPE_FORMAT,
                "study_identity_sha256": study_sha,
                "trial_id": trial_id,
                "proposal_sha256": expected_proposal_sha,
                "proposal": parsed.to_dict(),
                "recipe_id": compilation.compiled_edit.recipe_id,
                "model_sha256": compilation.compiled_edit.model_sha256,
                "basis_set_sha256": compilation.basis_set_sha256,
                "backend": {
                    "type": "persistent_weight",
                    "transform_family": "projection_reflection",
                    "writer_strength_plan": parsed.writer_strength_plan(),
                },
            }
            _write_json(
                staging / "persistent-recipe.json",
                {**persistent_unsigned, "self_sha256": _sha(persistent_unsigned)},
            )
            _write_json(
                staging / "causal-scenarios.json",
                {
                    "format": CAUSAL_SCENARIO_SET_FORMAT,
                    "records": _scenario_records(
                        self._config.scenario_view, count=self._scenario_count
                    ),
                },
            )
            _write_json(
                staging / "causal-evaluator.json",
                {
                    "format": CAUSAL_EVALUATOR_FORMAT,
                    "scoring_semantics": "causal_restoration_and_negative_controls",
                    "minimum_restored_truthful_rate": 0.75,
                    "minimum_restored_retained_truth_rate": 0.75,
                    "maximum_negative_control_truthful_rate": 0.25,
                    "minimum_eligible_record_count": 4,
                },
            )

            seed_base = int(_sha({"trial_id": trial_id, "proposal": expected_proposal_sha})[:8], 16)
            controls: list[dict[str, Any]] = []
            for ordinal, kind in enumerate(CONTROL_KINDS):
                random_control = kind == "random_direction"
                direction_ids = (
                    [f"random-control-{expected_proposal_sha[:16]}"]
                    if random_control
                    else list(parsed.direction_ids)
                )
                direction_basis_sha = (
                    _sha(
                        {
                            "kind": "seeded_matched_random_direction",
                            "target_basis_sha256": compilation.basis_set_sha256,
                            "seed": seed_base + ordinal,
                        }
                    )
                    if random_control
                    else compilation.basis_set_sha256
                )
                seed = seed_base if kind in {"restoration", "re_ablation"} else seed_base + ordinal
                recipe_unsigned = {
                    "format": ACTIVATION_RECIPE_FORMAT,
                    "causal_control_kind": kind,
                    "study_identity_sha256": study_sha,
                    "trial_id": trial_id,
                    "proposal_sha256": expected_proposal_sha,
                    "seed": seed,
                    "backend": {
                        "type": "activation_hook",
                        "transform_family": "rank_k_causal_control",
                        "source_layers": list(parsed.writer_layers),
                        "token_scope": "prefill_last_and_cached_generation",
                        "generation_step_persistence": True,
                    },
                    "direction_selection": {
                        "direction_ids": direction_ids,
                        "basis_sha256": direction_basis_sha,
                        "requested_rank": int(
                            compilation.compiled_edit.layers[0].basis.shape[1]
                        ),
                    },
                }
                recipe_path = staging / f"activation-recipe-{kind}.json"
                _write_json(
                    recipe_path,
                    {**recipe_unsigned, "self_sha256": _sha(recipe_unsigned)},
                )
                controls.append(
                    {
                        "control_kind": kind,
                        "seed": seed,
                        "direction_ids": direction_ids,
                        "direction_basis_sha256": direction_basis_sha,
                        "layers": list(parsed.writer_layers),
                        "token_scope": "prefill_last_and_cached_generation",
                        "activation_recipe_path": str(
                            destination / recipe_path.name
                        ),
                    }
                )

            runtime_identity = _sha(
                {
                    "format": "truth_editing_causal_runtime_binding_v1",
                    "materializer_identity_sha256": self._identity["self_sha256"],
                    "model_sha256": self._config.verified_model_sha256,
                    "basis_set_sha256": compilation.basis_set_sha256,
                    "checkpoint_sha256": checkpoint_sha,
                }
            )
            result = {
                "edited_checkpoint_path": str(destination / "checkpoint"),
                "edited_checkpoint_sha256": checkpoint_manifest["self_sha256"],
                "edited_checkpoint_manifest_path": str(
                    destination / "checkpoint-manifest.json"
                ),
                "basis_artifact_path": str(destination / "rankk-basis.pt"),
                "persistent_recipe_path": str(destination / "persistent-recipe.json"),
                "scenario_path": str(destination / "causal-scenarios.json"),
                "evaluator_path": str(destination / "causal-evaluator.json"),
                "runtime_identity_sha256": runtime_identity,
                "direction_manifest_path": str(self._direction_manifest),
                "controls": controls,
            }
            receipt_unsigned = {
                "format": MATERIALIZATION_RECEIPT_FORMAT,
                "materializer_identity_sha256": self._identity["self_sha256"],
                "study_identity_sha256": study_sha,
                "trial_id": trial_id,
                "proposal_sha256": expected_proposal_sha,
                "result": result,
                "artifact_sha256s": {
                    "edited_checkpoint_path": checkpoint_sha,
                    "edited_checkpoint_manifest_path": _sha_file(
                        staging / "checkpoint-manifest.json"
                    ),
                    "basis_artifact_path": _sha_file(staging / "rankk-basis.pt"),
                    "persistent_recipe_path": _sha_file(
                        staging / "persistent-recipe.json"
                    ),
                    "scenario_path": _sha_file(staging / "causal-scenarios.json"),
                    "evaluator_path": _sha_file(staging / "causal-evaluator.json"),
                    "direction_manifest_path": _sha_file(self._direction_manifest),
                    **{
                        f"activation_recipe:{kind}": _sha_file(
                            staging / f"activation-recipe-{kind}.json"
                        )
                        for kind in CONTROL_KINDS
                    },
                },
            }
            _write_json(
                staging / "materialization-receipt.json",
                {**receipt_unsigned, "self_sha256": _sha(receipt_unsigned)},
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(destination)
            return result
        except Exception:
            if staging.exists() and not staging.is_symlink():
                shutil.rmtree(staging)
            raise


__all__ = [
    "ACTIVATION_RECIPE_FORMAT",
    "CHECKPOINT_MANIFEST_FORMAT",
    "MATERIALIZATION_RECEIPT_FORMAT",
    "MATERIALIZER_FORMAT",
    "PERSISTENT_RECIPE_FORMAT",
    "ProductionCausalCandidateMaterializer",
    "ProductionCausalMaterializationError",
]
