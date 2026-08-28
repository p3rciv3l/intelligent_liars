"""Prepare the bounded causal evidence required before adaptive finalization.

This module deliberately owns orchestration, not model-specific artifact
construction.  A production materializer turns one selected proposal into the
exact checkpoint, basis, recipes, and frozen evaluation inputs.  The
orchestrator then builds and executes the strict public causal contracts for
every strong candidate before returning a receipt inventory to finalization.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .truth_editing_adaptive_finalization import open_adaptive_finalization_handoff
from .truth_editing_causal_activation_controls import (
    CausalActivationControlExecutor,
    build_causal_activation_control_plan,
    open_causal_activation_control_receipt,
    run_causal_activation_controls,
)
from .truth_editing_finalist_checkpoint import (
    rank_pareto_finalists,
    select_pareto_finalists,
)
from .truth_editing_production_judge_budget import (
    ProductionJudgeBudgetError,
    parse_production_judge_budget_receipt,
)
from .truth_editing_qwen_causal_backend import build_causal_backend_config


PREPARATION_RECEIPT_FORMAT = "truth_editing_adaptive_causal_preparation_receipt_v1"


class AdaptiveCausalPreparationError(RuntimeError):
    """The causal finalist inventory cannot be completed without fabrication."""


class CausalCandidateMaterializer(Protocol):
    """Materialize exact model-specific artifacts for one selected proposal."""

    @property
    def identity(self) -> Mapping[str, Any]: ...

    def materialize_candidate(
        self,
        *,
        study_identity_sha256: str,
        trial_id: str,
        proposal: Mapping[str, Any],
        proposal_sha256: str,
        output_dir: Path,
    ) -> Mapping[str, Any]: ...


class CausalExecutorFactory(Protocol):
    def __call__(self, *, config_path: Path) -> CausalActivationControlExecutor: ...


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise AdaptiveCausalPreparationError("value is not canonical JSON") from error


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new_or_same(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_text() != payload:
            raise AdaptiveCausalPreparationError(f"immutable output differs: {path}")


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AdaptiveCausalPreparationError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdaptiveCausalPreparationError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise AdaptiveCausalPreparationError(f"{label} must be an object")
    return value


def _materialized(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
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
    raw = dict(value)
    if set(raw) != required:
        raise AdaptiveCausalPreparationError("causal candidate artifact fields differ")
    controls = raw["controls"]
    if (
        isinstance(controls, (str, bytes))
        or not isinstance(controls, Sequence)
        or len(controls) != 4
    ):
        raise AdaptiveCausalPreparationError(
            "causal candidate must materialize exactly four controls"
        )
    return raw


def prepare_adaptive_causal_controls(
    handoff_path: Path | str,
    *,
    compiler_identity: Mapping[str, Any],
    materializer: CausalCandidateMaterializer,
    executor_factory: CausalExecutorFactory,
    causal_root: Path | str | None = None,
    before_candidate_execute: Callable[[str], None] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    after_candidate_commit: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Path]:
    """Run four causal controls for every runtime-selected strong candidate.

    Returned paths are safe to pass directly as ``causal_control_receipts`` to
    :class:`ProductionAdaptiveFinalizationExecutor`.  The first receipt must
    begin at the authoritative judge-budget receipt and every later receipt
    must continue that exact ledger chain.
    """

    handoff = open_adaptive_finalization_handoff(handoff_path)
    report_path = Path(handoff["study_report"]["path"])
    artifact_path = Path(handoff["study_artifact_receipt"]["path"])
    report = _read_object(report_path, "study report")
    artifact = _read_object(artifact_path, "study artifact receipt")
    selection = select_pareto_finalists(
        report,
        study_artifact_receipt=artifact,
        report_bytes=report_path.read_bytes(),
        expected_compiler_identity=dict(compiler_identity),
    )
    candidate_ids = rank_pareto_finalists(selection)[
        : int(handoff["strong_candidate_count"])
    ]
    if len(candidate_ids) != int(handoff["strong_candidate_count"]):
        raise AdaptiveCausalPreparationError("strong-candidate inventory is incomplete")
    finalists = {str(item["trial_id"]): dict(item) for item in selection["finalists"]}
    judge_raw = _read_object(
        Path(handoff["judge_budget_receipt"]["path"]), "judge budget receipt"
    )
    try:
        judge_receipt = parse_production_judge_budget_receipt(judge_raw)
    except ProductionJudgeBudgetError as error:
        raise AdaptiveCausalPreparationError("judge budget receipt is invalid") from error
    ledger_cursor = str(judge_receipt["content_sha256"])
    deadline = datetime.fromisoformat(
        str(handoff["deadline_utc"]).removesuffix("Z") + "+00:00"
    ).astimezone(timezone.utc)
    root = (
        Path(causal_root).resolve()
        if causal_root is not None
        else Path(handoff["output_root"]) / "causal"
    )
    receipt_paths: dict[str, Path] = {}
    committed: list[dict[str, Any]] = []

    for trial_id in candidate_ids:
        finalist = finalists[trial_id]
        candidate_root = root / trial_id
        receipt_path = candidate_root / "receipt.json"
        if receipt_path.exists():
            opened = open_causal_activation_control_receipt(
                receipt_path,
                expected_study_identity_sha256=str(handoff["study_identity_sha256"]),
                expected_trial_id=trial_id,
                expected_proposal_sha256=str(finalist["proposal_sha256"]),
            )
            if opened["judge_ledger_before_sha256"] != ledger_cursor:
                raise AdaptiveCausalPreparationError(
                    "stored causal judge ledger does not continue from authoritative evidence"
                )
            ledger_cursor = str(opened["judge_ledger_after_sha256"])
            leftover_checkpoint = candidate_root / "artifacts/checkpoint"
            if leftover_checkpoint.exists() or leftover_checkpoint.is_symlink():
                if leftover_checkpoint.is_symlink() or not leftover_checkpoint.is_dir():
                    raise AdaptiveCausalPreparationError(
                        "leftover candidate checkpoint cannot be retired safely"
                    )
                shutil.rmtree(leftover_checkpoint)
            receipt_paths[trial_id] = receipt_path.resolve(strict=True)
            event = {
                "format": "truth_editing_adaptive_causal_candidate_commit_v1",
                "trial_id": trial_id,
                "ordinal": len(committed),
                "candidate_count": len(candidate_ids),
                "receipt_path": str(receipt_paths[trial_id]),
                "receipt_sha256": _sha_file(receipt_paths[trial_id]),
                "receipt_self_sha256": opened["self_sha256"],
                "judge_ledger_after_sha256": ledger_cursor,
            }
            committed.append(event)
            if after_candidate_commit is not None:
                after_candidate_commit(dict(event))
            continue
        if clock().astimezone(timezone.utc) >= deadline:
            raise AdaptiveCausalPreparationError(
                "finalization deadline reached before causal candidate"
            )
        if before_candidate_execute is not None:
            before_candidate_execute(trial_id)
        artifacts = _materialized(
            materializer.materialize_candidate(
                study_identity_sha256=str(handoff["study_identity_sha256"]),
                trial_id=trial_id,
                proposal=dict(finalist["proposal"]),
                proposal_sha256=str(finalist["proposal_sha256"]),
                output_dir=candidate_root / "artifacts",
            )
        )
        config = build_causal_backend_config(
            edited_checkpoint_path=artifacts["edited_checkpoint_path"],
            edited_checkpoint_sha256=artifacts["edited_checkpoint_sha256"],
            edited_checkpoint_manifest_path=artifacts[
                "edited_checkpoint_manifest_path"
            ],
            basis_artifact_path=artifacts["basis_artifact_path"],
            output_dir=candidate_root / "runtime",
            judge_ledger_start_sha256=ledger_cursor,
        )
        config_path = candidate_root / "backend-config.json"
        _write_new_or_same(config_path, config)
        controls = []
        for control in artifacts["controls"]:
            if not isinstance(control, Mapping):
                raise AdaptiveCausalPreparationError("causal control must be an object")
            normalized = dict(control)
            if "activation_recipe_path" in normalized:
                normalized["activation_recipe_path"] = str(
                    normalized["activation_recipe_path"]
                )
            controls.append(normalized)
        plan = build_causal_activation_control_plan(
            study_identity_sha256=str(handoff["study_identity_sha256"]),
            trial_id=trial_id,
            proposal_sha256=str(finalist["proposal_sha256"]),
            persistent_recipe_path=artifacts["persistent_recipe_path"],
            scenario_path=artifacts["scenario_path"],
            evaluator_path=artifacts["evaluator_path"],
            runtime_identity_sha256=artifacts["runtime_identity_sha256"],
            direction_manifest_path=artifacts["direction_manifest_path"],
            controls=controls,
        )
        plan_path = candidate_root / "plan.json"
        _write_new_or_same(plan_path, plan)
        executor = executor_factory(config_path=config_path)
        close_executor = getattr(executor, "close", None)
        if not callable(close_executor):
            raise AdaptiveCausalPreparationError(
                "causal executor must expose an explicit GPU-release close method"
            )
        try:
            run_causal_activation_controls(plan_path, executor, receipt_path)
        finally:
            close_executor()
        opened = open_causal_activation_control_receipt(
            receipt_path,
            expected_study_identity_sha256=str(handoff["study_identity_sha256"]),
            expected_trial_id=trial_id,
            expected_proposal_sha256=str(finalist["proposal_sha256"]),
        )
        if opened["judge_ledger_before_sha256"] != ledger_cursor:
            raise AdaptiveCausalPreparationError(
                "causal judge ledger does not continue from authoritative evidence"
            )
        ledger_cursor = str(opened["judge_ledger_after_sha256"])
        checkpoint_path = Path(str(artifacts["edited_checkpoint_path"])).resolve()
        artifact_root = (candidate_root / "artifacts").resolve()
        try:
            checkpoint_path.relative_to(artifact_root)
        except ValueError as error:
            raise AdaptiveCausalPreparationError(
                "candidate checkpoint is outside its scoped artifact root"
            ) from error
        if checkpoint_path.is_symlink() or not checkpoint_path.is_dir():
            raise AdaptiveCausalPreparationError(
                "candidate checkpoint cannot be retired safely"
            )
        shutil.rmtree(checkpoint_path)
        receipt_paths[trial_id] = receipt_path.resolve(strict=True)
        event = {
            "format": "truth_editing_adaptive_causal_candidate_commit_v1",
            "trial_id": trial_id,
            "ordinal": len(committed),
            "candidate_count": len(candidate_ids),
            "receipt_path": str(receipt_paths[trial_id]),
            "receipt_sha256": _sha_file(receipt_paths[trial_id]),
            "receipt_self_sha256": opened["self_sha256"],
            "judge_ledger_after_sha256": ledger_cursor,
        }
        committed.append(event)
        if after_candidate_commit is not None:
            after_candidate_commit(dict(event))

    unsigned = {
        "format": PREPARATION_RECEIPT_FORMAT,
        "handoff_sha256": handoff["self_sha256"],
        "materializer_identity": dict(materializer.identity),
        "compiler_identity_sha256": _sha(dict(compiler_identity)),
        "candidate_trial_ids": candidate_ids,
        "candidate_commits": committed,
        "starting_judge_ledger_sha256": judge_receipt["content_sha256"],
        "ending_judge_ledger_sha256": ledger_cursor,
    }
    preparation = {**unsigned, "self_sha256": _sha(unsigned)}
    _write_new_or_same(root / "preparation-receipt.json", preparation)
    return receipt_paths


__all__ = [
    "AdaptiveCausalPreparationError",
    "CausalCandidateMaterializer",
    "CausalExecutorFactory",
    "PREPARATION_RECEIPT_FORMAT",
    "prepare_adaptive_causal_controls",
]
