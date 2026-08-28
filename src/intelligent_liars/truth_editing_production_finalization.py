"""Concrete hash-bound executor for the reserved adaptive finalization lane.

The coordinator owns ordering, budgets, durable evidence, and final selection.
This module owns the smaller production seam: resolve a finalist proposal,
derive a fresh execution/cache identity from the complete repeat or control
request, and invoke the real production evaluation adapter.  An exact retry
therefore reuses its paid judge call, while distinct repeats and controls can
never alias one another's semantic evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, Protocol

from .truth_editing_study import SearchProposal
from .truth_editing_production import (
    ProductionStudyEvaluator,
    open_finalist_export_inputs,
)
from .truth_editing_finalist_checkpoint import (
    export_finalist_checkpoint,
    open_finalist_checkpoint,
)
from .truth_editing_study import EvaluationResult
from .truth_editing_production_judge_budget import ProductionJudgeBudget


class ProductionFinalizationError(RuntimeError):
    """A finalization request cannot be bound to production evidence."""


class ProductionFinalizationBackend(Protocol):
    """Public seam implemented by the production evaluator/runtime composition."""

    @property
    def identity(self) -> Mapping[str, Any]: ...

    @property
    def compiler_identity(self) -> Mapping[str, Any]: ...

    def estimate_evaluation_cost_usd(self, request: Mapping[str, Any]) -> Decimal: ...

    def evaluate_finalization(
        self,
        proposal: SearchProposal,
        *,
        request: Mapping[str, Any],
        execution_identity_sha256: str,
        control_kind: Literal["orthogonal", "shuffled"] | None,
    ) -> Mapping[str, Any]: ...

    def export_finalist(
        self,
        *,
        selection_receipt: Mapping[str, Any],
        trial_id: str,
        output_dir: Path,
    ) -> Mapping[str, Any]: ...


class FinalizationCostMeter(Protocol):
    """Measure paid evaluation cost around one exact production call."""

    def estimate_cost_usd(self, request: Mapping[str, Any]) -> Decimal: ...

    def measure(
        self,
        execution_identity_sha256: str,
        action: Callable[[], EvaluationResult],
    ) -> tuple[EvaluationResult, Decimal]: ...


class FinalistCheckpointExporter(Protocol):
    @property
    def compiler_identity(self) -> Mapping[str, Any]: ...

    def export_finalist(
        self,
        *,
        selection_receipt: Mapping[str, Any],
        trial_id: str,
        output_dir: Path,
    ) -> Mapping[str, Any]: ...


class ProductionJudgeLedgerCostMeter:
    """Measure exact paid-judge deltas from the authoritative shared ledger."""

    def __init__(
        self,
        ledger: ProductionJudgeBudget,
        *,
        maximum_evaluation_cost_usd: Decimal | str,
    ) -> None:
        self._ledger = ledger
        self._maximum_cost = _money(
            maximum_evaluation_cost_usd, "maximum finalization evaluation cost"
        )
        if self._maximum_cost <= 0:
            raise ProductionFinalizationError(
                "maximum finalization evaluation cost must be positive"
            )

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "adapter": "production_judge_ledger_finalization_cost_meter_v1",
            "budget_config_sha256": self._ledger.config.identity_sha256,
            "maximum_evaluation_cost_usd": format(self._maximum_cost, "f"),
        }

    def estimate_cost_usd(self, request: Mapping[str, Any]) -> Decimal:
        del request
        return self._maximum_cost

    def measure(
        self,
        execution_identity_sha256: str,
        action: Callable[[], EvaluationResult],
    ) -> tuple[EvaluationResult, Decimal]:
        _digest(execution_identity_sha256, "cost measurement execution identity")
        before = _money(
            self._ledger.receipt()["actual_spend_usd"],
            "judge spend before evaluation",
        )
        result = action()
        after = _money(
            self._ledger.receipt()["actual_spend_usd"],
            "judge spend after evaluation",
        )
        if after < before:
            raise ProductionFinalizationError("judge ledger spend regressed")
        return result, after - before


class ProductionFinalistCheckpointExporter:
    """Verified deployment-checkpoint exporter bound to one production study."""

    def __init__(
        self,
        *,
        production_config_path: Path | str,
        study_artifact_receipt_path: Path | str,
        study_identity_sha256: str,
        study_artifact_receipt_sha256: str,
        registry_bucket: str,
        model_slug: str,
        compiler: Any | None = None,
        bundle: Any | None = None,
    ) -> None:
        self._config_path = Path(production_config_path)
        self._receipt_path = Path(study_artifact_receipt_path)
        self._study_identity = _digest(study_identity_sha256, "study identity")
        self._receipt_sha = _digest(
            study_artifact_receipt_sha256, "study artifact receipt identity"
        )
        if not registry_bucket or not model_slug:
            raise ProductionFinalizationError(
                "checkpoint registry bucket and model slug are required"
            )
        self._registry_bucket = registry_bucket
        self._model_slug = model_slug
        if (compiler is None) != (bundle is None):
            raise ProductionFinalizationError(
                "preloaded finalist compiler and bundle must be supplied together"
            )
        self._loaded: tuple[Any, Any] | None = (
            (compiler, bundle) if compiler is not None and bundle is not None else None
        )

    def _inputs(self) -> tuple[Any, Any]:
        if self._loaded is None:
            self._loaded = open_finalist_export_inputs(
                self._config_path,
                study_artifact_receipt_path=self._receipt_path,
                expected_study_identity_sha256=self._study_identity,
                expected_study_artifact_receipt_sha256=self._receipt_sha,
            )
        return self._loaded

    @property
    def compiler_identity(self) -> Mapping[str, Any]:
        compiler, _bundle = self._inputs()
        return dict(compiler.identity)

    def export_finalist(
        self,
        *,
        selection_receipt: Mapping[str, Any],
        trial_id: str,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        compiler, bundle = self._inputs()
        exported = export_finalist_checkpoint(
            selection_receipt=selection_receipt,
            trial_id=trial_id,
            compiler=compiler,
            bundle=bundle,
            output_dir=output_dir,
            registry_bucket=self._registry_bucket,
            model_slug=self._model_slug,
        )
        reopened = open_finalist_checkpoint(output_dir)
        if dict(exported) != reopened:
            raise ProductionFinalizationError(
                "reopened checkpoint differs from exported checkpoint"
            )
        return reopened

_HEX = frozenset("0123456789abcdef")
_OBJECTIVES = frozenset(
    {
        "valid_false_report_rate_lcb",
        "truth_report_dissociation_lcb",
        "capability_preservation_lcb",
    }
)
_TARGET_EFFECT_VIOLATIONS = frozenset(
    {
        "direct.every_response_valid",
        "direct.valid_false_report_rate",
        "broad_qa.every_false_report_valid",
        "broad_qa.missing_weighted_lane",
    }
)


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
        raise ProductionFinalizationError("value is not canonical JSON") from error


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ProductionFinalizationError(f"{label} must be a lowercase SHA-256")
    return value


def _money(value: Any, label: str) -> Decimal:
    if not isinstance(value, (str, Decimal)) or isinstance(value, bool):
        raise ProductionFinalizationError(f"{label} must be an exact decimal")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ProductionFinalizationError(f"{label} is invalid") from error
    if not parsed.is_finite() or parsed < 0:
        raise ProductionFinalizationError(f"{label} must be finite and nonnegative")
    return parsed


def _read_report(path: Path) -> tuple[str, dict[str, SearchProposal], str]:
    if path.is_symlink() or not path.is_file():
        raise ProductionFinalizationError("study report must be a regular file")
    data = path.read_bytes()
    try:
        raw = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProductionFinalizationError("study report is invalid JSON") from error
    if not isinstance(raw, Mapping):
        raise ProductionFinalizationError("study report must be an object")
    study_identity = _digest(raw.get("study_identity_sha256"), "study identity")
    trials = raw.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ProductionFinalizationError("study report has no trials")
    proposals: dict[str, SearchProposal] = {}
    for index, row in enumerate(trials):
        if not isinstance(row, Mapping):
            raise ProductionFinalizationError(f"study trial {index} is not an object")
        trial_id = row.get("trial_id")
        proposal_raw = row.get("proposal")
        if not isinstance(trial_id, str) or not trial_id or trial_id in proposals:
            raise ProductionFinalizationError("study trial identity is invalid")
        if not isinstance(proposal_raw, Mapping):
            raise ProductionFinalizationError("study trial proposal is invalid")
        try:
            proposals[trial_id] = SearchProposal.from_dict(proposal_raw)
        except Exception as error:
            raise ProductionFinalizationError("study trial proposal is invalid") from error
    return study_identity, proposals, hashlib.sha256(data).hexdigest()


class ProductionEvaluatorFinalizationBackend:
    """Real production evaluator/runtime adapter for reserved finalization work.

    This adapter does not recreate prompts, model loading, preservation, or
    judge logic. It crosses the same public ``ProductionStudyEvaluator`` seam
    as routine search and adds only explicit finalization identity, matched
    control kind, cost measurement, and verified checkpoint export.
    """

    def __init__(
        self,
        *,
        evaluator: ProductionStudyEvaluator,
        finalist_record_ids: tuple[str, ...],
        cost_meter: FinalizationCostMeter,
        checkpoint_exporter: FinalistCheckpointExporter,
    ) -> None:
        if not finalist_record_ids or len(set(finalist_record_ids)) != len(
            finalist_record_ids
        ):
            raise ProductionFinalizationError(
                "finalist record IDs must be nonempty and unique"
            )
        self._evaluator = evaluator
        self._record_ids = tuple(finalist_record_ids)
        self._cost_meter = cost_meter
        self._exporter = checkpoint_exporter

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "adapter": "production_evaluator_finalization_backend_v1",
            "evaluator": dict(self._evaluator.identity),
            "finalist_record_ids_sha256": _sha(list(self._record_ids)),
            "cost_meter": dict(getattr(self._cost_meter, "identity", {})),
        }

    @property
    def compiler_identity(self) -> Mapping[str, Any]:
        return dict(self._exporter.compiler_identity)

    def estimate_evaluation_cost_usd(self, request: Mapping[str, Any]) -> Decimal:
        return _money(
            self._cost_meter.estimate_cost_usd(request),
            "production evaluation cost estimate",
        )

    def evaluate_finalization(
        self,
        proposal: SearchProposal,
        *,
        request: Mapping[str, Any],
        execution_identity_sha256: str,
        control_kind: Literal["orthogonal", "shuffled"] | None,
    ) -> Mapping[str, Any]:
        execution_identity = _digest(
            execution_identity_sha256, "finalization execution identity"
        )

        def evaluate() -> EvaluationResult:
            return self._evaluator.evaluate(
                proposal,
                trial_id=f"finalization-{execution_identity[:24]}",
                record_ids=self._record_ids,
                objective_names=tuple(sorted(_OBJECTIVES)),
                finalization_execution_identity_sha256=execution_identity,
                control_kind=control_kind,
            )

        result, actual_cost = self._cost_meter.measure(execution_identity, evaluate)
        actual = _money(actual_cost, "measured production evaluation cost")
        if result.outcome_kind == "operational_failure":
            raise ProductionFinalizationError(
                f"production finalization evaluation failed: {result.detail}"
            )
        if set(result.metrics) != _OBJECTIVES:
            raise ProductionFinalizationError(
                "production finalization objectives differ"
            )
        assessment = self._evaluator.last_assessment
        if assessment is None:
            raise ProductionFinalizationError(
                "production evaluator omitted the frozen assessment"
            )
        artifact_value = self._evaluator.last_runtime_artifact_path
        if artifact_value is None:
            raise ProductionFinalizationError(
                "production evaluator omitted its runtime artifact"
            )
        artifact = Path(artifact_value).resolve(strict=True)
        if artifact.is_symlink() or not artifact.is_file():
            raise ProductionFinalizationError(
                "production runtime artifact is not a regular file"
            )
        violations = set(assessment.constraint_violations)
        non_target_violations = violations - _TARGET_EFFECT_VIOLATIONS
        if control_kind is None:
            hard_gates_passed = result.outcome_kind == "success"
            supports_targeted_effect = False
        else:
            # A valid matched control must remain otherwise healthy while
            # failing specifically to reproduce the targeted false-report
            # behavior. Capability, retained-truth, truthful-control, and
            # false-trigger failures are never accepted as causal support.
            hard_gates_passed = not non_target_violations
            supports_targeted_effect = bool(
                violations & _TARGET_EFFECT_VIOLATIONS
            ) and hard_gates_passed
        return {
            "metrics": dict(result.metrics),
            "hard_gates_passed": hard_gates_passed,
            "supports_targeted_effect": supports_targeted_effect,
            "actual_evaluation_cost_usd": format(actual, "f"),
            "artifact_path": str(artifact),
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }

    def export_finalist(
        self,
        *,
        selection_receipt: Mapping[str, Any],
        trial_id: str,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        return self._exporter.export_finalist(
            selection_receipt=selection_receipt,
            trial_id=trial_id,
            output_dir=output_dir,
        )


class ProductionAdaptiveFinalizationExecutor:
    """Bind coordinator requests to fresh production evaluation identities."""

    def __init__(
        self,
        study_report_path: Path | str,
        backend: ProductionFinalizationBackend,
        *,
        causal_control_receipts: Mapping[str, Path | str] | None = None,
    ) -> None:
        self._report_path = Path(study_report_path).resolve(strict=True)
        (
            self._study_identity_sha256,
            self._proposals,
            self._study_report_sha256,
        ) = _read_report(self._report_path)
        self._backend = backend
        self._backend_identity = dict(backend.identity)
        self._compiler_identity = dict(backend.compiler_identity)
        self._causal_control_receipts = {
            trial_id: Path(path).resolve(strict=True)
            for trial_id, path in dict(causal_control_receipts or {}).items()
        }
        if any(
            trial_id not in self._proposals
            or path.is_symlink()
            or not path.is_file()
            for trial_id, path in self._causal_control_receipts.items()
        ):
            raise ProductionFinalizationError(
                "causal activation control receipt inventory is invalid"
            )
        _canonical(self._backend_identity)
        _canonical(self._compiler_identity)

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "adapter": "production_adaptive_finalization_executor_v1",
            "study_identity_sha256": self._study_identity_sha256,
            "study_report_sha256": self._study_report_sha256,
            "backend": dict(self._backend_identity),
            "causal_control_receipt_sha256_by_trial": {
                trial_id: hashlib.sha256(path.read_bytes()).hexdigest()
                for trial_id, path in sorted(self._causal_control_receipts.items())
            },
        }

    @property
    def compiler_identity(self) -> Mapping[str, Any]:
        return dict(self._compiler_identity)

    def estimate_repeat_cost_usd(self, request: Mapping[str, Any]) -> Decimal:
        self._validated_request(request, kind="repeat")
        return _money(
            self._backend.estimate_evaluation_cost_usd(request),
            "repeat cost estimate",
        )

    def estimate_control_cost_usd(self, request: Mapping[str, Any]) -> Decimal:
        self._validated_request(request, kind="control")
        return _money(
            self._backend.estimate_evaluation_cost_usd(request),
            "control cost estimate",
        )

    def run_repeat(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        proposal, execution_identity = self._validated_request(request, kind="repeat")
        raw = self._backend.evaluate_finalization(
            proposal,
            request=dict(request),
            execution_identity_sha256=execution_identity,
            control_kind=None,
        )
        return self._result(raw, control=False)

    def run_control(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        proposal, execution_identity = self._validated_request(request, kind="control")
        kind = request["control_kind"]
        assert kind in {"orthogonal", "shuffled"}
        raw = self._backend.evaluate_finalization(
            proposal,
            request=dict(request),
            execution_identity_sha256=execution_identity,
            control_kind=kind,
        )
        return self._result(raw, control=True)

    def export_finalist(
        self,
        *,
        selection_receipt: Mapping[str, Any],
        trial_id: str,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        if trial_id not in self._proposals:
            raise ProductionFinalizationError("export trial is absent from study report")
        self.verify_causal_control(
            trial_id=trial_id,
            proposal_sha256=_sha(self._proposals[trial_id].to_dict()),
        )
        return self._backend.export_finalist(
            selection_receipt=selection_receipt,
            trial_id=trial_id,
            output_dir=output_dir,
        )

    def verify_causal_control(
        self, *, trial_id: str, proposal_sha256: str
    ) -> str:
        if trial_id not in self._proposals:
            raise ProductionFinalizationError(
                "causal control trial is absent from study report"
            )
        expected_proposal = _sha(self._proposals[trial_id].to_dict())
        if proposal_sha256 != expected_proposal:
            raise ProductionFinalizationError(
                "causal control proposal binding differs"
            )
        receipt_path = self._causal_control_receipts.get(trial_id)
        if receipt_path is None:
            raise ProductionFinalizationError(
                "full scientific finalization requires executed causal activation controls"
            )
        try:
            from .truth_editing_causal_activation_controls import (
                open_causal_activation_control_receipt,
            )

            opened = open_causal_activation_control_receipt(
                receipt_path,
                expected_study_identity_sha256=self._study_identity_sha256,
                expected_trial_id=trial_id,
                expected_proposal_sha256=_sha(self._proposals[trial_id].to_dict()),
            )
        except Exception as error:
            raise ProductionFinalizationError(
                f"causal activation control evidence is invalid: {error}"
            ) from error
        return _digest(opened["self_sha256"], "causal control receipt self identity")

    def causal_control_budget_summary(
        self,
        *,
        trial_ids: tuple[str, ...],
        expected_starting_judge_ledger_sha256: str,
    ) -> Mapping[str, Any]:
        """Verify the pre-executed causal lane's exact spend and ledger chain."""

        starting = _digest(
            expected_starting_judge_ledger_sha256,
            "starting causal judge ledger identity",
        )
        if not trial_ids or len(set(trial_ids)) != len(trial_ids):
            raise ProductionFinalizationError(
                "causal budget trial inventory must be nonempty and unique"
            )
        try:
            from .truth_editing_causal_activation_controls import (
                open_causal_activation_control_receipt,
            )

            receipts = []
            for trial_id in trial_ids:
                proposal = self._proposals.get(trial_id)
                path = self._causal_control_receipts.get(trial_id)
                if proposal is None or path is None:
                    raise ProductionFinalizationError(
                        "causal budget inventory is incomplete"
                    )
                receipts.append(
                    open_causal_activation_control_receipt(
                        path,
                        expected_study_identity_sha256=self._study_identity_sha256,
                        expected_trial_id=trial_id,
                        expected_proposal_sha256=_sha(proposal.to_dict()),
                    )
                )
        except ProductionFinalizationError:
            raise
        except Exception as error:
            raise ProductionFinalizationError(
                f"causal activation budget evidence is invalid: {error}"
            ) from error
        by_before = {
            item["judge_ledger_before_sha256"]: item for item in receipts
        }
        if len(by_before) != len(receipts):
            raise ProductionFinalizationError(
                "causal judge ledger chain has duplicate starting identities"
            )
        ordered = []
        cursor = starting
        while cursor in by_before:
            item = by_before.pop(cursor)
            ordered.append(item)
            cursor = item["judge_ledger_after_sha256"]
        if by_before or len(ordered) != len(receipts):
            raise ProductionFinalizationError(
                "causal judge ledger chain does not start at the authoritative receipt"
            )
        total_cost = sum(
            (
                _money(item["actual_evaluation_cost_usd"], "causal receipt cost")
                for item in ordered
            ),
            Decimal("0"),
        )
        return {
            "format": "truth_editing_causal_activation_budget_summary_v1",
            "actual_evaluation_cost_usd": format(total_cost, "f"),
            "judge_call_count": sum(item["judge_call_count"] for item in ordered),
            "judge_ledger_before_sha256": starting,
            "judge_ledger_after_sha256": cursor,
            "receipt_self_sha256s": [item["self_sha256"] for item in ordered],
        }

    def _validated_request(
        self,
        request: Mapping[str, Any],
        *,
        kind: Literal["repeat", "control"],
    ) -> tuple[SearchProposal, str]:
        if not isinstance(request, Mapping):
            raise ProductionFinalizationError(f"{kind} request must be an object")
        expected = (
            {
                "study_identity_sha256",
                "trial_id",
                "proposal_sha256",
                "repeat_index",
                "selection_receipt_sha256",
                "request_id",
            }
            if kind == "repeat"
            else {
                "study_identity_sha256",
                "trial_id",
                "proposal_sha256",
                "control_id",
                "control_kind",
                "direction_ids",
                "source_layer",
                "requested_rank",
                "writer_layers",
                "writer_strength_plan_sha256",
                "selection_receipt_sha256",
                "request_id",
            }
        )
        if set(request) != expected:
            raise ProductionFinalizationError(f"{kind} request fields differ")
        if request["study_identity_sha256"] != self._study_identity_sha256:
            raise ProductionFinalizationError("request study identity differs")
        trial_id = request["trial_id"]
        if not isinstance(trial_id, str) or trial_id not in self._proposals:
            raise ProductionFinalizationError("request trial is absent from study report")
        proposal = self._proposals[trial_id]
        if request["proposal_sha256"] != _sha(proposal.to_dict()):
            raise ProductionFinalizationError("request proposal binding differs")
        _digest(request["selection_receipt_sha256"], "selection receipt identity")
        body = dict(request)
        request_id = body.pop("request_id")
        if request_id != f"{kind}-{_sha(body)[:24]}":
            raise ProductionFinalizationError(f"{kind} request identity differs")
        if kind == "repeat":
            repeat_index = request["repeat_index"]
            if isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or repeat_index < 0:
                raise ProductionFinalizationError("repeat index is invalid")
        elif request["control_kind"] not in {"orthogonal", "shuffled"}:
            raise ProductionFinalizationError("control kind is unsupported")
        elif (
            request["direction_ids"] != list(proposal.direction_ids)
            or request["source_layer"] != proposal.source_layer
            or request["requested_rank"] != proposal.requested_rank
            or request["writer_layers"] != list(proposal.writer_layers)
            or request["writer_strength_plan_sha256"]
            != _sha(proposal.writer_strength_plan())
        ):
            raise ProductionFinalizationError(
                "control geometry differs from its parent proposal"
            )
        # This is also the judge-cache namespace. It includes every execution
        # field, so only an exact retry can reuse a paid semantic judgment.
        execution_identity = _sha(
            {
                "format": "truth_editing_finalization_execution_identity_v1",
                "kind": kind,
                "request": dict(request),
                "backend_identity": self._backend_identity,
            }
        )
        return proposal, execution_identity

    @staticmethod
    def _result(raw: Mapping[str, Any], *, control: bool) -> Mapping[str, Any]:
        if not isinstance(raw, Mapping):
            raise ProductionFinalizationError("production evaluation result is not an object")
        common = {
            "metrics",
            "hard_gates_passed",
            "supports_targeted_effect",
            "actual_evaluation_cost_usd",
            "artifact_path",
            "artifact_sha256",
        }
        if set(raw) != common:
            raise ProductionFinalizationError("production evaluation result fields differ")
        if not isinstance(raw["hard_gates_passed"], bool):
            raise ProductionFinalizationError("production hard-gate result is invalid")
        if not isinstance(raw["supports_targeted_effect"], bool):
            raise ProductionFinalizationError("production control result is invalid")
        metrics = raw["metrics"]
        if not isinstance(metrics, Mapping) or set(metrics) != _OBJECTIVES:
            raise ProductionFinalizationError("production objective metrics differ")
        _money(raw["actual_evaluation_cost_usd"], "actual evaluation cost")
        artifact_path = Path(str(raw["artifact_path"])).resolve(strict=True)
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise ProductionFinalizationError("production evidence artifact is not a regular file")
        artifact_sha = _digest(raw["artifact_sha256"], "production artifact SHA-256")
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != artifact_sha:
            raise ProductionFinalizationError("production evidence artifact identity differs")
        result = {
            "hard_gates_passed": raw["hard_gates_passed"],
            "actual_evaluation_cost_usd": str(raw["actual_evaluation_cost_usd"]),
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha,
        }
        if control:
            result["supports_targeted_effect"] = raw["supports_targeted_effect"]
        else:
            result["metrics"] = dict(metrics)
        return result


__all__ = [
    "ProductionAdaptiveFinalizationExecutor",
    "ProductionEvaluatorFinalizationBackend",
    "FinalizationCostMeter",
    "FinalistCheckpointExporter",
    "ProductionJudgeLedgerCostMeter",
    "ProductionFinalistCheckpointExporter",
    "ProductionFinalizationBackend",
    "ProductionFinalizationError",
]
