"""Durable post-search repeats, controls, selection, and checkpoint export."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from .heretic_truth_editing import OBJECTIVES
from .truth_editing_finalist_checkpoint import (
    finalize_audited_selection,
    open_finalist_checkpoint,
    rank_pareto_finalists,
    select_pareto_finalists,
)
from .truth_editing_production_judge_budget import (
    ProductionJudgeBudgetError,
    parse_production_judge_budget_receipt,
)


HANDOFF_FORMAT = "truth_editing_adaptive_finalization_handoff_v1"
AUDIT_FORMAT = "truth_editing_adaptive_finalization_audit_v1"
RECEIPT_FORMAT = "truth_editing_adaptive_finalization_receipt_v1"
_HEX = frozenset("0123456789abcdef")


class AdaptiveFinalizationError(RuntimeError):
    """Finalization cannot continue without trustworthy scientific evidence."""


class AdaptiveFinalizationExecutor(Protocol):
    """Scientific adapter used after search workers have closed."""

    @property
    def identity(self) -> Mapping[str, Any]: ...

    @property
    def compiler_identity(self) -> Mapping[str, Any]: ...

    def estimate_repeat_cost_usd(self, request: Mapping[str, Any]) -> Decimal: ...

    def run_repeat(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def estimate_control_cost_usd(self, request: Mapping[str, Any]) -> Decimal: ...

    def run_control(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def verify_causal_control(
        self, *, trial_id: str, proposal_sha256: str
    ) -> str: ...

    def causal_control_budget_summary(
        self,
        *,
        trial_ids: tuple[str, ...],
        expected_starting_judge_ledger_sha256: str,
    ) -> Mapping[str, Any]: ...

    def export_finalist(
        self,
        *,
        selection_receipt: Mapping[str, Any],
        trial_id: str,
        output_dir: Path,
    ) -> Mapping[str, Any]: ...


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
        raise AdaptiveFinalizationError("value is not canonical JSON") from error


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
        or any(character not in _HEX for character in value)
    ):
        raise AdaptiveFinalizationError(f"{label} must be a lowercase SHA-256")
    return value


def _money(value: Any, label: str) -> Decimal:
    if not isinstance(value, (str, Decimal)) or isinstance(value, bool):
        raise AdaptiveFinalizationError(f"{label} must be an exact decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise AdaptiveFinalizationError(f"{label} is invalid") from error
    if not result.is_finite() or result < 0:
        raise AdaptiveFinalizationError(f"{label} must be finite and nonnegative")
    return result


def _money_text(value: Decimal) -> str:
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _load_regular(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise AdaptiveFinalizationError(f"{label} is not a regular file")
    data = path.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AdaptiveFinalizationError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise AdaptiveFinalizationError(f"{label} must be an object")
    return value, data


def _source(path: Path, label: str) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    _load_regular(resolved, label)
    return {"path": str(resolved), "sha256": _sha_file(resolved)}


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AdaptiveFinalizationError(f"{label} must be a UTC Z timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise AdaptiveFinalizationError(f"{label} is invalid") from error
    return parsed.astimezone(timezone.utc)


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    data = json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        if path.is_file() and not path.is_symlink() and path.read_bytes() == data:
            return
        raise AdaptiveFinalizationError(f"immutable output differs: {path}")


def write_adaptive_finalization_handoff(
    destination: Path | str,
    *,
    study_report_path: Path | str,
    study_artifact_receipt_path: Path | str,
    production_config_path: Path | str,
    adaptive_checkpoint_path: Path | str,
    judge_budget_receipt_path: Path | str,
    output_root: Path | str,
    deadline_utc: str,
    study_identity_sha256: str,
    maximum_evaluation_spend_usd: str = "1",
    strong_candidate_count: int = 3,
    repeat_count_per_candidate: int = 2,
) -> dict[str, Any]:
    """Write the immutable controller-to-finalizer handoff."""

    study_sha = _digest(study_identity_sha256, "study_identity_sha256")
    if strong_candidate_count != 3 or repeat_count_per_candidate != 2:
        raise AdaptiveFinalizationError("finalization candidate/repeat policy differs")
    if _money(maximum_evaluation_spend_usd, "maximum evaluation spend") > Decimal("1"):
        raise AdaptiveFinalizationError("finalization evaluation spend exceeds $1")
    deadline = _utc(deadline_utc, "deadline_utc")
    report_source = _source(Path(study_report_path), "study report")
    receipt_source = _source(Path(study_artifact_receipt_path), "study receipt")
    report, report_bytes = _load_regular(Path(report_source["path"]), "study report")
    artifact, _ = _load_regular(Path(receipt_source["path"]), "study receipt")
    selection = select_pareto_finalists(
        report, study_artifact_receipt=artifact, report_bytes=report_bytes
    )
    if selection["study_identity_sha256"] != study_sha:
        raise AdaptiveFinalizationError("study identity differs from report")
    checkpoint_source = _source(Path(adaptive_checkpoint_path), "adaptive checkpoint")
    checkpoint, _ = _load_regular(
        Path(checkpoint_source["path"]), "adaptive checkpoint"
    )
    if checkpoint.get("phase") != "finalization_reserved":
        raise AdaptiveFinalizationError("adaptive run is not in finalization_reserved")
    if checkpoint.get("study_identity_sha256") != study_sha:
        raise AdaptiveFinalizationError("adaptive checkpoint study identity differs")
    search_deadline = _utc(
        checkpoint.get("search_deadline_utc"), "adaptive search deadline"
    )
    hard_deadline = _utc(
        checkpoint.get("hard_deadline_utc"), "adaptive hard deadline"
    )
    if hard_deadline - search_deadline != timedelta(hours=3):
        raise AdaptiveFinalizationError("adaptive finalization reserve is not three hours")
    if deadline != hard_deadline:
        raise AdaptiveFinalizationError("handoff deadline differs from adaptive checkpoint")
    claimed_checkpoint = _digest(
        checkpoint.get("checkpoint_sha256"), "adaptive checkpoint SHA-256"
    )
    unsigned_checkpoint = dict(checkpoint)
    unsigned_checkpoint.pop("checkpoint_sha256")
    if claimed_checkpoint != _sha(unsigned_checkpoint):
        raise AdaptiveFinalizationError("adaptive checkpoint identity differs")
    unsigned = {
        "format": HANDOFF_FORMAT,
        "study_identity_sha256": study_sha,
        "study_report": report_source,
        "study_artifact_receipt": receipt_source,
        "production_config": _source(Path(production_config_path), "production config"),
        "adaptive_checkpoint": checkpoint_source,
        "judge_budget_receipt": _source(
            Path(judge_budget_receipt_path), "judge budget receipt"
        ),
        "output_root": str(Path(output_root).resolve()),
        "deadline_utc": deadline_utc,
        "maximum_evaluation_spend_usd": _money_text(
            _money(maximum_evaluation_spend_usd, "maximum evaluation spend")
        ),
        "strong_candidate_count": strong_candidate_count,
        "repeat_count_per_candidate": repeat_count_per_candidate,
    }
    handoff = {**unsigned, "self_sha256": _sha(unsigned)}
    _write_new(Path(destination), handoff)
    return handoff


def _open_handoff(path: Path) -> dict[str, Any]:
    raw, _ = _load_regular(path, "finalization handoff")
    fields = {
        "format", "study_identity_sha256", "study_report",
        "study_artifact_receipt", "production_config", "adaptive_checkpoint",
        "judge_budget_receipt", "output_root", "deadline_utc",
        "maximum_evaluation_spend_usd", "strong_candidate_count",
        "repeat_count_per_candidate", "self_sha256",
    }
    if set(raw) != fields or raw["format"] != HANDOFF_FORMAT:
        raise AdaptiveFinalizationError("finalization handoff fields or format differ")
    claimed = _digest(raw.pop("self_sha256"), "handoff self_sha256")
    if claimed != _sha(raw):
        raise AdaptiveFinalizationError("finalization handoff identity differs")
    for name in (
        "study_report", "study_artifact_receipt", "production_config",
        "adaptive_checkpoint", "judge_budget_receipt",
    ):
        source = raw[name]
        if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
            raise AdaptiveFinalizationError(f"handoff {name} source differs")
        path_value = Path(str(source["path"]))
        if _sha_file(path_value) != _digest(source["sha256"], f"{name} SHA-256"):
            raise AdaptiveFinalizationError(f"handoff {name} content differs")
    return {**raw, "self_sha256": claimed}


def open_adaptive_finalization_handoff(path: Path | str) -> dict[str, Any]:
    """Validate and reopen one immutable controller handoff."""

    return _open_handoff(Path(path))


def _request_id(kind: str, body: Mapping[str, Any]) -> str:
    return f"{kind}-{_sha(body)[:24]}"


def _result(
    value: Mapping[str, Any], *, request: Mapping[str, Any], kind: str,
    executor_sha256: str,
) -> dict[str, Any]:
    fields = (
        {"metrics", "hard_gates_passed", "actual_evaluation_cost_usd", "artifact_path", "artifact_sha256"}
        if kind == "repeat"
        else {"supports_targeted_effect", "hard_gates_passed", "actual_evaluation_cost_usd", "artifact_path", "artifact_sha256"}
    )
    if set(value) != fields:
        raise AdaptiveFinalizationError(f"{kind} evidence fields differ")
    if not isinstance(value["hard_gates_passed"], bool):
        raise AdaptiveFinalizationError(f"{kind} hard-gate result is invalid")
    if kind == "control" and not isinstance(value["supports_targeted_effect"], bool):
        raise AdaptiveFinalizationError("control targeted-effect result is invalid")
    if kind == "repeat":
        metrics = value["metrics"]
        if not isinstance(metrics, Mapping) or set(metrics) != set(OBJECTIVES):
            raise AdaptiveFinalizationError("repeat objective metrics differ")
        normalized_metrics = {name: float(metrics[name]) for name in OBJECTIVES}
        if not all(float("-inf") < item < float("inf") for item in normalized_metrics.values()):
            raise AdaptiveFinalizationError("repeat objective metric is not finite")
    else:
        normalized_metrics = None
    cost = _money(value["actual_evaluation_cost_usd"], f"{kind} actual cost")
    artifact_path = Path(str(value["artifact_path"])).resolve(strict=True)
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise AdaptiveFinalizationError(f"{kind} evidence artifact is not a regular file")
    artifact_sha = _digest(value["artifact_sha256"], "evidence artifact SHA-256")
    if _sha_file(artifact_path) != artifact_sha:
        raise AdaptiveFinalizationError(f"{kind} evidence artifact identity differs")
    body = {
        "format": f"truth_editing_adaptive_finalization_{kind}_evidence_v1",
        "request": dict(request),
        "executor_identity_sha256": executor_sha256,
        "metrics": normalized_metrics,
        "hard_gates_passed": value["hard_gates_passed"],
        "supports_targeted_effect": (
            value["supports_targeted_effect"] if kind == "control" else None
        ),
        "actual_evaluation_cost_usd": _money_text(cost),
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha,
    }
    return {**body, "self_sha256": _sha(body)}


def _verify_evidence(
    evidence: Mapping[str, Any], *, request: Mapping[str, Any], kind: str,
    executor_sha256: str,
) -> dict[str, Any]:
    expected = {
        "format", "request", "executor_identity_sha256", "metrics",
        "hard_gates_passed", "supports_targeted_effect",
        "actual_evaluation_cost_usd", "artifact_path", "artifact_sha256", "self_sha256",
    }
    if set(evidence) != expected:
        raise AdaptiveFinalizationError(f"stored {kind} evidence fields differ")
    raw = dict(evidence)
    claimed = _digest(raw.pop("self_sha256"), f"stored {kind} evidence SHA-256")
    if claimed != _sha(raw):
        raise AdaptiveFinalizationError(f"stored {kind} evidence identity differs")
    expected_format = f"truth_editing_adaptive_finalization_{kind}_evidence_v1"
    if (
        raw["format"] != expected_format
        or raw["request"] != request
        or raw["executor_identity_sha256"] != executor_sha256
    ):
        raise AdaptiveFinalizationError(f"stored {kind} evidence binding differs")
    result_value = {
        "hard_gates_passed": raw["hard_gates_passed"],
        "actual_evaluation_cost_usd": raw["actual_evaluation_cost_usd"],
        "artifact_path": raw["artifact_path"],
        "artifact_sha256": raw["artifact_sha256"],
    }
    if kind == "repeat":
        result_value["metrics"] = raw["metrics"]
    else:
        result_value["supports_targeted_effect"] = raw["supports_targeted_effect"]
    expected_receipt = _result(
        result_value, request=request, kind=kind, executor_sha256=executor_sha256
    )
    if expected_receipt != evidence:
        raise AdaptiveFinalizationError(f"stored {kind} evidence content differs")
    return dict(evidence)


def run_adaptive_finalization(
    handoff_path: Path | str,
    executor: AdaptiveFinalizationExecutor,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
    before_unit_execute: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute the reserved lane and publish a receipt only after verified export."""

    handoff = _open_handoff(Path(handoff_path))
    deadline = _utc(handoff["deadline_utc"], "deadline_utc")
    budget = _money(handoff["maximum_evaluation_spend_usd"], "evaluation budget")
    executor_identity = dict(executor.identity)
    compiler_identity = dict(executor.compiler_identity)
    executor_sha = _sha(executor_identity)
    report_path = Path(handoff["study_report"]["path"])
    receipt_path = Path(handoff["study_artifact_receipt"]["path"])
    report, report_bytes = _load_regular(report_path, "study report")
    artifact, _ = _load_regular(receipt_path, "study receipt")
    selection = select_pareto_finalists(
        report,
        study_artifact_receipt=artifact,
        report_bytes=report_bytes,
        expected_compiler_identity=compiler_identity,
    )
    candidate_ids = rank_pareto_finalists(selection)[: handoff["strong_candidate_count"]]
    finalists_by_id = {item["trial_id"]: item for item in selection["finalists"]}
    judge_budget_raw, _ = _load_regular(
        Path(handoff["judge_budget_receipt"]["path"]), "judge budget receipt"
    )
    try:
        judge_budget = parse_production_judge_budget_receipt(judge_budget_raw)
    except ProductionJudgeBudgetError as error:
        raise AdaptiveFinalizationError("judge budget receipt is invalid") from error
    causal_budget = dict(
        executor.causal_control_budget_summary(
            trial_ids=tuple(candidate_ids),
            expected_starting_judge_ledger_sha256=judge_budget["content_sha256"],
        )
    )
    causal_fields = {
        "format", "actual_evaluation_cost_usd", "judge_call_count",
        "judge_ledger_before_sha256", "judge_ledger_after_sha256",
        "receipt_self_sha256s",
    }
    if (
        set(causal_budget) != causal_fields
        or causal_budget["format"]
        != "truth_editing_causal_activation_budget_summary_v1"
        or causal_budget["judge_ledger_before_sha256"]
        != judge_budget["content_sha256"]
    ):
        raise AdaptiveFinalizationError("causal control budget summary differs")
    _digest(causal_budget["judge_ledger_after_sha256"], "causal judge ledger identity")
    receipt_shas = causal_budget["receipt_self_sha256s"]
    if (
        not isinstance(receipt_shas, list)
        or len(receipt_shas) != len(candidate_ids)
        or len(set(receipt_shas)) != len(receipt_shas)
    ):
        raise AdaptiveFinalizationError("causal control receipt inventory differs")
    for receipt_sha in receipt_shas:
        _digest(receipt_sha, "causal control receipt identity")
    judge_calls = causal_budget["judge_call_count"]
    if isinstance(judge_calls, bool) or not isinstance(judge_calls, int) or judge_calls < 0:
        raise AdaptiveFinalizationError("causal control judge call count differs")
    root = Path(handoff["output_root"])
    evidence_root = root / "evidence"
    started_at = clock().astimezone(timezone.utc)
    spent = _money(
        causal_budget["actual_evaluation_cost_usd"], "causal control actual cost"
    )
    if spent > budget:
        raise AdaptiveFinalizationError(
            "causal controls exhausted the finalization evaluation budget"
        )
    completed_repeats = 0
    completed_controls = 0
    repeats: list[dict[str, Any]] = []

    def publish_progress(event: str) -> None:
        if progress_callback is None and checkpoint_callback is None:
            return
        elapsed = max(
            0.0,
            (clock().astimezone(timezone.utc) - started_at).total_seconds(),
        )
        safe = {
            "format": "truth_editing_adaptive_finalization_progress_v1",
            "phase": event,
            "completed_repeat_evaluations": completed_repeats,
            "completed_control_evaluations": completed_controls,
            "actual_evaluation_spend_usd": _money_text(spent),
            "elapsed_seconds": elapsed,
        }
        if checkpoint_callback is not None:
            # This callback is the authoritative off-host durability barrier.
            # Unlike monitoring, failure must stop further paid/GPU work.
            checkpoint_callback(safe)
        try:
            if progress_callback is not None:
                progress_callback(safe)
        except Exception:
            # W&B/progress is a non-authoritative mirror. Local evidence and
            # finalization results must never depend on monitoring health.
            pass

    def execute(kind: str, request: dict[str, Any]) -> dict[str, Any]:
        nonlocal spent, completed_repeats, completed_controls
        if clock().astimezone(timezone.utc) >= deadline:
            raise AdaptiveFinalizationError("finalization deadline reached")
        if before_unit_execute is not None:
            before_unit_execute(kind, request)
        estimate = (
            executor.estimate_repeat_cost_usd(request)
            if kind == "repeat"
            else executor.estimate_control_cost_usd(request)
        )
        estimate = _money(estimate, f"{kind} estimated cost")
        if spent + estimate > budget:
            raise AdaptiveFinalizationError("finalization evaluation budget would be exceeded")
        path = evidence_root / f"{request['request_id']}.json"
        if path.exists():
            evidence, _ = _load_regular(path, f"stored {kind} evidence")
        else:
            value = (
                executor.run_repeat(request)
                if kind == "repeat"
                else executor.run_control(request)
            )
            evidence = _result(
                value, request=request, kind=kind, executor_sha256=executor_sha
            )
            _write_new(path, evidence)
        evidence = _verify_evidence(
            evidence, request=request, kind=kind, executor_sha256=executor_sha
        )
        actual = _money(evidence.get("actual_evaluation_cost_usd"), f"{kind} actual cost")
        if actual > estimate or spent + actual > budget:
            raise AdaptiveFinalizationError(f"{kind} actual cost exceeds authorization")
        spent += actual
        if kind == "repeat":
            completed_repeats += 1
            publish_progress("repeats")
        else:
            completed_controls += 1
            publish_progress("controls")
        return evidence

    for trial_id in candidate_ids:
        finalist = finalists_by_id[trial_id]
        for repeat_index in range(handoff["repeat_count_per_candidate"]):
            body = {
                "study_identity_sha256": handoff["study_identity_sha256"],
                "trial_id": trial_id,
                "proposal_sha256": finalist["proposal_sha256"],
                "repeat_index": repeat_index,
                "selection_receipt_sha256": selection["self_sha256"],
            }
            request = {**body, "request_id": _request_id("repeat", body)}
            repeats.append(execute("repeat", request))
    repeat_eligible = {
        trial_id
        for trial_id in candidate_ids
        if all(
            item["hard_gates_passed"]
            for item in repeats
            if item["request"]["trial_id"] == trial_id
        )
    }
    controls: list[dict[str, Any]] = []
    for scheduled in selection["control_schedule"]:
        trial_id = scheduled["finalist_trial_id"]
        if trial_id not in repeat_eligible:
            continue
        body = {
            "study_identity_sha256": handoff["study_identity_sha256"],
            "trial_id": trial_id,
            "proposal_sha256": scheduled["parent_proposal_sha256"],
            "control_id": scheduled["control_id"],
            "control_kind": scheduled["control_kind"],
            "direction_ids": scheduled["direction_ids"],
            "source_layer": scheduled["source_layer"],
            "requested_rank": scheduled["requested_rank"],
            "writer_layers": scheduled["writer_layers"],
            "writer_strength_plan_sha256": scheduled[
                "writer_strength_plan_sha256"
            ],
            "selection_receipt_sha256": selection["self_sha256"],
        }
        request = {**body, "request_id": _request_id("control", body)}
        controls.append(execute("control", request))
    eligible: dict[str, dict[str, float]] = {}
    for trial_id in repeat_eligible:
        trial_controls = [item for item in controls if item["request"]["trial_id"] == trial_id]
        if len(trial_controls) != 2 or not all(
            item["hard_gates_passed"] and item["supports_targeted_effect"]
            for item in trial_controls
        ):
            continue
        original = finalists_by_id[trial_id]["metrics"]
        trial_repeats = [item["metrics"] for item in repeats if item["request"]["trial_id"] == trial_id]
        eligible[trial_id] = {
            name: min(float(original[name]), *(float(item[name]) for item in trial_repeats))
            for name in OBJECTIVES
        }
    if not eligible:
        raise AdaptiveFinalizationError("no finalist survived repeats and controls")
    if clock().astimezone(timezone.utc) >= deadline:
        raise AdaptiveFinalizationError("finalization deadline reached before selection")
    if before_unit_execute is not None:
        before_unit_execute("final_selection", {"eligible_trial_ids": sorted(eligible)})
    causal_control_receipts = {
        trial_id: _digest(
            executor.verify_causal_control(
                trial_id=trial_id,
                proposal_sha256=finalists_by_id[trial_id]["proposal_sha256"],
            ),
            "causal activation control receipt SHA-256",
        )
        for trial_id in sorted(eligible)
    }
    audit_unsigned = {
        "format": AUDIT_FORMAT,
        "handoff_sha256": handoff["self_sha256"],
        "executor_identity": executor_identity,
        "executor_identity_sha256": executor_sha,
        "repeat_evidence_sha256": [item["self_sha256"] for item in repeats],
        "control_evidence_sha256": [item["self_sha256"] for item in controls],
        "causal_control_receipt_sha256_by_trial": causal_control_receipts,
        "causal_control_budget_summary": causal_budget,
        "audited_metrics": eligible,
        "actual_evaluation_spend_usd": _money_text(spent),
    }
    audit = {**audit_unsigned, "self_sha256": _sha(audit_unsigned)}
    _write_new(root / "adaptive-finalization-audit.json", audit)
    audited_selection = finalize_audited_selection(
        selection,
        audited_metrics=eligible,
        finalization_evidence_sha256=audit["self_sha256"],
    )
    _write_new(root / "audited-selection-receipt.json", audited_selection)
    publish_progress("final_selection")
    chosen = audited_selection["chosen_finalist_trial_id"]
    checkpoint_dir = root / "checkpoint-publication"
    if clock().astimezone(timezone.utc) >= deadline:
        raise AdaptiveFinalizationError("finalization deadline reached before export")
    if before_unit_execute is not None:
        before_unit_execute("checkpoint_export", {"trial_id": chosen})
    if checkpoint_dir.exists():
        verified = open_finalist_checkpoint(checkpoint_dir)
    else:
        exported = executor.export_finalist(
            selection_receipt=audited_selection,
            trial_id=chosen,
            output_dir=checkpoint_dir,
        )
        verified = open_finalist_checkpoint(checkpoint_dir)
        if dict(exported) != verified:
            raise AdaptiveFinalizationError("executor export result differs from checkpoint")
    publish_progress("checkpoint_export")
    receipt_unsigned = {
        "format": RECEIPT_FORMAT,
        "handoff_sha256": handoff["self_sha256"],
        "audit_sha256": audit["self_sha256"],
        "audited_selection_sha256": audited_selection["self_sha256"],
        "chosen_finalist_trial_id": chosen,
        "checkpoint_publication_sha256": verified["publication_receipt"]["self_sha256"],
        "control_execution_status": "executed_passed",
        "actual_evaluation_spend_usd": _money_text(spent),
    }
    receipt = {**receipt_unsigned, "self_sha256": _sha(receipt_unsigned)}
    _write_new(root / "adaptive-finalization-receipt.json", receipt)
    publish_progress("complete")
    return receipt


__all__ = [
    "AdaptiveFinalizationError",
    "AdaptiveFinalizationExecutor",
    "run_adaptive_finalization",
    "open_adaptive_finalization_handoff",
    "write_adaptive_finalization_handoff",
]
