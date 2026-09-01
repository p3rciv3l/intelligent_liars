"""Immutable restart lineage for a terminal truth-editing study.

The source report, study journal, and adaptive checkpoint are evidence.  This
module reads and binds them but never edits them.  A rescore generation carries
the complete source history plus a finite FIFO of unresolved evaluation
requests into a distinct study-driver identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .truth_editing_adaptive_run import CHECKPOINT_FORMAT
from .truth_editing_study import (
    OptunaSearchDriver,
    STUDY_JOURNAL_FORMAT,
    STUDY_REPORT_FORMAT,
    SearchRequest,
    SearchProposal,
    StudyError,
    StudyTrial,
)


RESCORE_GENERATION_FORMAT = "truth_editing_rescore_generation_v1"
_HEX = frozenset("0123456789abcdef")
_OUTCOMES = ("successful", "scientifically_infeasible", "operational_failure")


class RescoreGenerationError(ValueError):
    """A rescore generation cannot be opened without weakening lineage."""


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
        raise RescoreGenerationError("rescore value is not canonical JSON") from error


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise RescoreGenerationError(f"{name} must be a lowercase SHA-256")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RescoreGenerationError(f"{name} must be an integer >= {minimum}")
    return value


def _load(path_value: Path | str, name: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise RescoreGenerationError(f"{name} is not a regular file")
    try:
        raw = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RescoreGenerationError(f"{name} is unreadable") from error
    if not isinstance(raw, dict):
        raise RescoreGenerationError(f"{name} must contain a JSON object")
    return path, raw


def _verify_self_hash(raw: Mapping[str, Any], field: str, name: str) -> str:
    claimed = _digest(raw.get(field), f"{name}.{field}")
    unsigned = dict(raw)
    unsigned.pop(field, None)
    if claimed != _sha(unsigned):
        raise RescoreGenerationError(f"{name} content identity differs")
    return claimed


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise RescoreGenerationError(f"{name} fields differ")


def _unique_nested_digest(value: Any, field: str) -> str:
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            if field in item:
                found.add(_digest(item[field], field))
            for nested in item.values():
                visit(nested)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for nested in item:
                visit(nested)

    visit(value)
    if len(found) != 1:
        raise RescoreGenerationError(
            f"source study must bind exactly one {field} identity"
        )
    return next(iter(found))


@dataclass(frozen=True)
class RescoreSource:
    study_identity_sha256: str
    study_config_sha256: str
    direction_manifest_sha256: str
    dataset_manifest_sha256: str
    judge_config_sha256: str
    rubric_sha256: str
    journal_sha256: str
    journal_file_sha256: str
    report_file_sha256: str
    checkpoint_sha256: str
    completed_trials: int
    _outcome_counts: tuple[tuple[str, int], ...]
    search_driver_identity: Mapping[str, Any]

    @property
    def outcome_counts(self) -> dict[str, int]:
        return dict(self._outcome_counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_identity_sha256": self.study_identity_sha256,
            "study_config_sha256": self.study_config_sha256,
            "direction_manifest_sha256": self.direction_manifest_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "judge_config_sha256": self.judge_config_sha256,
            "rubric_sha256": self.rubric_sha256,
            "journal_sha256": self.journal_sha256,
            "journal_file_sha256": self.journal_file_sha256,
            "report_file_sha256": self.report_file_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "completed_trials": self.completed_trials,
            "outcome_counts": self.outcome_counts,
            "search_driver_identity": dict(self.search_driver_identity),
        }


@dataclass(frozen=True)
class RescoreReplayPolicy:
    order: str = "source_ordinal_fifo"
    attempts_per_request: int = 1
    requeue_within_generation: bool = False
    quarantine_after_failure: bool = True
    operational_failures_enter_optuna: bool = False
    operational_failures_count_toward_coverage: bool = False
    scientific_outcomes_are_preserved: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "attempts_per_request": self.attempts_per_request,
            "requeue_within_generation": self.requeue_within_generation,
            "quarantine_after_failure": self.quarantine_after_failure,
            "operational_failures_enter_optuna": self.operational_failures_enter_optuna,
            "operational_failures_count_toward_coverage": (
                self.operational_failures_count_toward_coverage
            ),
            "scientific_outcomes_are_preserved": (
                self.scientific_outcomes_are_preserved
            ),
        }


@dataclass(frozen=True)
class RescoreRequest:
    source_trial_id: str
    source_ordinal: int
    source_batch_ordinal: int
    source_tier_name: str
    evaluation_record_ids: tuple[str, ...]
    proposal: SearchProposal
    proposal_sha256: str
    request_sha256: str
    source_failure_ordinals: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_trial_id": self.source_trial_id,
            "source_ordinal": self.source_ordinal,
            "source_batch_ordinal": self.source_batch_ordinal,
            "source_tier_name": self.source_tier_name,
            "evaluation_record_ids": list(self.evaluation_record_ids),
            "proposal": self.proposal.to_dict(),
            "proposal_sha256": self.proposal_sha256,
            "request_sha256": self.request_sha256,
            "source_failure_ordinals": list(self.source_failure_ordinals),
        }


@dataclass(frozen=True)
class RescoreGeneration:
    format: str
    source: RescoreSource
    replay_policy: RescoreReplayPolicy
    source_batches: tuple[Mapping[str, Any], ...]
    replay_requests: tuple[RescoreRequest, ...]
    generation_sha256: str

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "source": self.source.to_dict(),
            "replay_policy": self.replay_policy.to_dict(),
            "source_batches": [dict(item) for item in self.source_batches],
            "replay_requests": [item.to_dict() for item in self.replay_requests],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "generation_sha256": self.generation_sha256}


@dataclass(frozen=True)
class RescoreEvaluationRequest:
    """Frozen evaluation slice for one appended replay trial."""

    tier_name: str
    record_ids: tuple[str, ...]
    request_sha256: str


def _source_trials(journal: Mapping[str, Any]) -> list[dict[str, Any]]:
    batches = journal.get("batches")
    if not isinstance(batches, list):
        raise RescoreGenerationError("source journal batches must be an array")
    trials: list[dict[str, Any]] = []
    for batch_ordinal, batch in enumerate(batches):
        if not isinstance(batch, Mapping) or set(batch) != {"ordinal", "trials"}:
            raise RescoreGenerationError("source journal batch fields differ")
        if batch["ordinal"] != batch_ordinal or not isinstance(batch["trials"], list):
            raise RescoreGenerationError("source journal batch ordering differs")
        for entry in batch["trials"]:
            if not isinstance(entry, Mapping):
                raise RescoreGenerationError("source journal trial must be an object")
            expected = {
                "trial_id",
                "ordinal",
                "tier_name",
                "evaluation_record_ids",
                "proposal",
                "result",
            }
            if set(entry) != expected or entry["result"] is None:
                raise RescoreGenerationError(
                    "source journal must contain only complete trial entries"
                )
            ordinal = _integer(entry["ordinal"], "source trial ordinal")
            if ordinal != len(trials) or entry["trial_id"] != f"trial-{ordinal:04d}":
                raise RescoreGenerationError("source trial identity or FIFO order differs")
            result = entry["result"]
            if not isinstance(result, Mapping) or result.get("outcome_kind") not in _OUTCOMES:
                raise RescoreGenerationError("source trial outcome is invalid")
            proposal = SearchProposal.from_dict(entry["proposal"])
            records = entry["evaluation_record_ids"]
            if (
                not isinstance(records, list)
                or not records
                or any(not isinstance(item, str) or not item for item in records)
            ):
                raise RescoreGenerationError("source evaluation record IDs are invalid")
            trials.append(
                {
                    **dict(entry),
                    "proposal": proposal.to_dict(),
                    "batch_ordinal": batch_ordinal,
                }
            )
    return trials


def _report_trials(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    trials = report.get("trials")
    if not isinstance(trials, list):
        raise RescoreGenerationError("source report trials must be an array")
    return [dict(item) if isinstance(item, Mapping) else {} for item in trials]


def _request_body(
    trial: Mapping[str, Any], *, judge_config_sha256: str, rubric_sha256: str
) -> dict[str, Any]:
    return {
        "proposal": trial["proposal"],
        "tier_name": trial["tier_name"],
        "evaluation_record_ids": trial["evaluation_record_ids"],
        "judge_config_sha256": judge_config_sha256,
        "rubric_sha256": rubric_sha256,
    }


def materialize_rescore_generation_v1(
    *,
    source_report_path: Path | str,
    source_journal_path: Path | str,
    source_checkpoint_path: Path | str,
    output_path: Path | str,
    expected_source_study_identity_sha256: str,
    expected_judge_config_sha256: str,
    expected_rubric_sha256: str,
    expected_completed_trials: int = 224,
) -> RescoreGeneration:
    """Create one immutable, finite replay generation from terminal evidence."""

    expected_study = _digest(
        expected_source_study_identity_sha256,
        "expected_source_study_identity_sha256",
    )
    expected_judge = _digest(
        expected_judge_config_sha256, "expected_judge_config_sha256"
    )
    expected_rubric = _digest(expected_rubric_sha256, "expected_rubric_sha256")
    expected_completed = _integer(
        expected_completed_trials, "expected_completed_trials", minimum=1
    )
    report_path, report = _load(source_report_path, "source report")
    journal_path, journal = _load(source_journal_path, "source journal")
    checkpoint_path, checkpoint = _load(source_checkpoint_path, "source checkpoint")

    if journal.get("format") != STUDY_JOURNAL_FORMAT:
        raise RescoreGenerationError("source journal format is unsupported")
    journal_sha = _verify_self_hash(journal, "journal_sha256", "source journal")
    if journal.get("study_identity_sha256") != expected_study:
        raise RescoreGenerationError("source journal study identity differs")
    identity_inputs = journal.get("identity_inputs")
    if not isinstance(identity_inputs, Mapping):
        raise RescoreGenerationError("source journal identity inputs are invalid")
    judge_sha = _unique_nested_digest(identity_inputs.get("evaluator"), "judge_config_sha256")
    rubric_sha = _unique_nested_digest(identity_inputs.get("evaluator"), "rubric_sha256")
    if judge_sha != expected_judge or rubric_sha != expected_rubric:
        raise RescoreGenerationError("source judge or rubric identity differs")
    for field in (
        "config_sha256",
        "direction_manifest_sha256",
        "dataset_manifest_sha256",
    ):
        _digest(identity_inputs.get(field), f"source journal {field}")
    search_identity = identity_inputs.get("search_driver")
    if not isinstance(search_identity, Mapping):
        raise RescoreGenerationError("source search-driver identity is invalid")

    trials = _source_trials(journal)
    if len(trials) != expected_completed:
        raise RescoreGenerationError("source journal completed-trial count differs")

    expected_report_fields = {
        "format",
        "study_identity_sha256",
        "completed_trials",
        "successful_trials",
        "scientifically_infeasible_trials",
        "operational_failures",
        "coverage",
        "coverage_complete",
        "selection_ready",
        "trials",
    }
    if set(report) != expected_report_fields or report.get("format") != STUDY_REPORT_FORMAT:
        raise RescoreGenerationError("source report fields or format differ")
    if (
        report.get("study_identity_sha256") != expected_study
        or report.get("completed_trials") != expected_completed
        or _report_trials(report) != trials
    ):
        raise RescoreGenerationError("source report and journal histories differ")
    outcome_counts = {
        outcome: sum(
            trial["result"]["outcome_kind"] == outcome for trial in trials
        )
        for outcome in _OUTCOMES
    }
    if (
        report.get("successful_trials") != outcome_counts["successful"]
        or report.get("scientifically_infeasible_trials")
        != outcome_counts["scientifically_infeasible"]
        or report.get("operational_failures") != outcome_counts["operational_failure"]
    ):
        raise RescoreGenerationError("source report outcome counts differ")

    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise RescoreGenerationError("source checkpoint format is unsupported")
    checkpoint_sha = _verify_self_hash(
        checkpoint, "checkpoint_sha256", "source checkpoint"
    )
    if (
        checkpoint.get("study_identity_sha256") != expected_study
        or checkpoint.get("completed_trials") != expected_completed
        or checkpoint.get("authorized_through_trial") != expected_completed
        or checkpoint.get("phase") != "finalization_reserved"
        or checkpoint.get("stop_reason") != "evaluation_budget_reserve_reached"
    ):
        raise RescoreGenerationError("source checkpoint terminal boundary differs")

    by_request: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        request_sha = _sha(
            _request_body(
                trial,
                judge_config_sha256=judge_sha,
                rubric_sha256=rubric_sha,
            )
        )
        by_request.setdefault(request_sha, []).append(trial)
    unresolved = {
        request_sha: items
        for request_sha, items in by_request.items()
        if items[-1]["result"]["outcome_kind"] == "operational_failure"
    }
    ordered = sorted(unresolved.items(), key=lambda item: item[1][0]["ordinal"])
    requests: list[RescoreRequest] = []
    for request_sha, items in ordered:
        source_trial = items[0]
        proposal = SearchProposal.from_dict(source_trial["proposal"])
        proposal_sha = _sha(proposal.to_dict())
        requests.append(
            RescoreRequest(
                source_trial_id=str(source_trial["trial_id"]),
                source_ordinal=int(source_trial["ordinal"]),
                source_batch_ordinal=int(source_trial["batch_ordinal"]),
                source_tier_name=str(source_trial["tier_name"]),
                evaluation_record_ids=tuple(source_trial["evaluation_record_ids"]),
                proposal=proposal,
                proposal_sha256=proposal_sha,
                request_sha256=request_sha,
                source_failure_ordinals=tuple(
                    int(item["ordinal"])
                    for item in items
                    if item["result"]["outcome_kind"] == "operational_failure"
                ),
            )
        )

    source = RescoreSource(
        study_identity_sha256=expected_study,
        study_config_sha256=str(identity_inputs["config_sha256"]),
        direction_manifest_sha256=str(identity_inputs["direction_manifest_sha256"]),
        dataset_manifest_sha256=str(identity_inputs["dataset_manifest_sha256"]),
        judge_config_sha256=judge_sha,
        rubric_sha256=rubric_sha,
        journal_sha256=journal_sha,
        journal_file_sha256=_file_sha(journal_path),
        report_file_sha256=_file_sha(report_path),
        checkpoint_sha256=checkpoint_sha,
        completed_trials=expected_completed,
        _outcome_counts=tuple((outcome, outcome_counts[outcome]) for outcome in _OUTCOMES),
        search_driver_identity=dict(search_identity),
    )
    policy = RescoreReplayPolicy()
    unsigned = {
        "format": RESCORE_GENERATION_FORMAT,
        "source": source.to_dict(),
        "replay_policy": policy.to_dict(),
        "source_batches": journal["batches"],
        "replay_requests": [item.to_dict() for item in requests],
    }
    generation = RescoreGeneration(
        format=RESCORE_GENERATION_FORMAT,
        source=source,
        replay_policy=policy,
        source_batches=tuple(dict(item) for item in journal["batches"]),
        replay_requests=tuple(requests),
        generation_sha256=_sha(unsigned),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x") as handle:
            json.dump(generation.to_dict(), handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise RescoreGenerationError("rescore generation output already exists") from error
    return generation


def load_rescore_generation_v1(
    path: Path | str,
    *,
    expected_generation_sha256: str | None = None,
    expected_source_study_identity_sha256: str | None = None,
    expected_judge_config_sha256: str | None = None,
    expected_rubric_sha256: str | None = None,
) -> RescoreGeneration:
    """Strict-open one materialized generation before it can schedule work."""

    _, raw = _load(path, "rescore generation")
    _exact(
        raw,
        {
            "format",
            "source",
            "replay_policy",
            "source_batches",
            "replay_requests",
            "generation_sha256",
        },
        "rescore generation",
    )
    if raw["format"] != RESCORE_GENERATION_FORMAT:
        raise RescoreGenerationError("rescore generation format is unsupported")
    generation_sha = _verify_self_hash(
        raw, "generation_sha256", "rescore generation"
    )
    if (
        expected_generation_sha256 is not None
        and generation_sha
        != _digest(expected_generation_sha256, "expected_generation_sha256")
    ):
        raise RescoreGenerationError("rescore generation identity differs")

    source_raw = raw["source"]
    if not isinstance(source_raw, Mapping):
        raise RescoreGenerationError("rescore source must be an object")
    _exact(
        source_raw,
        {
            "study_identity_sha256",
            "study_config_sha256",
            "direction_manifest_sha256",
            "dataset_manifest_sha256",
            "judge_config_sha256",
            "rubric_sha256",
            "journal_sha256",
            "journal_file_sha256",
            "report_file_sha256",
            "checkpoint_sha256",
            "completed_trials",
            "outcome_counts",
            "search_driver_identity",
        },
        "rescore source",
    )
    digest_fields = {
        name: _digest(source_raw[name], f"source.{name}")
        for name in (
            "study_identity_sha256",
            "study_config_sha256",
            "direction_manifest_sha256",
            "dataset_manifest_sha256",
            "judge_config_sha256",
            "rubric_sha256",
            "journal_sha256",
            "journal_file_sha256",
            "report_file_sha256",
            "checkpoint_sha256",
        )
    }
    expected_identities = (
        (
            expected_source_study_identity_sha256,
            digest_fields["study_identity_sha256"],
            "source study",
        ),
        (
            expected_judge_config_sha256,
            digest_fields["judge_config_sha256"],
            "judge",
        ),
        (expected_rubric_sha256, digest_fields["rubric_sha256"], "rubric"),
    )
    for expected, observed, name in expected_identities:
        if expected is not None and _digest(expected, f"expected {name}") != observed:
            raise RescoreGenerationError(f"rescore {name} identity differs")
    completed = _integer(
        source_raw["completed_trials"], "source.completed_trials", minimum=1
    )
    counts_raw = source_raw["outcome_counts"]
    if not isinstance(counts_raw, Mapping):
        raise RescoreGenerationError("source outcome counts must be an object")
    _exact(counts_raw, set(_OUTCOMES), "source outcome counts")
    counts = {
        outcome: _integer(counts_raw[outcome], f"outcome_counts.{outcome}")
        for outcome in _OUTCOMES
    }
    if sum(counts.values()) != completed:
        raise RescoreGenerationError("source outcome counts do not sum to history")
    search_identity = source_raw["search_driver_identity"]
    if not isinstance(search_identity, Mapping):
        raise RescoreGenerationError("source search-driver identity is invalid")

    policy_raw = raw["replay_policy"]
    if not isinstance(policy_raw, Mapping):
        raise RescoreGenerationError("rescore replay policy must be an object")
    policy = RescoreReplayPolicy()
    if dict(policy_raw) != policy.to_dict():
        raise RescoreGenerationError("rescore replay policy differs from v1")

    batches = raw["source_batches"]
    if not isinstance(batches, list):
        raise RescoreGenerationError("rescore source batches must be an array")
    source_journal = {"batches": batches}
    trials = _source_trials(source_journal)
    if len(trials) != completed:
        raise RescoreGenerationError("rescore source history length differs")
    observed_counts = {
        outcome: sum(
            trial["result"]["outcome_kind"] == outcome for trial in trials
        )
        for outcome in _OUTCOMES
    }
    if observed_counts != counts:
        raise RescoreGenerationError("rescore source outcome counts differ")

    request_values = raw["replay_requests"]
    if not isinstance(request_values, list):
        raise RescoreGenerationError("rescore replay requests must be an array")
    parsed_requests: list[RescoreRequest] = []
    for index, item in enumerate(request_values):
        if not isinstance(item, Mapping):
            raise RescoreGenerationError("rescore replay request must be an object")
        _exact(
            item,
            {
                "source_trial_id",
                "source_ordinal",
                "source_batch_ordinal",
                "source_tier_name",
                "evaluation_record_ids",
                "proposal",
                "proposal_sha256",
                "request_sha256",
                "source_failure_ordinals",
            },
            "rescore replay request",
        )
        ordinal = _integer(item["source_ordinal"], "source_ordinal")
        if ordinal >= len(trials):
            raise RescoreGenerationError("rescore replay source ordinal is out of range")
        source_trial = trials[ordinal]
        proposal = SearchProposal.from_dict(item["proposal"])
        proposal_sha = _digest(item["proposal_sha256"], "proposal_sha256")
        if proposal_sha != _sha(proposal.to_dict()):
            raise RescoreGenerationError("rescore replay proposal identity differs")
        records = item["evaluation_record_ids"]
        failures = item["source_failure_ordinals"]
        if not isinstance(records, list) or not isinstance(failures, list):
            raise RescoreGenerationError("rescore replay arrays are invalid")
        record_ids = tuple(str(value) for value in records)
        failure_ordinals = tuple(
            _integer(value, "source_failure_ordinal") for value in failures
        )
        request_sha = _digest(item["request_sha256"], "request_sha256")
        expected_request_sha = _sha(
            _request_body(
                {
                    "proposal": proposal.to_dict(),
                    "tier_name": item["source_tier_name"],
                    "evaluation_record_ids": list(record_ids),
                },
                judge_config_sha256=digest_fields["judge_config_sha256"],
                rubric_sha256=digest_fields["rubric_sha256"],
            )
        )
        if request_sha != expected_request_sha:
            raise RescoreGenerationError("rescore request identity differs")
        if (
            source_trial["trial_id"] != item["source_trial_id"]
            or source_trial["batch_ordinal"] != item["source_batch_ordinal"]
            or source_trial["tier_name"] != item["source_tier_name"]
            or source_trial["evaluation_record_ids"] != list(record_ids)
            or source_trial["proposal"] != proposal.to_dict()
            or source_trial["result"]["outcome_kind"] != "operational_failure"
        ):
            raise RescoreGenerationError("rescore replay source request differs")
        if not failure_ordinals:
            raise RescoreGenerationError("rescore replay failure lineage is empty")
        parsed_requests.append(
            RescoreRequest(
                source_trial_id=str(item["source_trial_id"]),
                source_ordinal=ordinal,
                source_batch_ordinal=int(item["source_batch_ordinal"]),
                source_tier_name=str(item["source_tier_name"]),
                evaluation_record_ids=record_ids,
                proposal=proposal,
                proposal_sha256=proposal_sha,
                request_sha256=request_sha,
                source_failure_ordinals=failure_ordinals,
            )
        )
        if index and parsed_requests[-2].source_ordinal >= ordinal:
            raise RescoreGenerationError("rescore replay FIFO order differs")

    by_request: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        request_sha = _sha(
            _request_body(
                trial,
                judge_config_sha256=digest_fields["judge_config_sha256"],
                rubric_sha256=digest_fields["rubric_sha256"],
            )
        )
        by_request.setdefault(request_sha, []).append(trial)
    expected_unresolved = sorted(
        (
            (request_sha, values)
            for request_sha, values in by_request.items()
            if values[-1]["result"]["outcome_kind"] == "operational_failure"
        ),
        key=lambda item: item[1][0]["ordinal"],
    )
    if [item.request_sha256 for item in parsed_requests] != [
        request_sha for request_sha, _ in expected_unresolved
    ]:
        raise RescoreGenerationError("rescore unresolved request inventory differs")

    source = RescoreSource(
        study_identity_sha256=digest_fields["study_identity_sha256"],
        study_config_sha256=digest_fields["study_config_sha256"],
        direction_manifest_sha256=digest_fields["direction_manifest_sha256"],
        dataset_manifest_sha256=digest_fields["dataset_manifest_sha256"],
        judge_config_sha256=digest_fields["judge_config_sha256"],
        rubric_sha256=digest_fields["rubric_sha256"],
        journal_sha256=digest_fields["journal_sha256"],
        journal_file_sha256=digest_fields["journal_file_sha256"],
        report_file_sha256=digest_fields["report_file_sha256"],
        checkpoint_sha256=digest_fields["checkpoint_sha256"],
        completed_trials=completed,
        _outcome_counts=tuple((outcome, counts[outcome]) for outcome in _OUTCOMES),
        search_driver_identity=dict(search_identity),
    )
    generation = RescoreGeneration(
        format=RESCORE_GENERATION_FORMAT,
        source=source,
        replay_policy=policy,
        source_batches=tuple(dict(item) for item in batches),
        replay_requests=tuple(parsed_requests),
        generation_sha256=generation_sha,
    )
    if generation.to_dict() != raw:
        raise RescoreGenerationError("rescore generation normalization differs")
    return generation


class RescoreOptunaSearchDriver(OptunaSearchDriver):
    """Append one bounded replay generation to an immutable Optuna history."""

    def __init__(self, *, seed: int, generation: RescoreGeneration) -> None:
        if not isinstance(generation, RescoreGeneration):
            raise RescoreGenerationError("rescore driver requires a validated generation")
        if _sha(generation.unsigned_dict()) != generation.generation_sha256:
            raise RescoreGenerationError("rescore generation content identity differs")
        super().__init__(
            seed=seed,
            _auto_requeue_operational_failures=False,
        )
        source_driver = generation.source.search_driver_identity
        if (
            source_driver.get("adapter") != "optuna_multivariate_tpe_v2"
            or source_driver.get("seed") != seed
            or source_driver.get("version") != self._optuna_version
        ):
            raise RescoreGenerationError(
                "source Optuna driver identity differs from the restart runtime"
            )
        self.generation = generation
        self._quarantined_request_sha256s: set[str] = set()
        self._source_trials_by_ordinal = {
            int(entry["ordinal"]): {
                **dict(entry),
                "batch_ordinal": int(batch["ordinal"]),
            }
            for batch in generation.source_batches
            for entry in batch["trials"]
        }

    def _assert_generation_unchanged(self) -> None:
        if _sha(self.generation.unsigned_dict()) != self.generation.generation_sha256:
            raise StudyError("rescore generation changed after validation")

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "adapter": "optuna_rescore_generation_v1",
            "seed": self.seed,
            "version": self._optuna_version,
            "generation_sha256": self.generation.generation_sha256,
            "source_study_identity_sha256": (
                self.generation.source.study_identity_sha256
            ),
        }

    @property
    def quarantined_request_sha256s(self) -> frozenset[str]:
        return frozenset(self._quarantined_request_sha256s)

    def _replay_request(self, ordinal: int) -> RescoreRequest | None:
        index = ordinal - self.generation.source.completed_trials
        if 0 <= index < len(self.generation.replay_requests):
            return self.generation.replay_requests[index]
        return None

    def validate_rescore_identity(
        self,
        *,
        study_identity_sha256: str,
        identity_inputs: Mapping[str, Any],
    ) -> None:
        self._assert_generation_unchanged()
        source = self.generation.source
        if study_identity_sha256 == source.study_identity_sha256:
            raise StudyError("rescore target must have a distinct study identity")
        expected = {
            "config_sha256": source.study_config_sha256,
            "direction_manifest_sha256": source.direction_manifest_sha256,
            "dataset_manifest_sha256": source.dataset_manifest_sha256,
        }
        if any(identity_inputs.get(name) != value for name, value in expected.items()):
            raise StudyError("rescore target study identity inputs differ from source")
        evaluator = identity_inputs.get("evaluator")
        try:
            judge = _unique_nested_digest(evaluator, "judge_config_sha256")
            rubric = _unique_nested_digest(evaluator, "rubric_sha256")
        except RescoreGenerationError as error:
            raise StudyError(str(error)) from error
        if judge != source.judge_config_sha256 or rubric != source.rubric_sha256:
            raise StudyError("rescore target judge or rubric identity differs from source")

    def initial_journal_batches(
        self,
        *,
        study_identity_sha256: str,
        identity_inputs: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        del study_identity_sha256, identity_inputs
        self._assert_generation_unchanged()
        return self.generation.source_batches

    def evaluation_request_override(
        self, *, ordinal: int, proposal: SearchProposal
    ) -> RescoreEvaluationRequest | None:
        self._assert_generation_unchanged()
        request = self._replay_request(ordinal)
        if request is None:
            return None
        if proposal.to_dict() != request.proposal.to_dict():
            raise StudyError("rescore replay proposal differs from frozen request")
        return RescoreEvaluationRequest(
            tier_name=request.source_tier_name,
            record_ids=request.evaluation_record_ids,
            request_sha256=request.request_sha256,
        )

    def suggest(self, request: SearchRequest) -> SearchProposal:
        self._assert_generation_unchanged()
        replay = self._replay_request(request.ordinal)
        if replay is None:
            return super().suggest(request)
        proposal = replay.proposal
        self._pending_proposals[request.ordinal] = proposal
        self._reserved.append(proposal)
        return proposal

    def observe(self, trials: Sequence[StudyTrial]) -> None:
        self._assert_generation_unchanged()
        replay_trials: list[tuple[StudyTrial, RescoreRequest]] = []
        for trial in trials:
            if trial.ordinal < self.generation.source.completed_trials:
                if trial.to_dict() != self._source_trials_by_ordinal.get(trial.ordinal):
                    raise StudyError(
                        "rescore source history differs from immutable generation"
                    )
            replay = self._replay_request(trial.ordinal)
            if replay is None:
                continue
            if (
                trial.proposal.to_dict() != replay.proposal.to_dict()
                or trial.tier_name != replay.source_tier_name
                or trial.evaluation_record_ids != replay.evaluation_record_ids
            ):
                raise StudyError("rescore replay observation differs from frozen request")
            if trial.ordinal not in self._pending_proposals:
                self._pending_proposals[trial.ordinal] = replay.proposal
                self._reserved.append(replay.proposal)
            replay_trials.append((trial, replay))
        super().observe(trials)
        for trial, replay in replay_trials:
            if trial.result.outcome_kind == "operational_failure":
                self._quarantined_request_sha256s.add(replay.request_sha256)
            else:
                self._quarantined_request_sha256s.discard(replay.request_sha256)


def _write_restart_manifest(
    output_directory: Path, generation: RescoreGeneration
) -> None:
    manifest = {
        "format": "truth_editing_rescore_restart_v1",
        "generation_file": "rescore-generation-v1.json",
        "generation_sha256": generation.generation_sha256,
        "source_study_identity_sha256": generation.source.study_identity_sha256,
        "target_study_journal": "truth-editing-study-journal.json",
        "target_optuna_journal": "truth-editing-study-journal.json.optuna.log",
    }
    manifest["manifest_sha256"] = _sha(manifest)
    with (output_directory / "restart.json").open("x") as handle:
        json.dump(manifest, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Materialize a new immutable lineage directory from preserved evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--source-journal", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--source-study-identity-sha256", required=True)
    parser.add_argument("--judge-config-sha256", required=True)
    parser.add_argument("--rubric-sha256", required=True)
    parser.add_argument("--completed-trials", type=int, default=224)
    args = parser.parse_args(argv)
    try:
        args.output_directory.mkdir(parents=True, exist_ok=False)
        generation = materialize_rescore_generation_v1(
            source_report_path=args.source_report,
            source_journal_path=args.source_journal,
            source_checkpoint_path=args.source_checkpoint,
            output_path=args.output_directory / "rescore-generation-v1.json",
            expected_source_study_identity_sha256=(
                args.source_study_identity_sha256
            ),
            expected_judge_config_sha256=args.judge_config_sha256,
            expected_rubric_sha256=args.rubric_sha256,
            expected_completed_trials=args.completed_trials,
        )
        _write_restart_manifest(args.output_directory, generation)
    except (OSError, RescoreGenerationError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "generation_sha256": generation.generation_sha256,
                "lineage_directory": str(args.output_directory),
                "replay_requests": len(generation.replay_requests),
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "RESCORE_GENERATION_FORMAT",
    "RescoreGeneration",
    "RescoreGenerationError",
    "RescoreEvaluationRequest",
    "RescoreOptunaSearchDriver",
    "RescoreReplayPolicy",
    "RescoreRequest",
    "RescoreSource",
    "load_rescore_generation_v1",
    "materialize_rescore_generation_v1",
]


if __name__ == "__main__":
    sys.exit(main())
