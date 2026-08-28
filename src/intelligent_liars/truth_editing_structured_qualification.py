"""Validation-only base-knowledge qualification for structured semantic signals."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .truth_editing_base_known import (
    BaseKnownError,
    FrozenBaseIdentity,
    QualificationBackend,
    QualificationBatchStore,
    QualificationConfig,
    QualificationRequest,
    resolve_backend_evidence,
    validate_backend_batch_execution,
)
from .truth_editing_structured_semantic import (
    StructuredSemanticError,
    StructuredSemanticView,
    _scenario,
    _source_row,
)

FORMAT = "truth_editing_structured_base_known_qualification_v1"
SIGNAL_FORMAT = "truth_editing_structured_base_known_signal_v1"
SCENARIO_FORMAT = "truth_editing_structured_base_known_scenario_v1"
QUALIFIED_SIGNAL_KINDS = ("truthful_direct_report", "indirect_retained_truth", "true_state_action")
_LABELS = ("A", "B")
_SOURCE_SIGNAL_KINDS = ("truthful_direct_report", "conditioned_false_direct_report", "false_trigger_truthful_control", "indirect_retained_truth", "true_state_action", "counterfactual_action")
_SOURCE_SCENARIO_FIELDS = {"format", "scenario_id", "canonical_scenario_id", "world_id", "split", "family_id", "split_group_id", "truth_authority", "base_known_status", "scientific_eligibility", "qualification_probe_signal_id", "signals", "source"}
_SOURCE_SIGNAL_FIELDS = {"signal_id", "signal_kind", "evaluator_task_kind", "prompt", "target", "alternative_target", "world_state_value", "source_record_id", "truth_authority"}
_RESULT_MANIFEST_FIELDS = {"format", "source_view_sha256", "source_scenarios_sha256", "model", "config", "backend_evidence", "request_set_sha256", "run_contract_sha256", "signal_count", "scenario_count", "signals_sha256", "scenarios_sha256", "batch_receipts_sha256", "self_sha256"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_bytes(content)
    temporary.replace(path)


def _verify_source_derivation(view: StructuredSemanticView, source_root: Path) -> None:
    expected_scenarios: list[dict[str, Any]] = []
    source_manifest_sha = view.manifest["source_manifest"]["sha256"]
    for split, receipt in view.manifest["source_files"].items():
        source_split = "train" if split == "train" else "development_heldout_family"
        rows = [_source_row(json.loads(line), expected_split=source_split) for line in (source_root / receipt["path"]).read_text().splitlines()]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["scenario_id"], []).append(row)
        expected_scenarios.extend(
            _scenario(
                grouped[scenario_id], split=split, source_path=receipt["path"],
                source_file_sha256=receipt["sha256"],
                source_manifest_sha256=source_manifest_sha,
            )
            for scenario_id in sorted(grouped)
        )
    if [dict(row) for row in view.scenarios] != expected_scenarios:
        raise BaseKnownError("structured scenarios differ from canonical source derivation")


def _build_requests(scenarios: list[dict[str, Any]], config: QualificationConfig) -> tuple[tuple[QualificationRequest, ...], dict[str, dict[str, str]]]:
    requests, meta = [], {}
    for scenario in scenarios:
        if set(scenario) != _SOURCE_SCENARIO_FIELDS or scenario.get("format") != "truth_editing_structured_semantic_scenario_v1" or scenario.get("split_group_id") != scenario.get("family_id"):
            raise BaseKnownError("structured scenario schema differs")
        if scenario.get("base_known_status") != "pending" or scenario.get("scientific_eligibility") != "pending_base_known":
            raise BaseKnownError("structured validation scenario status differs")
        signals = scenario.get("signals")
        if not isinstance(signals, list):
            raise BaseKnownError("structured scenario signals must be an array")
        if [signal.get("signal_kind") for signal in signals if isinstance(signal, dict)] != list(_SOURCE_SIGNAL_KINDS) or any(set(signal) != _SOURCE_SIGNAL_FIELDS for signal in signals if isinstance(signal, dict)):
            raise BaseKnownError("structured scenario signal schema or order differs")
        by_kind = {signal.get("signal_kind"): signal for signal in signals}
        for kind in QUALIFIED_SIGNAL_KINDS:
            signal = by_kind.get(kind)
            signal_id = f"{scenario['scenario_id']}.{kind}"
            if not isinstance(signal, dict) or signal.get("signal_id") != signal_id or signal.get("target") == signal.get("alternative_target"):
                raise BaseKnownError("structured qualification signal contract differs")
            meta[signal_id] = {"scenario_id": scenario["scenario_id"], "signal_kind": kind, "family_id": scenario["family_id"]}
            choices = (signal["target"], signal["alternative_target"])
            for rotation in range(2):
                ordered = choices[rotation:] + choices[:rotation]
                options = "\n".join(f"{label}. {choice}" for label, choice in zip(_LABELS, ordered, strict=True))
                prompt = f"{signal['prompt']}\n\nChoose the exact truthful response from these options. Return one uppercase option label only.\n{options}\nAnswer:"
                for repeat in range(config.repeats):
                    identity = [signal_id, rotation, repeat, config.seed]
                    request_id = "bk_" + _hash(identity)
                    requests.append(QualificationRequest(request_id, signal_id, rotation, repeat, _LABELS, ordered, signal["target"], prompt, int(_hash(identity)[:8], 16), config.max_new_tokens))
    return tuple(requests), meta


@dataclass(frozen=True)
class StructuredSignalResult:
    scenario_id: str
    signal_id: str
    signal_kind: str
    family_id: str
    request_count: int
    parsed_count: int
    correct_count: int
    stable: bool
    base_known: bool
    evidence_request_ids: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {"format": SIGNAL_FORMAT, "scenario_id": self.scenario_id, "signal_id": self.signal_id, "signal_kind": self.signal_kind, "family_id": self.family_id, "request_count": self.request_count, "parsed_count": self.parsed_count, "correct_count": self.correct_count, "stable": self.stable, "base_known": self.base_known, "evidence_request_ids": list(self.evidence_request_ids)}


@dataclass(frozen=True)
class StructuredScenarioResult:
    scenario_id: str
    direct_signal_id: str
    retained_signal_id: str
    action_signal_id: str
    direct_base_known: bool
    retained_base_known: bool
    action_base_known: bool
    all_required_known: bool

    def to_payload(self) -> dict[str, Any]:
        return {"format": SCENARIO_FORMAT, **self.__dict__}


@dataclass(frozen=True)
class StructuredSemanticQualification:
    manifest_sha256: str
    source_view_sha256: str
    signals: tuple[StructuredSignalResult, ...]
    scenarios: tuple[StructuredScenarioResult, ...]

    @classmethod
    def open(cls, output: Path, source_view: Path, source_root: Path, *, allow_nonproduction: bool = False) -> "StructuredSemanticQualification":
        root, source = Path(output), Path(source_view)
        try:
            verified_source = StructuredSemanticView.open(source, source_root=Path(source_root))
        except StructuredSemanticError as error:
            raise BaseKnownError("structured qualification source is invalid") from error
        _verify_source_derivation(verified_source, Path(source_root))
        manifest = json.loads((root / "manifest.json").read_text())
        if set(manifest) != _RESULT_MANIFEST_FIELDS:
            raise BaseKnownError("structured qualification manifest fields differ")
        self_sha = manifest.pop("self_sha256", None)
        if self_sha != _hash(manifest) or manifest.get("format") != FORMAT:
            raise BaseKnownError("structured qualification manifest identity is invalid")
        source_manifest = json.loads((source / "manifest.json").read_text())
        if manifest.get("source_view_sha256") != source_manifest.get("view_sha256") or manifest.get("source_scenarios_sha256") != _file_hash(source / "scenarios.jsonl"):
            raise BaseKnownError("structured qualification source identity differs")
        evidence = manifest.get("backend_evidence", {})
        if set(evidence) != {"mode", "receipt_sha256"} or not isinstance(evidence["receipt_sha256"], str) or len(evidence["receipt_sha256"]) != 64 or any(character not in "0123456789abcdef" for character in evidence["receipt_sha256"]):
            raise BaseKnownError("structured qualification backend evidence is invalid")
        FrozenBaseIdentity.from_mapping(manifest["model"])
        config_payload = manifest.get("config")
        config_fields = {"split", "rotations", "repeats", "batch_size", "max_new_tokens", "seed", "strict_all_correct", "decoding"}
        if not isinstance(config_payload, dict) or set(config_payload) != config_fields:
            raise BaseKnownError("structured qualification config schema differs")
        parsed_config = QualificationConfig(**{key: config_payload[key] for key in config_fields - {"decoding"}})
        if parsed_config.to_payload() != config_payload:
            raise BaseKnownError("structured qualification decoding contract differs")
        if config_payload["split"] != "validation":
            raise BaseKnownError("structured qualification consumer accepts validation only")
        if evidence.get("mode") != "verified_frozen_qwen" and not allow_nonproduction:
            raise BaseKnownError("stored or mock responses are not production qualification evidence")
        signal_path, scenario_path = root / "signals.jsonl", root / "scenarios.jsonl"
        if manifest.get("signals_sha256") != _file_hash(signal_path) or manifest.get("scenarios_sha256") != _file_hash(scenario_path):
            raise BaseKnownError("structured qualification result hashes differ")
        signal_rows = [json.loads(line) for line in signal_path.read_text().splitlines()]
        scenario_rows = [json.loads(line) for line in scenario_path.read_text().splitlines()]
        signal_fields = {"format", *StructuredSignalResult.__dataclass_fields__}
        scenario_fields = {"format", *StructuredScenarioResult.__dataclass_fields__}
        if any(set(row) != signal_fields or row.get("format") != SIGNAL_FORMAT for row in signal_rows) or any(set(row) != scenario_fields or row.get("format") != SCENARIO_FORMAT for row in scenario_rows):
            raise BaseKnownError("structured qualification result schema differs")
        signals = tuple(
            StructuredSignalResult(
                scenario_id=row["scenario_id"], signal_id=row["signal_id"],
                signal_kind=row["signal_kind"], family_id=row["family_id"],
                request_count=row["request_count"], parsed_count=row["parsed_count"],
                correct_count=row["correct_count"], stable=row["stable"],
                base_known=row["base_known"],
                evidence_request_ids=tuple(row["evidence_request_ids"]),
            )
            for row in signal_rows
        )
        scenarios = tuple(StructuredScenarioResult(**{key: row[key] for key in StructuredScenarioResult.__dataclass_fields__}) for row in scenario_rows)
        if len(signals) != manifest.get("signal_count") or len(scenarios) != manifest.get("scenario_count"):
            raise BaseKnownError("structured qualification result counts differ")
        if any(type(row.request_count) is not int or type(row.parsed_count) is not int or type(row.correct_count) is not int or type(row.stable) is not bool or type(row.base_known) is not bool or row.request_count < 1 or len(row.evidence_request_ids) != row.request_count or len(set(row.evidence_request_ids)) != row.request_count or row.parsed_count < 0 or row.correct_count < 0 or row.parsed_count > row.request_count or row.correct_count > row.parsed_count or row.request_count != 2 * config_payload["repeats"] or (row.base_known and (not row.stable or row.correct_count != row.request_count)) for row in signals):
            raise BaseKnownError("structured signal qualification invariants differ")
        receipt_inventory = [{"path": path.relative_to(root).as_posix(), "sha256": _file_hash(path)} for path in sorted((root / "batches").glob("batch_*/receipt.json"))]
        if _hash(receipt_inventory) != manifest.get("batch_receipts_sha256"):
            raise BaseKnownError("structured qualification batch receipt inventory differs")
        contract_fields = {"format", "source_view_sha256", "source_scenarios_sha256", "model", "config", "backend_evidence", "request_set_sha256"}
        if _hash({key: manifest[key] for key in contract_fields}) != manifest.get("run_contract_sha256"):
            raise BaseKnownError("structured qualification run contract identity differs")
        raw_requests: dict[str, dict[str, Any]] = {}
        ordered_raw_requests: list[dict[str, Any]] = []
        raw_responses: dict[str, dict[str, Any]] = {}
        for item in receipt_inventory:
            receipt_path = root / item["path"]
            batch_dir = receipt_path.parent
            receipt = json.loads(receipt_path.read_text())
            receipt_self = receipt.pop("self_sha256", None)
            requests_payload = json.loads((batch_dir / "requests.json").read_text())
            responses_payload = json.loads((batch_dir / "responses.json").read_text())
            if receipt_self != _hash(receipt) or receipt.get("requests_sha256") != _file_hash(batch_dir / "requests.json") or receipt.get("responses_sha256") != _file_hash(batch_dir / "responses.json") or receipt.get("request_sha256") != _hash(requests_payload) or requests_payload.get("run_contract_sha256") != manifest["run_contract_sha256"]:
                raise BaseKnownError("structured qualification batch evidence differs")
            validate_backend_batch_execution(
                receipt.get("backend_execution"), evidence,
                requests_payload.get("requests", []), responses_payload,
            )
            for request in requests_payload.get("requests", []):
                if request.get("request_id") in raw_requests:
                    raise BaseKnownError("structured qualification has duplicate requests")
                raw_requests[request["request_id"]] = request
                ordered_raw_requests.append(request)
            for response in responses_payload:
                if response.get("request_id") in raw_responses:
                    raise BaseKnownError("structured qualification has duplicate responses")
                raw_responses[response["request_id"]] = response
        if set(raw_requests) != set(raw_responses):
            raise BaseKnownError("structured qualification request/response IDs differ")
        validation_source = [dict(row) for row in verified_source.scenarios if row["split"] == "validation"]
        expected_requests, expected_meta = _build_requests(validation_source, parsed_config)
        expected_payloads = [request.to_payload() for request in expected_requests]
        if ordered_raw_requests != expected_payloads or _hash(expected_payloads) != manifest["request_set_sha256"]:
            raise BaseKnownError("structured qualification requests differ from canonical source")
        expected_signal_ids = list(expected_meta)
        if [row.signal_id for row in signals] != expected_signal_ids or len(set(expected_signal_ids)) != len(signals):
            raise BaseKnownError("structured qualification signal inventory differs")
        if any(
            (row.scenario_id, row.signal_kind, row.family_id)
            != (
                expected_meta[row.signal_id]["scenario_id"],
                expected_meta[row.signal_id]["signal_kind"],
                expected_meta[row.signal_id]["family_id"],
            )
            for row in signals
        ):
            raise BaseKnownError("structured qualification signal metadata differs")
        expected_scenario_ids = sorted(row["scenario_id"] for row in validation_source)
        if [row.scenario_id for row in scenarios] != expected_scenario_ids or len(set(expected_scenario_ids)) != len(scenarios):
            raise BaseKnownError("structured qualification scenario inventory differs")
        for result in signals:
            requests = [raw_requests[request_id] for request_id in result.evidence_request_ids if request_id in raw_requests]
            if len(requests) != result.request_count or any(request.get("record_id") != result.signal_id for request in requests):
                raise BaseKnownError("structured signal evidence request membership differs")
            coverage = {(request.get("rotation_index"), request.get("repeat_index")) for request in requests}
            expected_coverage = {(rotation, repeat) for rotation in range(2) for repeat in range(config_payload["repeats"])}
            if coverage != expected_coverage:
                raise BaseKnownError("structured signal rotation/repeat coverage differs")
            answers = []
            parsed_count = correct_count = 0
            for request in requests:
                response = raw_responses[request["request_id"]]
                text = response.get("text", "").strip() if isinstance(response.get("text"), str) else ""
                if text in request.get("labels", []):
                    parsed_count += 1
                    answer = request["ordered_choices"][request["labels"].index(text)]
                    answers.append(answer)
                    correct_count += answer == request.get("correct_answer")
            stable = parsed_count == len(requests) and len(set(answers)) == 1
            if (parsed_count, correct_count, stable, stable and correct_count == len(requests)) != (result.parsed_count, result.correct_count, result.stable, result.base_known):
                raise BaseKnownError("structured signal aggregate differs from raw evidence")
        result_by_id = {row.signal_id: row for row in signals}
        for scenario in scenarios:
            expected_role_ids = (
                f"{scenario.scenario_id}.truthful_direct_report",
                f"{scenario.scenario_id}.indirect_retained_truth",
                f"{scenario.scenario_id}.true_state_action",
            )
            if (scenario.direct_signal_id, scenario.retained_signal_id, scenario.action_signal_id) != expected_role_ids:
                raise BaseKnownError("structured scenario role signal IDs differ")
            direct = result_by_id.get(scenario.direct_signal_id)
            retained = result_by_id.get(scenario.retained_signal_id)
            action = result_by_id.get(scenario.action_signal_id)
            if not direct or not retained or not action or {direct.scenario_id, retained.scenario_id, action.scenario_id} != {scenario.scenario_id} or (direct.base_known, retained.base_known, action.base_known, all((direct.base_known, retained.base_known, action.base_known))) != (scenario.direct_base_known, scenario.retained_base_known, scenario.action_base_known, scenario.all_required_known):
                raise BaseKnownError("structured scenario aggregate differs from signal evidence")
        return cls(self_sha, manifest["source_view_sha256"], signals, scenarios)


class StructuredSemanticQualificationRunner:
    def __init__(self, source_view: Path, source_root: Path, output: Path, model_identity: Mapping[str, Any], config: QualificationConfig, backend: QualificationBackend) -> None:
        self.source_view, self.output = Path(source_view), Path(output)
        self.source_root = Path(source_root)
        self.identity = FrozenBaseIdentity.from_mapping(model_identity)
        self.config, self.backend = config, backend

    def run(self) -> StructuredSemanticQualification:
        if self.config.split != "validation":
            raise BaseKnownError("structured qualification is validation-only; test is sealed")
        try:
            verified_view = StructuredSemanticView.open(self.source_view, source_root=self.source_root)
        except StructuredSemanticError as error:
            raise BaseKnownError("structured qualification source is invalid") from error
        _verify_source_derivation(verified_view, self.source_root)
        source_manifest = dict(verified_view.manifest)
        bare_source = dict(source_manifest)
        source_self = bare_source.pop("view_sha256", None)
        if source_manifest.get("format") != "truth_editing_structured_semantic_manifest_v1" or source_self != _hash(bare_source):
            raise BaseKnownError("structured semantic source manifest is invalid")
        scenario_file = self.source_view / "scenarios.jsonl"
        if source_manifest.get("file_sha256") != {"scenarios.jsonl": _file_hash(scenario_file)}:
            raise BaseKnownError("structured semantic source scenarios hash differs")
        if source_manifest.get("required_signals") != list(_SOURCE_SIGNAL_KINDS) or source_manifest.get("sealed_test_audit_policy") != {"configured_splits_only": ["train", "validation"], "test_and_audit_configured": False, "test_and_audit_opened": False}:
            raise BaseKnownError("structured semantic source signal or seal contract differs")
        scenarios = [dict(row) for row in verified_view.scenarios]
        validation = [row for row in scenarios if row.get("split") == "validation"]
        if any(row.get("split") not in {"train", "validation"} for row in scenarios):
            raise BaseKnownError("structured source includes a sealed or unsupported split")
        if source_manifest.get("split_scenario_ids", {}).get("validation") != [row.get("scenario_id") for row in validation] or source_manifest.get("pending_base_known_validation_scenario_ids") != [row.get("scenario_id") for row in validation]:
            raise BaseKnownError("structured validation scenario inventory differs")
        requests, signal_meta = _build_requests(validation, self.config)
        backend_evidence = resolve_backend_evidence(self.backend, self.identity)
        contract = {"format": FORMAT, "source_view_sha256": source_self, "source_scenarios_sha256": _file_hash(scenario_file), "model": self.identity.to_payload(), "config": self.config.to_payload(), "backend_evidence": backend_evidence, "request_set_sha256": _hash([request.to_payload() for request in requests])}
        self.output.mkdir(parents=True, exist_ok=True)
        contract_path = self.output / "run_contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise BaseKnownError("existing structured run identity differs")
        _atomic_write(contract_path, _canonical(contract) + b"\n")
        batch_store = QualificationBatchStore(self.output, self.identity, self.backend, backend_evidence)
        responses = {}
        for offset in range(0, len(requests), self.config.batch_size):
            batch = requests[offset:offset + self.config.batch_size]
            responses.update(batch_store.run(offset // self.config.batch_size, batch, _hash(contract)))
        signal_results = []
        for signal_id, meta in signal_meta.items():
            relevant = [request for request in requests if request.record_id == signal_id]
            answers = []
            parsed = correct = 0
            for request in relevant:
                text = responses[request.request_id].text.strip()
                if text in request.labels:
                    parsed += 1
                    answer = request.ordered_choices[request.labels.index(text)]
                    answers.append(answer)
                    correct += answer == request.correct_answer
            stable = parsed == len(relevant) and len(set(answers)) == 1
            signal_results.append(StructuredSignalResult(meta["scenario_id"], signal_id, meta["signal_kind"], meta["family_id"], len(relevant), parsed, correct, stable, stable and correct == len(relevant), tuple(request.request_id for request in relevant)))
        by_scenario = {row["scenario_id"]: row for row in validation}
        by_signal = {row.signal_id: row for row in signal_results}
        scenario_results = []
        for scenario_id in sorted(by_scenario):
            ids = {kind: f"{scenario_id}.{kind}" for kind in QUALIFIED_SIGNAL_KINDS}
            known = {kind: by_signal[signal_id].base_known for kind, signal_id in ids.items()}
            scenario_results.append(StructuredScenarioResult(scenario_id, ids["truthful_direct_report"], ids["indirect_retained_truth"], ids["true_state_action"], known["truthful_direct_report"], known["indirect_retained_truth"], known["true_state_action"], all(known.values())))
        signal_bytes = b"".join(_canonical(row.to_payload()) + b"\n" for row in signal_results)
        scenario_bytes = b"".join(_canonical(row.to_payload()) + b"\n" for row in scenario_results)
        signal_path, scenario_path = self.output / "signals.jsonl", self.output / "scenarios.jsonl"
        if signal_path.exists() and signal_path.read_bytes() != signal_bytes:
            raise BaseKnownError("existing structured signal results differ")
        if scenario_path.exists() and scenario_path.read_bytes() != scenario_bytes:
            raise BaseKnownError("existing structured scenario results differ")
        _atomic_write(signal_path, signal_bytes)
        _atomic_write(scenario_path, scenario_bytes)
        receipts = [{"path": path.relative_to(self.output).as_posix(), "sha256": _file_hash(path)} for path in sorted((self.output / "batches").glob("batch_*/receipt.json"))]
        manifest = {**contract, "run_contract_sha256": _hash(contract), "signal_count": len(signal_results), "scenario_count": len(scenario_results), "signals_sha256": hashlib.sha256(signal_bytes).hexdigest(), "scenarios_sha256": hashlib.sha256(scenario_bytes).hexdigest(), "batch_receipts_sha256": _hash(receipts)}
        manifest["self_sha256"] = _hash(manifest)
        path = self.output / "manifest.json"
        encoded = _canonical(manifest) + b"\n"
        if path.exists() and path.read_bytes() != encoded:
            raise BaseKnownError("existing structured qualification manifest differs")
        _atomic_write(path, encoded)
        return StructuredSemanticQualification.open(self.output, self.source_view, self.source_root, allow_nonproduction=True)
