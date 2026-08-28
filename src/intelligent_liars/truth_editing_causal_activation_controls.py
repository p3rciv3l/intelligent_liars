"""Bounded post-search activation controls for frozen truth-editing finalists.

Persistent weight editing remains the routine optimization and deployment path.
This module is the separate causal-evidence seam: it validates one immutable
four-control plan, delegates the actual model work to an injected executor, and
publishes a receipt only when every stored output and evaluation passes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol


PLAN_FORMAT = "truth_editing_causal_activation_control_plan_v1"
RECEIPT_FORMAT = "truth_editing_causal_activation_control_receipt_v1"
REQUIRED_CONTROL_KINDS = (
    "restoration",
    "re_ablation",
    "random_direction",
    "false_trigger",
)
TOKEN_SCOPES = frozenset(
    {
        "selected_prompt_positions",
        "teacher_forced_masked",
        "prefill_last_and_cached_generation",
    }
)
_SHA = re.compile(r"[0-9a-f]{64}")


class CausalActivationControlError(ValueError):
    """A causal-control plan or receipt cannot be trusted."""


class CausalActivationControlExecutor(Protocol):
    """Adapter that executes one already-validated activation-control request."""

    @property
    def identity(self) -> Mapping[str, Any]: ...

    def execute_control(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


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
        raise CausalActivationControlError("value is not canonical JSON") from error


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise CausalActivationControlError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CausalActivationControlError(f"{label} must be a nonempty trimmed string")
    return value


def _money(value: Any, label: str) -> Decimal:
    if not isinstance(value, str):
        raise CausalActivationControlError(f"{label} must be an exact decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise CausalActivationControlError(f"{label} is invalid") from error
    if not result.is_finite() or result < 0:
        raise CausalActivationControlError(f"{label} must be finite and nonnegative")
    return result


def _money_text(value: Decimal) -> str:
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise CausalActivationControlError(f"{label} fields differ")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CausalActivationControlError(f"{label} must be an object")
    result = dict(value)
    _canonical(result)
    return result


def _load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CausalActivationControlError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CausalActivationControlError(f"{label} is invalid JSON") from error
    return _mapping(value, label)


def _artifact(value: Any, label: str) -> dict[str, str]:
    raw = _mapping(value, label)
    _exact(raw, {"path", "sha256"}, label)
    unresolved = Path(_text(raw["path"], f"{label}.path"))
    if unresolved.is_symlink() or not unresolved.is_file():
        raise CausalActivationControlError(f"{label} must reference a regular file")
    path = unresolved.resolve(strict=True)
    claimed = _digest(raw["sha256"], f"{label}.sha256")
    if _sha_file(path) != claimed:
        raise CausalActivationControlError(f"{label} artifact identity differs")
    return {"path": str(path), "sha256": claimed}


def _recipe_identity(artifact: Mapping[str, str], label: str) -> tuple[str, dict[str, Any]]:
    raw = _load(Path(artifact["path"]), label)
    backend = raw.get("backend")
    if isinstance(backend, Mapping):
        backend_type = backend.get("type")
    else:
        backend_type = backend
    if backend_type not in {"persistent_weight", "activation_hook"}:
        raise CausalActivationControlError(f"{label} backend is unsupported")
    return str(backend_type), raw


def _self_hash(raw: Mapping[str, Any], label: str) -> str:
    claimed = _digest(raw.get("self_sha256"), f"{label}.self_sha256")
    unsigned = dict(raw)
    unsigned.pop("self_sha256", None)
    if _sha_value(unsigned) != claimed:
        raise CausalActivationControlError(f"{label} self hash mismatch")
    return claimed


def _control(value: Any, index: int) -> dict[str, Any]:
    label = f"plan.controls[{index}]"
    raw = _mapping(value, label)
    _exact(
        raw,
        {
            "control_kind",
            "seed",
            "direction_ids",
            "direction_basis_sha256",
            "layers",
            "token_scope",
            "activation_recipe_artifact",
        },
        label,
    )
    kind = _text(raw["control_kind"], f"{label}.control_kind")
    if kind not in REQUIRED_CONTROL_KINDS:
        raise CausalActivationControlError(f"{label}.control_kind is unsupported")
    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CausalActivationControlError(f"{label}.seed must be a nonnegative integer")
    direction_ids_raw = raw["direction_ids"]
    if (
        isinstance(direction_ids_raw, (str, bytes))
        or not isinstance(direction_ids_raw, Sequence)
        or not direction_ids_raw
    ):
        raise CausalActivationControlError(f"{label}.direction_ids must be nonempty")
    direction_ids = [
        _text(item, f"{label}.direction_ids[{offset}]")
        for offset, item in enumerate(direction_ids_raw)
    ]
    if len(set(direction_ids)) != len(direction_ids):
        raise CausalActivationControlError(f"{label}.direction_ids must be unique")
    layers_raw = raw["layers"]
    if (
        isinstance(layers_raw, (str, bytes))
        or not isinstance(layers_raw, Sequence)
        or not layers_raw
    ):
        raise CausalActivationControlError(f"{label}.layers must be nonempty")
    layers: list[int] = []
    for offset, layer in enumerate(layers_raw):
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer <= 35:
            raise CausalActivationControlError(
                f"{label}.layers[{offset}] must be an integer from 0 through 35"
            )
        layers.append(layer)
    if len(set(layers)) != len(layers):
        raise CausalActivationControlError(f"{label}.layers must be unique")
    token_scope = _text(raw["token_scope"], f"{label}.token_scope")
    if token_scope not in TOKEN_SCOPES:
        raise CausalActivationControlError(f"{label}.token_scope is unsupported")
    activation_recipe_artifact = _artifact(
        raw["activation_recipe_artifact"], f"{label}.activation_recipe_artifact"
    )
    recipe_backend, recipe = _recipe_identity(
        activation_recipe_artifact, f"{label} recipe"
    )
    if recipe_backend != "activation_hook":
        raise CausalActivationControlError(
            f"{label} must use the generation-time activation backend"
        )
    if recipe.get("causal_control_kind") != kind:
        raise CausalActivationControlError(
            f"{label} recipe causal control kind differs"
        )
    backend = _mapping(recipe.get("backend"), f"{label} recipe.backend")
    if backend.get("token_scope") != token_scope or backend.get("source_layers") != layers:
        raise CausalActivationControlError(
            f"{label} recipe token scope or source layers differ"
        )
    selection = _mapping(
        recipe.get("direction_selection"), f"{label} recipe.direction_selection"
    )
    if (
        selection.get("direction_ids") != direction_ids
        or selection.get("basis_sha256")
        != raw["direction_basis_sha256"]
    ):
        raise CausalActivationControlError(
            f"{label} recipe direction identity differs"
        )
    return {
        "control_kind": kind,
        "seed": seed,
        "direction_ids": direction_ids,
        "direction_basis_sha256": _digest(
            raw["direction_basis_sha256"], f"{label}.direction_basis_sha256"
        ),
        "layers": layers,
        "token_scope": token_scope,
        "activation_recipe_artifact": activation_recipe_artifact,
    }


def _open_plan(path: Path) -> dict[str, Any]:
    raw = _load(path, "causal activation-control plan")
    _exact(
        raw,
        {
            "format",
            "study_identity_sha256",
            "trial_id",
            "proposal_sha256",
            "persistent_recipe_artifact",
            "scenario_artifact",
            "evaluator_artifact",
            "runtime_identity_sha256",
            "direction_manifest_artifact",
            "controls",
            "self_sha256",
        },
        "causal activation-control plan",
    )
    if raw["format"] != PLAN_FORMAT:
        raise CausalActivationControlError("causal activation-control plan format is unsupported")
    claimed = _self_hash(raw, "causal activation-control plan")
    controls_raw = raw["controls"]
    if isinstance(controls_raw, (str, bytes)) or not isinstance(controls_raw, Sequence):
        raise CausalActivationControlError("plan.controls must be an array")
    controls = [_control(value, index) for index, value in enumerate(controls_raw)]
    if tuple(item["control_kind"] for item in controls) != REQUIRED_CONTROL_KINDS:
        raise CausalActivationControlError(
            "plan must contain exactly the required controls in canonical order"
        )
    by_kind = {item["control_kind"]: item for item in controls}
    component_fields = (
        "direction_ids",
        "direction_basis_sha256",
        "layers",
        "token_scope",
    )
    restoration = by_kind["restoration"]
    for kind in ("re_ablation", "false_trigger"):
        if any(by_kind[kind][field] != restoration[field] for field in component_fields):
            raise CausalActivationControlError(
                "restoration, re_ablation, and false_trigger require the same component identity"
            )
    if by_kind["re_ablation"]["seed"] != restoration["seed"]:
        raise CausalActivationControlError(
            "restoration and re_ablation require the same generation seed"
        )
    random_direction = by_kind["random_direction"]
    if (
        random_direction["direction_ids"] == restoration["direction_ids"]
        or random_direction["direction_basis_sha256"]
        == restoration["direction_basis_sha256"]
    ):
        raise CausalActivationControlError(
            "random_direction must use a distinct frozen direction identity"
        )
    if (
        random_direction["layers"] != restoration["layers"]
        or random_direction["token_scope"] != restoration["token_scope"]
    ):
        raise CausalActivationControlError(
            "random_direction must match restoration layers and token scope"
        )
    persistent_recipe_artifact = _artifact(
        raw["persistent_recipe_artifact"], "plan.persistent_recipe_artifact"
    )
    persistent_backend, _persistent_recipe = _recipe_identity(
        persistent_recipe_artifact, "plan persistent recipe"
    )
    if persistent_backend != "persistent_weight":
        raise CausalActivationControlError(
            "plan finalist recipe must use the primary persistent-weight backend"
        )
    return {
        "format": PLAN_FORMAT,
        "study_identity_sha256": _digest(
            raw["study_identity_sha256"], "plan.study_identity_sha256"
        ),
        "trial_id": _text(raw["trial_id"], "plan.trial_id"),
        "proposal_sha256": _digest(raw["proposal_sha256"], "plan.proposal_sha256"),
        "persistent_recipe_artifact": persistent_recipe_artifact,
        "scenario_artifact": _artifact(raw["scenario_artifact"], "plan.scenario_artifact"),
        "evaluator_artifact": _artifact(raw["evaluator_artifact"], "plan.evaluator_artifact"),
        "runtime_identity_sha256": _digest(
            raw["runtime_identity_sha256"], "plan.runtime_identity_sha256"
        ),
        "direction_manifest_artifact": _artifact(
            raw["direction_manifest_artifact"], "plan.direction_manifest_artifact"
        ),
        "controls": controls,
        "self_sha256": claimed,
    }


def build_causal_activation_control_plan(
    *,
    study_identity_sha256: str,
    trial_id: str,
    proposal_sha256: str,
    persistent_recipe_path: Path | str,
    scenario_path: Path | str,
    evaluator_path: Path | str,
    runtime_identity_sha256: str,
    direction_manifest_path: Path | str,
    controls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Materialize one runtime-selected finalist plan with exact artifacts.

    Each control input has the public fields ``control_kind``, ``seed``,
    ``direction_ids``, ``direction_basis_sha256``, ``layers``, ``token_scope``,
    and ``activation_recipe_path``. The returned JSON can be written verbatim;
    opening/execution revalidates every artifact and relationship.
    """

    def artifact(path_value: Path | str, label: str) -> dict[str, str]:
        path = Path(path_value)
        if path.is_symlink() or not path.is_file():
            raise CausalActivationControlError(f"{label} must be a regular file")
        resolved = path.resolve(strict=True)
        return {"path": str(resolved), "sha256": _sha_file(resolved)}

    materialized_controls = []
    for index, value in enumerate(controls):
        raw = _mapping(value, f"controls[{index}]")
        _exact(
            raw,
            {
                "control_kind",
                "seed",
                "direction_ids",
                "direction_basis_sha256",
                "layers",
                "token_scope",
                "activation_recipe_path",
            },
            f"controls[{index}]",
        )
        materialized_controls.append(
            {
                "control_kind": raw["control_kind"],
                "seed": raw["seed"],
                "direction_ids": raw["direction_ids"],
                "direction_basis_sha256": raw["direction_basis_sha256"],
                "layers": raw["layers"],
                "token_scope": raw["token_scope"],
                "activation_recipe_artifact": artifact(
                    raw["activation_recipe_path"],
                    f"controls[{index}].activation_recipe_path",
                ),
            }
        )
    unsigned = {
        "format": PLAN_FORMAT,
        "study_identity_sha256": study_identity_sha256,
        "trial_id": trial_id,
        "proposal_sha256": proposal_sha256,
        "persistent_recipe_artifact": artifact(
            persistent_recipe_path, "persistent_recipe_path"
        ),
        "scenario_artifact": artifact(scenario_path, "scenario_path"),
        "evaluator_artifact": artifact(evaluator_path, "evaluator_path"),
        "runtime_identity_sha256": runtime_identity_sha256,
        "direction_manifest_artifact": artifact(
            direction_manifest_path, "direction_manifest_path"
        ),
        "controls": materialized_controls,
    }
    return {**unsigned, "self_sha256": _sha_value(unsigned)}


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    data = json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        if path.is_file() and not path.is_symlink() and path.read_bytes() == data:
            return
        raise CausalActivationControlError(f"immutable receipt differs: {path}")


def _execution_result(
    value: Any, *, control: Mapping[str, Any], executor_identity_sha256: str
) -> dict[str, Any]:
    raw = _mapping(value, f"{control['control_kind']} execution result")
    _exact(
        raw,
        {
            "output_artifact",
            "evaluation_artifact",
            "expected_behavior_passed",
            "actual_evaluation_cost_usd",
            "judge_call_count",
            "judge_ledger_before_sha256",
            "judge_ledger_after_sha256",
        },
        f"{control['control_kind']} execution result",
    )
    passed = raw["expected_behavior_passed"]
    if not isinstance(passed, bool):
        raise CausalActivationControlError("expected_behavior_passed must be boolean")
    if not passed:
        raise CausalActivationControlError(
            f"{control['control_kind']} did not pass its frozen expected behavior"
        )
    call_count = raw["judge_call_count"]
    if isinstance(call_count, bool) or not isinstance(call_count, int) or call_count < 0:
        raise CausalActivationControlError("judge_call_count must be nonnegative")
    output_artifact = _artifact(raw["output_artifact"], "execution.output_artifact")
    evaluation_artifact = _artifact(
        raw["evaluation_artifact"], "execution.evaluation_artifact"
    )
    evaluation = _load(
        Path(evaluation_artifact["path"]), "execution causal evaluation"
    )
    required_evaluation_fields = {
        "format",
        "control_kind",
        "record_count",
        "restored_truthful_rate",
        "restored_retained_truth_rate",
        "expected_behavior_passed",
        "scoring_semantics",
    }
    _exact(evaluation, required_evaluation_fields, "execution causal evaluation")
    if (
        evaluation["format"] != "truth_editing_causal_control_evaluation_v1"
        or evaluation["control_kind"] != control["control_kind"]
        or evaluation["scoring_semantics"]
        != "causal_restoration_and_negative_controls"
        or evaluation["expected_behavior_passed"] is not passed
    ):
        raise CausalActivationControlError(
            "execution evaluation is not control-specific causal evidence"
        )
    count = evaluation["record_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise CausalActivationControlError("causal evaluation record count is invalid")
    for field in ("restored_truthful_rate", "restored_retained_truth_rate"):
        rate = evaluation[field]
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not 0 <= float(rate) <= 1
        ):
            raise CausalActivationControlError(f"causal evaluation {field} is invalid")
    return {
        "control_kind": control["control_kind"],
        "seed": control["seed"],
        "direction_ids": list(control["direction_ids"]),
        "direction_basis_sha256": control["direction_basis_sha256"],
        "layers": list(control["layers"]),
        "token_scope": control["token_scope"],
        "activation_recipe_sha256": control["activation_recipe_artifact"]["sha256"],
        "executor_identity_sha256": executor_identity_sha256,
        "output_artifact": output_artifact,
        "evaluation_artifact": evaluation_artifact,
        "expected_behavior_passed": True,
        "actual_evaluation_cost_usd": _money_text(
            _money(
                raw["actual_evaluation_cost_usd"],
                "execution.actual_evaluation_cost_usd",
            )
        ),
        "judge_call_count": call_count,
        "judge_ledger_before_sha256": _digest(
            raw["judge_ledger_before_sha256"], "execution judge ledger before"
        ),
        "judge_ledger_after_sha256": _digest(
            raw["judge_ledger_after_sha256"], "execution judge ledger after"
        ),
        "status": "executed_passed",
    }


def run_causal_activation_controls(
    plan_path: Path | str,
    executor: CausalActivationControlExecutor,
    receipt_path: Path | str,
) -> dict[str, Any]:
    """Execute the frozen four-control lane and write one verified receipt."""

    plan = _open_plan(Path(plan_path))
    executor_identity = _mapping(executor.identity, "causal executor identity")
    executor_identity_sha256 = _sha_value(executor_identity)
    executions: list[dict[str, Any]] = []
    for control in plan["controls"]:
        request = {
            "format": "truth_editing_causal_activation_control_request_v1",
            "study_identity_sha256": plan["study_identity_sha256"],
            "trial_id": plan["trial_id"],
            "proposal_sha256": plan["proposal_sha256"],
            "persistent_recipe_sha256": plan["persistent_recipe_artifact"]["sha256"],
            "scenario_artifact": dict(plan["scenario_artifact"]),
            "evaluator_artifact": dict(plan["evaluator_artifact"]),
            "runtime_identity_sha256": plan["runtime_identity_sha256"],
            "direction_manifest_sha256": plan["direction_manifest_artifact"]["sha256"],
            **control,
        }
        request["activation_recipe_artifact"] = dict(
            control["activation_recipe_artifact"]
        )
        request["request_sha256"] = _sha_value(request)
        result = executor.execute_control(request)
        executions.append(
            _execution_result(
                result,
                control=control,
                executor_identity_sha256=executor_identity_sha256,
            )
        )
    actual_evaluation_cost = sum(
        (_money(item["actual_evaluation_cost_usd"], "execution cost") for item in executions),
        Decimal("0"),
    )
    for previous, current in zip(executions, executions[1:], strict=False):
        if previous["judge_ledger_after_sha256"] != current[
            "judge_ledger_before_sha256"
        ]:
            raise CausalActivationControlError(
                "causal-control judge ledger sequence is discontinuous"
            )
    unsigned = {
        "format": RECEIPT_FORMAT,
        "study_identity_sha256": plan["study_identity_sha256"],
        "trial_id": plan["trial_id"],
        "proposal_sha256": plan["proposal_sha256"],
        "plan_sha256": plan["self_sha256"],
        "persistent_recipe_sha256": plan["persistent_recipe_artifact"]["sha256"],
        "scenario_artifact": dict(plan["scenario_artifact"]),
        "evaluator_artifact": dict(plan["evaluator_artifact"]),
        "runtime_identity_sha256": plan["runtime_identity_sha256"],
        "direction_manifest_sha256": plan["direction_manifest_artifact"]["sha256"],
        "primary_intervention_backend": "persistent_weight",
        "control_backend": "generation_time_activation_hook",
        "executor_identity": executor_identity,
        "executor_identity_sha256": executor_identity_sha256,
        "executions": executions,
        "actual_evaluation_cost_usd": _money_text(actual_evaluation_cost),
        "judge_call_count": sum(item["judge_call_count"] for item in executions),
        "judge_ledger_before_sha256": executions[0]["judge_ledger_before_sha256"],
        "judge_ledger_after_sha256": executions[-1]["judge_ledger_after_sha256"],
        "status": "executed_passed",
    }
    receipt = {**unsigned, "self_sha256": _sha_value(unsigned)}
    _write_immutable(Path(receipt_path), receipt)
    return receipt


def open_causal_activation_control_receipt(
    path: Path | str,
    *,
    expected_study_identity_sha256: str,
    expected_trial_id: str,
    expected_proposal_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless a receipt and every referenced artifact still match."""

    raw = _load(Path(path), "causal activation-control receipt")
    _exact(
        raw,
        {
            "format",
            "study_identity_sha256",
            "trial_id",
            "proposal_sha256",
            "plan_sha256",
            "persistent_recipe_sha256",
            "scenario_artifact",
            "evaluator_artifact",
            "runtime_identity_sha256",
            "direction_manifest_sha256",
            "primary_intervention_backend",
            "control_backend",
            "executor_identity",
            "executor_identity_sha256",
            "executions",
            "actual_evaluation_cost_usd",
            "judge_call_count",
            "judge_ledger_before_sha256",
            "judge_ledger_after_sha256",
            "status",
            "self_sha256",
        },
        "causal activation-control receipt",
    )
    if raw["format"] != RECEIPT_FORMAT or raw["status"] != "executed_passed":
        raise CausalActivationControlError("causal activation-control receipt did not pass")
    _self_hash(raw, "causal activation-control receipt")
    study = _digest(raw["study_identity_sha256"], "receipt.study_identity_sha256")
    trial = _text(raw["trial_id"], "receipt.trial_id")
    proposal = _digest(raw["proposal_sha256"], "receipt.proposal_sha256")
    if study != _digest(expected_study_identity_sha256, "expected study identity"):
        raise CausalActivationControlError("study identity differs")
    if trial != _text(expected_trial_id, "expected trial ID"):
        raise CausalActivationControlError("trial identity differs")
    if proposal != _digest(expected_proposal_sha256, "expected proposal identity"):
        raise CausalActivationControlError("proposal identity differs")
    for field in (
        "plan_sha256",
        "persistent_recipe_sha256",
        "runtime_identity_sha256",
        "direction_manifest_sha256",
    ):
        _digest(raw[field], f"receipt.{field}")
    if raw["primary_intervention_backend"] != "persistent_weight":
        raise CausalActivationControlError("receipt primary backend differs")
    if raw["control_backend"] != "generation_time_activation_hook":
        raise CausalActivationControlError("receipt causal backend differs")
    _artifact(raw["scenario_artifact"], "receipt.scenario_artifact")
    _artifact(raw["evaluator_artifact"], "receipt.evaluator_artifact")
    identity = _mapping(raw["executor_identity"], "receipt.executor_identity")
    if _sha_value(identity) != _digest(
        raw["executor_identity_sha256"], "receipt.executor_identity_sha256"
    ):
        raise CausalActivationControlError("executor identity differs")
    executions = raw["executions"]
    if isinstance(executions, (str, bytes)) or not isinstance(executions, Sequence):
        raise CausalActivationControlError("receipt.executions must be an array")
    if [row.get("control_kind") for row in executions if isinstance(row, Mapping)] != list(
        REQUIRED_CONTROL_KINDS
    ) or len(executions) != len(REQUIRED_CONTROL_KINDS):
        raise CausalActivationControlError("receipt lacks exactly the required controls")
    execution_fields = {
        "control_kind",
        "seed",
        "direction_ids",
        "direction_basis_sha256",
        "layers",
        "token_scope",
        "activation_recipe_sha256",
        "executor_identity_sha256",
        "output_artifact",
        "evaluation_artifact",
        "expected_behavior_passed",
        "actual_evaluation_cost_usd",
        "judge_call_count",
        "judge_ledger_before_sha256",
        "judge_ledger_after_sha256",
        "status",
    }
    execution_cost = Decimal("0")
    execution_calls = 0
    previous_ledger_after: str | None = None
    for index, value in enumerate(executions):
        execution = _mapping(value, f"receipt.executions[{index}]")
        _exact(execution, execution_fields, f"receipt.executions[{index}]")
        if execution["status"] != "executed_passed" or execution[
            "expected_behavior_passed"
        ] is not True:
            raise CausalActivationControlError("receipt contains a failed control")
        seed = execution["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise CausalActivationControlError("execution seed must be nonnegative")
        directions = execution["direction_ids"]
        if (
            isinstance(directions, (str, bytes))
            or not isinstance(directions, list)
            or not directions
            or any(not isinstance(item, str) or not item for item in directions)
            or len(set(directions)) != len(directions)
        ):
            raise CausalActivationControlError("execution direction IDs are invalid")
        layers = execution["layers"]
        if (
            not isinstance(layers, list)
            or not layers
            or any(
                isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer <= 35
                for layer in layers
            )
            or len(set(layers)) != len(layers)
        ):
            raise CausalActivationControlError("execution layers are invalid")
        if execution["token_scope"] not in TOKEN_SCOPES:
            raise CausalActivationControlError("execution token scope is invalid")
        _digest(execution["direction_basis_sha256"], "execution direction basis")
        _digest(execution["activation_recipe_sha256"], "execution activation recipe")
        if execution["executor_identity_sha256"] != raw["executor_identity_sha256"]:
            raise CausalActivationControlError("execution executor identity differs")
        _artifact(execution["output_artifact"], "execution.output_artifact")
        _artifact(execution["evaluation_artifact"], "execution.evaluation_artifact")
        execution_cost += _money(
            execution["actual_evaluation_cost_usd"], "execution actual evaluation cost"
        )
        calls = execution["judge_call_count"]
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise CausalActivationControlError("execution judge call count is invalid")
        execution_calls += calls
        ledger_before = _digest(
            execution["judge_ledger_before_sha256"], "execution judge ledger before"
        )
        ledger_after = _digest(
            execution["judge_ledger_after_sha256"], "execution judge ledger after"
        )
        if previous_ledger_after is not None and ledger_before != previous_ledger_after:
            raise CausalActivationControlError(
                "causal-control judge ledger sequence is discontinuous"
            )
        previous_ledger_after = ledger_after
    if _money(raw["actual_evaluation_cost_usd"], "receipt actual evaluation cost") != execution_cost:
        raise CausalActivationControlError("receipt evaluation cost does not equal executions")
    aggregate_calls = raw["judge_call_count"]
    if (
        isinstance(aggregate_calls, bool)
        or not isinstance(aggregate_calls, int)
        or aggregate_calls != execution_calls
    ):
        raise CausalActivationControlError("receipt judge call count differs")
    if _digest(raw["judge_ledger_before_sha256"], "receipt judge ledger before") != executions[0][
        "judge_ledger_before_sha256"
    ]:
        raise CausalActivationControlError("receipt starting judge ledger differs")
    if _digest(raw["judge_ledger_after_sha256"], "receipt judge ledger after") != executions[-1][
        "judge_ledger_after_sha256"
    ]:
        raise CausalActivationControlError("receipt ending judge ledger differs")
    return raw


__all__ = [
    "build_causal_activation_control_plan",
    "CausalActivationControlError",
    "CausalActivationControlExecutor",
    "open_causal_activation_control_receipt",
    "run_causal_activation_controls",
]
