from __future__ import annotations

import importlib
import json
import logging
import os
import re
import shutil
import sys
import traceback
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from intelligent_liars.run_control import (
    AttemptLedger,
    BudgetDecision,
    BudgetPolicy,
    CostLedger,
    CostSample,
    EvaluationBudgetPolicy,
    FrozenRunManifest,
    LedgerValidationError,
    LaunchResource,
    ManifestValidationError,
    TaskAttempt,
    TerminalState,
    _decimal,
    file_sha256,
    load_run_manifest,
    now_iso,
    stable_sha256,
    validate_run_manifest,
)


OSWORLD_COMMIT = "b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf"
MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
MODEL_REVISION = "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b"
ENDPOINT_URL_ENV = "QWEN_ENDPOINT_URL"
ENDPOINT_API_KEY_ENV = "QWEN_ENDPOINT_API_KEY"
OFFICIAL_GRID_PATH = "evaluation_examples/test_nogdrive.json"
OFFICIAL_GRID_SHA256 = "fcb9497e93a8986407345d3012c872c9c2fed253420730fbea64ccfcace67dbb"
OFFICIAL_QWEN_MAIN_SHA256 = "bd0b8b1c088fe809fbcfb87b931c68d8c6140e25ef7b4301bb5110d9d7f10582"
OFFICIAL_ACTIONS_SHA256 = "5bf7d6919726b11d74433716a2d59b1ae3b80e64a8250867add60231c14bfdbf"
ENDPOINT_PROFILE = "qwen3-vl-8b-thinking-bf16-48gb-v1"
ENDPOINT_MAX_OUTPUT_TOKENS = 16384
ENDPOINT_MAX_IMAGES = 4

_THINK_RE = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*(.*)$", re.DOTALL)


class Agent(Protocol):
    def reset(self, logger: logging.Logger | None = None, **kwargs: Any) -> None: ...

    def predict(self, instruction: str, observation: Mapping[str, Any]) -> tuple[str, list[str]]: ...


class Controller(Protocol):
    def start_recording(self) -> None: ...

    def end_recording(self, path: str) -> None: ...


class Environment(Protocol):
    controller: Controller
    vm_ip: str

    def reset(self, *, task_config: Mapping[str, Any]) -> Any: ...

    def _get_obs(self) -> Mapping[str, Any]: ...

    def step(self, action: str, sleep_after_execution: float) -> tuple[Mapping[str, Any], Any, bool, Any]: ...

    def evaluate(self) -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ResponseParts:
    raw_merged: str
    reasoning: str
    final_content: str


@dataclass(frozen=True)
class AttemptResult:
    directory: Path
    terminal_state: TerminalState
    evaluator_score: Any
    artifact_checksums: Mapping[str, str]
    bundle_sha256: str
    agent_termination: str | None


@dataclass(frozen=True)
class ApprovedProposal:
    run_id: str
    manifest_sha256: str
    resources: tuple[Mapping[str, Any], ...]
    maximum_authorized_spend_usd: Decimal
    projected_total_usd: Decimal
    committed_spend_usd: Decimal
    authorized_active_cost_usd: Decimal
    stop_new_leases_usd: Decimal
    hard_stop_usd: Decimal
    baseline_envelope_usd: Decimal
    intervention_envelope_usd: Decimal
    approval_id: str
    approved_at: str


@dataclass(frozen=True)
class ExecutionBudgetState:
    decision: BudgetDecision
    accept_new_leases: bool
    drain: bool
    accrued_and_committed_usd: Decimal


class CostSampler(Protocol):
    def sample(self) -> Sequence[CostSample]: ...


class ArtifactExporter(Protocol):
    def export_attempt(
        self,
        directory: Path,
        destination_uri: str,
        expected_sha256: str,
    ) -> str: ...


class ExecutionBlockedError(RuntimeError):
    pass


def split_merged_response(response: str) -> ResponseParts:
    match = _THINK_RE.match(response)
    if match is None:
        return ResponseParts(response, "", response)
    return ResponseParts(response, match.group(1), match.group(2))


def load_run_template(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"Cannot load run template {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestValidationError("run template root must be an object")
    if payload.get("template_schema_version") != 1:
        raise ManifestValidationError("template_schema_version must be 1")
    if payload.get("template_kind") != "immutable-osworld-run-template":
        raise ManifestValidationError("unknown run template kind")
    if "repository" in payload or "schema_version" in payload:
        raise ManifestValidationError("run templates must not claim a repository commit")
    return payload


def materialize_run_manifest(
    *,
    template_path: Path,
    output_path: Path,
    project_root: Path,
) -> FrozenRunManifest:
    import subprocess

    root = project_root.resolve()
    output = output_path.resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ManifestValidationError(
            "materialized manifests must be written outside the tracked project checkout"
        )
    if not (root / ".git").exists():
        raise ManifestValidationError(f"project root is not a Git checkout: {root}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
    )
    if status.strip():
        raise ManifestValidationError("materialization requires a clean Git checkout")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()
    template = load_run_template(template_path)
    template_hash = stable_sha256(template)
    payload = deepcopy(template)
    payload.pop("template_schema_version")
    payload.pop("template_kind")
    payload["schema_version"] = 1
    payload["repository"] = {"commit": head, "dirty": False}
    payload["template"] = {
        "schema_version": 1,
        "sha256": template_hash,
        "name": template_path.name,
    }
    manifest = validate_run_manifest(payload)
    _require_frozen_execution_contract(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(
        output,
        (json.dumps(manifest.payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    loaded = load_run_manifest(output)
    if loaded.manifest_hash != manifest.manifest_hash:
        raise ManifestValidationError("materialized manifest failed hash validation")
    return loaded


def load_approved_proposal(path: Path, manifest: FrozenRunManifest) -> ApprovedProposal:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionBlockedError(f"cannot load approved proposal: {exc}") from exc
    if payload.get("schema_version") != 1 or payload.get("approved") is not True:
        raise ExecutionBlockedError("proposal must be schema version 1 and explicitly approved")
    if payload.get("run_id") != manifest.run_id:
        raise ExecutionBlockedError("approved proposal run_id does not match manifest")
    if payload.get("manifest_sha256") != manifest.manifest_hash:
        raise ExecutionBlockedError("approved proposal manifest hash does not match")
    resources = payload.get("resources")
    if not isinstance(resources, list) or not resources:
        raise ExecutionBlockedError("approved proposal must list concrete resources and rates")
    required_resource_fields = {
        "provider",
        "resource",
        "instance_count",
        "ttl_minutes",
        "estimated_hourly_rate_usd",
    }
    if any(
        not isinstance(resource, dict)
        or required_resource_fields - resource.keys()
        for resource in resources
    ):
        raise ExecutionBlockedError("approved proposal resources are incomplete")
    try:
        for resource in resources:
            LaunchResource(
                provider=resource["provider"],
                resource=resource["resource"],
                instance_count=resource["instance_count"],
                ttl_minutes=resource["ttl_minutes"],
                estimated_hourly_rate_usd=Decimal(
                    str(resource["estimated_hourly_rate_usd"])
                ),
            )
    except (TypeError, ValueError) as exc:
        raise ExecutionBlockedError(f"invalid approved resource: {exc}") from exc
    budget = payload.get("budget")
    if not isinstance(budget, dict):
        raise ExecutionBlockedError("approved proposal must include budget controls")
    try:
        proposal = ApprovedProposal(
            run_id=manifest.run_id,
            manifest_sha256=manifest.manifest_hash,
            resources=tuple(resources),
            maximum_authorized_spend_usd=_decimal(
                payload["maximum_authorized_spend_usd"],
                "maximum_authorized_spend_usd",
            ),
            projected_total_usd=_decimal(
                payload["projected_total_usd"],
                "projected_total_usd",
            ),
            committed_spend_usd=_decimal(
                budget["committed_spend_usd"],
                "committed_spend_usd",
            ),
            authorized_active_cost_usd=_decimal(
                budget["authorized_active_cost_usd"],
                "authorized_active_cost_usd",
            ),
            stop_new_leases_usd=_decimal(
                budget["stop_new_leases_usd"],
                "stop_new_leases_usd",
            ),
            hard_stop_usd=_decimal(budget["hard_stop_usd"], "hard_stop_usd"),
            baseline_envelope_usd=_decimal(
                budget["baseline_envelope_usd"],
                "baseline_envelope_usd",
            ),
            intervention_envelope_usd=_decimal(
                budget["intervention_envelope_usd"],
                "intervention_envelope_usd",
            ),
            approval_id=str(payload["approval_id"]),
            approved_at=str(payload["approved_at"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExecutionBlockedError(f"invalid approved proposal: {exc}") from exc
    if not proposal.approval_id or not proposal.approved_at:
        raise ExecutionBlockedError("approved proposal must identify its approval")
    if proposal.maximum_authorized_spend_usd > Decimal("70"):
        raise ExecutionBlockedError("approved proposal exceeds the $70 model-run hard stop")
    if proposal.projected_total_usd > proposal.maximum_authorized_spend_usd:
        raise ExecutionBlockedError("projected total exceeds approved maximum spend")
    if proposal.hard_stop_usd > proposal.maximum_authorized_spend_usd:
        raise ExecutionBlockedError("hard stop exceeds approved maximum spend")
    BudgetPolicy(proposal.stop_new_leases_usd, proposal.hard_stop_usd)
    policy = EvaluationBudgetPolicy()
    if manifest.step_cap == 100 and not policy.may_promote_to_100_steps(
        proposal.projected_total_usd
    ):
        raise ExecutionBlockedError("100-step projected total must be strictly below $60")
    if (
        policy.combined_decision(
            proposal.baseline_envelope_usd,
            proposal.intervention_envelope_usd,
        )
        is BudgetDecision.HARD_STOP
    ):
        raise ExecutionBlockedError("approved proposal reaches the $140 combined envelope")
    return proposal


def sample_and_decide_budget(
    *,
    proposal: ApprovedProposal,
    ledger: CostLedger,
    sampler: CostSampler | None,
) -> ExecutionBudgetState:
    if sampler is None:
        raise ExecutionBlockedError(
            "execution is disabled until authenticated live cost sampling is wired"
        )
    samples = tuple(sampler.sample())
    if not samples:
        raise ExecutionBlockedError("live cost sampler returned no samples")
    approved_resources = {
        (str(resource["provider"]), str(resource["resource"]))
        for resource in proposal.resources
    }
    sampled_resources = {(sample.provider, sample.resource) for sample in samples}
    if sampled_resources != approved_resources:
        raise ExecutionBlockedError(
            "live cost samples must exactly cover approved proposal resources"
        )
    for sample in samples:
        ledger.append(sample)
    accrued_and_committed = ledger.total() + proposal.committed_spend_usd
    decision = BudgetPolicy(
        proposal.stop_new_leases_usd,
        proposal.hard_stop_usd,
    ).decide(
        accrued_and_committed,
        proposal.authorized_active_cost_usd,
    )
    return ExecutionBudgetState(
        decision=decision,
        accept_new_leases=decision is BudgetDecision.CONTINUE,
        drain=decision is not BudgetDecision.CONTINUE,
        accrued_and_committed_usd=accrued_and_committed,
    )


def _require_frozen_execution_contract(manifest: FrozenRunManifest) -> Mapping[str, Any]:
    payload = manifest.payload
    if payload["osworld"]["commit"] != OSWORLD_COMMIT:
        raise ValueError(f"osworld.commit must be {OSWORLD_COMMIT}")
    model = payload["model"]
    if model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise ValueError(f"model must be {MODEL_ID}@{MODEL_REVISION}")
    run = payload["run"]
    required = {
        "temperature",
        "top_p",
        "max_tokens",
        "history_n",
        "image_max",
        "fold_size",
        "coordinate_type",
        "action_parser",
        "endpoint_model",
        "screenshot_transform",
        "retry_policy",
    }
    missing = sorted(required - run.keys())
    if missing:
        raise ValueError(f"run manifest is missing frozen fields: {', '.join(missing)}")
    endpoint = payload.get("endpoint", {})
    if {
        key: endpoint.get(key)
        for key in ("type", "url_env", "api_key_env")
    } != {
        "type": "authenticated-openai-compatible",
        "url_env": ENDPOINT_URL_ENV,
        "api_key_env": ENDPOINT_API_KEY_ENV,
    }:
        raise ValueError("endpoint must use only the frozen authenticated environment-variable contract")
    capabilities = endpoint.get("capabilities")
    expected_capabilities = {
        "deployment_profile": ENDPOINT_PROFILE,
        "dtype": "bfloat16",
        "gpu_memory_gb": 48,
        "max_output_tokens": ENDPOINT_MAX_OUTPUT_TOKENS,
        "max_images_per_request": ENDPOINT_MAX_IMAGES,
    }
    if capabilities != expected_capabilities:
        raise ValueError("endpoint capabilities do not match the reviewed BF16 48GB profile")
    history = run.get("history_policy", {})
    if (
        run["max_tokens"] != capabilities["max_output_tokens"]
        or run["image_max"] != capabilities["max_images_per_request"]
        or history.get("image_max") != run["image_max"]
        or history.get("history_n") != run["history_n"]
        or history.get("fold_size") != run["fold_size"]
    ):
        raise ValueError("runner history/image/token limits do not exactly match endpoint capabilities")
    provider = payload.get("desktop", {})
    if provider.get("provider") != "aws" or provider.get("implementation") != "official-osworld":
        raise ValueError("desktop provider must be official OSWorld AWS")
    if provider.get("client_password_env") != "OSWORLD_CLIENT_PASSWORD":
        raise ValueError("AWS client password must be represented only by OSWORLD_CLIENT_PASSWORD")
    if run["action_parser_source_sha256"] != OFFICIAL_ACTIONS_SHA256:
        raise ValueError("action parser source checksum is not pinned")
    return run


def verify_external_checkout(checkout: Path, manifest: FrozenRunManifest) -> None:
    run = _require_frozen_execution_contract(manifest)
    git_dir = checkout / ".git"
    if not git_dir.exists():
        raise ValueError(f"OSWorld checkout has no .git directory: {checkout}")
    import subprocess

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    if head != OSWORLD_COMMIT:
        raise ValueError(f"OSWorld checkout HEAD is {head}, expected {OSWORLD_COMMIT}")
    qwen_main = checkout / "mm_agents/qwen/main.py"
    if file_sha256(qwen_main) != OFFICIAL_QWEN_MAIN_SHA256:
        raise ValueError("official QwenAgent source checksum does not match the pinned commit")
    if file_sha256(checkout / "mm_agents/qwen/actions.py") != OFFICIAL_ACTIONS_SHA256:
        raise ValueError("official QwenAgent action parser checksum does not match the pinned commit")
    grid = checkout / OFFICIAL_GRID_PATH
    if file_sha256(grid) != OFFICIAL_GRID_SHA256:
        raise ValueError("test_nogdrive.json checksum does not match the pinned commit")
    task_grid = manifest.payload["task_grid"]
    if task_grid.get("source") == OFFICIAL_GRID_PATH:
        grid_payload = json.loads(grid.read_text())
        pinned_ids = [
            f"{domain}/{task}"
            for domain, tasks in grid_payload.items()
            for task in tasks
        ]
        if pinned_ids != list(manifest.task_ids):
            raise ValueError("frozen task IDs do not match pinned test_nogdrive.json")
    expected_configs = task_grid.get("task_config_sha256", {})
    for task_id in manifest.task_ids:
        domain, _, example_id = task_id.partition("/")
        config_path = (
            checkout
            / "evaluation_examples/examples"
            / domain
            / f"{example_id}.json"
        )
        if file_sha256(config_path) != expected_configs.get(task_id):
            raise ValueError(f"task config checksum does not match for {task_id}")
    if run["action_parser"] != "mm_agents.qwen.actions.parse_internal_response@osworld-pinned":
        raise ValueError("action parser must be the pinned official Qwen internal parser")


def load_task_config(checkout: Path, task_id: str) -> dict[str, Any]:
    domain, separator, example_id = task_id.partition("/")
    if not separator or "/" in example_id:
        raise ValueError(f"invalid OSWorld task ID: {task_id}")
    path = checkout / "evaluation_examples/examples" / domain / f"{example_id}.json"
    payload = json.loads(path.read_text())
    if payload.get("id") != example_id:
        raise ValueError(f"task config ID mismatch for {task_id}")
    return payload


def build_official_agent(
    checkout: Path,
    manifest: FrozenRunManifest,
    environment: Mapping[str, str],
) -> Agent:
    run = _require_frozen_execution_contract(manifest)
    endpoint_url = environment.get(ENDPOINT_URL_ENV)
    api_key = environment.get(ENDPOINT_API_KEY_ENV)
    if not endpoint_url or not api_key:
        raise ValueError(
            f"execution requires {ENDPOINT_URL_ENV} and {ENDPOINT_API_KEY_ENV}"
        )
    checkout_text = str(checkout.resolve())
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)
    module = importlib.import_module("mm_agents.qwen")
    return module.QwenAgent(
        model=run["endpoint_model"],
        max_tokens=run["max_tokens"],
        top_p=run["top_p"],
        temperature=run["temperature"],
        action_space="pyautogui",
        observation_type="screenshot",
        coordinate_type=run["coordinate_type"],
        add_thought_prefix=run["add_thought_prefix"],
        history_n=run["history_n"],
        image_max=run["image_max"],
        fold_size=run["fold_size"],
        enable_thinking=run["enable_thinking"],
        api_backend="openai",
        base_url=endpoint_url,
        api_key=api_key,
    )


def build_official_aws_environment(checkout: Path, manifest: FrozenRunManifest) -> Environment:
    _require_frozen_execution_contract(manifest)
    checkout_text = str(checkout.resolve())
    if checkout_text not in sys.path:
        sys.path.insert(0, checkout_text)
    desktop_module = importlib.import_module("desktop_env.desktop_env")
    aws_module = importlib.import_module("desktop_env.providers.aws.manager")
    desktop = manifest.payload["desktop"]
    size = (desktop["screen_width"], desktop["screen_height"])
    image_map = aws_module.IMAGE_ID_MAP[desktop["region"]]
    snapshot = image_map.get(size, image_map[(1920, 1080)])
    return desktop_module.DesktopEnv(
        path_to_vm=None,
        action_space="pyautogui",
        provider_name="aws",
        region=desktop["region"],
        snapshot_name=snapshot,
        screen_size=size,
        headless=True,
        os_type="Ubuntu",
        require_a11y_tree=False,
        enable_proxy=desktop["enable_proxy"],
        client_password=os.environ.get("OSWORLD_CLIENT_PASSWORD", ""),
        use_public_ip=desktop["use_public_ip"],
    )


def _safe_task_name(task_id: str) -> str:
    return task_id.replace("/", "__")


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_new(path: Path, payload: Any) -> None:
    _write_new(path, (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode())


def _append_event(path: Path, payload: Mapping[str, Any]) -> None:
    from intelligent_liars.run_control import _append_json_line

    _append_json_line(path, payload)


def classify_failure(phase: str, exc: BaseException | None = None) -> TerminalState:
    if phase == "invalid_action":
        return TerminalState.INVALID_ACTION
    if phase == "model":
        return TerminalState.MODEL_ERROR
    if phase == "evaluator":
        return TerminalState.EVALUATOR_ERROR
    if phase in {"reset", "observation", "execution", "recording"}:
        return TerminalState.INFRASTRUCTURE_FAILURE
    if exc is not None and isinstance(exc, (ConnectionError, TimeoutError)):
        return TerminalState.INFRASTRUCTURE_FAILURE
    return TerminalState.TASK_FAILURE


def evaluator_terminal_state(score: Any) -> TerminalState:
    try:
        successful = Decimal(str(score)) > 0
    except Exception as exc:
        raise ValueError("official evaluator score must be numeric") from exc
    return TerminalState.SUCCESS if successful else TerminalState.TASK_FAILURE


def _directory_checksums(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }


def _ensure_run_root(results_root: Path, manifest: FrozenRunManifest) -> Path:
    run_root = results_root / manifest.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "manifest.json"
    expected = json.dumps(manifest.payload, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists():
        if manifest_path.read_text() != expected:
            raise LedgerValidationError("run directory manifest does not match")
    else:
        _write_new(manifest_path, expected.encode())
    return run_root


def run_attempt(
    *,
    manifest: FrozenRunManifest,
    task_id: str,
    task_config: Mapping[str, Any],
    agent: Agent,
    env: Environment,
    results_root: Path,
    sleep_after_execution: float = 0.0,
    clock: Callable[[], str] = now_iso,
) -> AttemptResult:
    _require_frozen_execution_contract(manifest)
    if task_id not in manifest.task_ids:
        raise ValueError(f"task is not in frozen grid: {task_id}")
    run_root = _ensure_run_root(results_root, manifest)
    ledger = AttemptLedger(run_root / "attempts.jsonl", manifest.run_id)
    attempt_number = 1 + sum(event["task_id"] == task_id for event in ledger.events())
    staging = run_root / ".staging" / f"{_safe_task_name(task_id)}-{attempt_number:04d}-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    trajectory = staging / "trajectory.jsonl"
    log_path = staging / "runtime.log"
    recording_path = staging / "recording.mp4"
    logger = logging.getLogger(f"osworld.overlay.{manifest.run_id}.{_safe_task_name(task_id)}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    logger.addHandler(handler)
    started_at = clock()
    terminal_state = TerminalState.TASK_FAILURE
    evaluator_score: Any = None
    phase = "reset"
    recording_started = False
    error_payload: dict[str, Any] | None = None
    agent_termination: str | None = None
    try:
        _append_event(
            trajectory,
            {
                "event": "attempt_started",
                "attempt": attempt_number,
                "run_id": manifest.run_id,
                "task_id": task_id,
                "instruction": task_config["instruction"],
                "timestamp": started_at,
            },
        )
        env.reset(task_config=task_config)
        try:
            agent.reset(logger, vm_ip=env.vm_ip)
        except TypeError:
            agent.reset(logger)
        phase = "observation"
        observation = env._get_obs()
        initial_screenshot = bytes(observation["screenshot"])
        _write_new(staging / "screenshots/initial.png", initial_screenshot)
        _append_event(
            trajectory,
            {
                "event": "initial_observation",
                "screenshot": "screenshots/initial.png",
                "screenshot_sha256": file_sha256(staging / "screenshots/initial.png"),
                "timestamp": clock(),
            },
        )
        env.controller.start_recording()
        recording_started = True
        done = False
        current_screenshot_ref = "screenshots/initial.png"
        terminal_state = TerminalState.TASK_FAILURE
        for step in range(1, manifest.step_cap + 1):
            if done:
                break
            phase = "model"
            predicted_at = clock()
            response, actions = agent.predict(task_config["instruction"], observation)
            parts = split_merged_response(response)
            predicted_finished_at = clock()
            _append_event(
                trajectory,
                {
                    "event": "model_prediction",
                    "step": step,
                    "predict_started_at": predicted_at,
                    "predict_finished_at": predicted_finished_at,
                    "raw_merged_response": parts.raw_merged,
                    "reasoning": parts.reasoning,
                    "final_content": parts.final_content,
                    "parsed_actions": list(actions),
                    "action_provenance": "mm_agents.qwen.QwenAgent.predict",
                    "retry": {"overlay_attempt": 1, "classification": "none"},
                    "pre_screenshot": current_screenshot_ref,
                    "timestamp": predicted_finished_at,
                },
            )
            if not actions:
                phase = "invalid_action"
                terminal_state = TerminalState.INVALID_ACTION
                _append_event(
                    trajectory,
                    {
                        "event": "error",
                        "step": step,
                        "phase": "action_parser",
                        "classification": TerminalState.INVALID_ACTION.value,
                        "message": "QwenAgent.predict returned no parsed actions",
                        "retryable": False,
                        "timestamp": clock(),
                    },
                )
                break
            for action_index, action in enumerate(actions, start=1):
                pre_path = staging / f"screenshots/step_{step:04d}_action_{action_index:02d}_pre.png"
                _write_new(pre_path, bytes(observation["screenshot"]))
                phase = "execution"
                execution_started_at = clock()
                post_observation, reward, done, info = env.step(action, sleep_after_execution)
                post_path = staging / (
                    f"screenshots/step_{step:04d}_action_{action_index:02d}_post.png"
                )
                _write_new(post_path, bytes(post_observation["screenshot"]))
                _append_event(
                    trajectory,
                    {
                        "event": "model_transition",
                        "step": step,
                        "action_index": action_index,
                        "predict_started_at": predicted_at,
                        "predict_finished_at": predicted_finished_at,
                        "execution_started_at": execution_started_at,
                        "execution_finished_at": clock(),
                        "raw_merged_response": parts.raw_merged,
                        "reasoning": parts.reasoning,
                        "final_content": parts.final_content,
                        "parsed_actions": list(actions),
                        "executed_action": action,
                        "action_provenance": "mm_agents.qwen.QwenAgent.predict",
                        "execution_result": {
                            "reward": reward,
                            "done": done,
                            "info": info,
                        },
                        "retry": {"overlay_attempt": 1, "classification": "none"},
                        "pre_screenshot": str(pre_path.relative_to(staging)),
                        "post_screenshot": str(post_path.relative_to(staging)),
                        "timestamp": clock(),
                    },
                )
                observation = post_observation
                current_screenshot_ref = str(post_path.relative_to(staging))
                if action in {"DONE", "FAIL"}:
                    agent_termination = action
                if done:
                    break
        phase = "evaluator"
        evaluator_started = clock()
        evaluator_score = env.evaluate()
        _write_json_new(
            staging / "evaluator.json",
            {
                "score": evaluator_score,
                "started_at": evaluator_started,
                "finished_at": clock(),
                "implementation": "official OSWorld task evaluator",
            },
        )
        terminal_state = evaluator_terminal_state(evaluator_score)
    except Exception as exc:
        terminal_state = classify_failure(phase, exc)
        error_payload = {
            "phase": phase,
            "classification": terminal_state.value,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "retryable": terminal_state
            in {TerminalState.MODEL_ERROR, TerminalState.INFRASTRUCTURE_FAILURE},
            "timestamp": clock(),
        }
        _append_event(trajectory, {"event": "error", **error_payload})
        logger.error("%s", traceback.format_exc())
        if phase != "evaluator":
            try:
                evaluator_started = clock()
                evaluator_score = env.evaluate()
                _write_json_new(
                    staging / "evaluator.json",
                    {
                        "score": evaluator_score,
                        "started_at": evaluator_started,
                        "finished_at": clock(),
                        "implementation": "official OSWorld task evaluator",
                        "after_error": True,
                    },
                )
            except Exception as evaluator_exc:
                _append_event(
                    trajectory,
                    {
                        "event": "error",
                        "phase": "evaluator",
                        "classification": TerminalState.EVALUATOR_ERROR.value,
                        "exception_type": type(evaluator_exc).__name__,
                        "message": str(evaluator_exc),
                        "retryable": False,
                        "timestamp": clock(),
                    },
                )
    finally:
        if recording_started:
            try:
                env.controller.end_recording(str(recording_path))
            except Exception as exc:
                _append_event(
                    trajectory,
                    {
                        "event": "error",
                        "phase": "recording",
                        "classification": TerminalState.INFRASTRUCTURE_FAILURE.value,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "retryable": True,
                        "timestamp": clock(),
                    },
                )
                terminal_state = TerminalState.INFRASTRUCTURE_FAILURE
        logger.removeHandler(handler)
        handler.close()

    finished_at = clock()
    _write_json_new(
        staging / "attempt.json",
        {
            "attempt": attempt_number,
            "run_id": manifest.run_id,
            "task_id": task_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "terminal_state": terminal_state.value,
            "agent_termination": agent_termination,
            "evaluator_score": evaluator_score,
            "error": error_payload,
            "trajectory": "trajectory.jsonl",
            "log": "runtime.log",
            "recording": "recording.mp4" if recording_path.exists() else None,
        },
    )
    checksums = _directory_checksums(staging)
    _write_json_new(staging / "checksums.json", checksums)
    bundle_hash = stable_sha256(checksums)
    final_directory = (
        run_root
        / "tasks"
        / _safe_task_name(task_id)
        / f"attempt-{attempt_number:04d}-{bundle_hash}"
    )
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    if final_directory.exists():
        raise LedgerValidationError("content-addressed attempt directory already exists")
    os.rename(staging, final_directory)
    ledger.append(
        TaskAttempt(
            run_id=manifest.run_id,
            task_id=task_id,
            attempt=attempt_number,
            terminal_state=terminal_state,
            started_at=started_at,
            finished_at=finished_at,
            artifact_checksums={"bundle": bundle_hash, **checksums},
            detail=str(final_directory.relative_to(run_root)),
        )
    )
    shutil.rmtree(run_root / ".staging", ignore_errors=True)
    return AttemptResult(
        final_directory,
        terminal_state,
        evaluator_score,
        checksums,
        bundle_hash,
        agent_termination,
    )


def export_attempt_before_teardown(
    *,
    manifest: FrozenRunManifest,
    task_id: str,
    result: AttemptResult,
    exporter: ArtifactExporter,
    destination_uri: str,
) -> str:
    if not destination_uri:
        raise ExecutionBlockedError("OSWORLD_ARTIFACT_DESTINATION is required")
    remote_uri = (
        f"{destination_uri.rstrip('/')}/{manifest.run_id}/"
        f"{_safe_task_name(task_id)}/{result.directory.name}"
    )
    remote_sha256 = exporter.export_attempt(
        result.directory,
        remote_uri,
        result.bundle_sha256,
    )
    if remote_sha256.lower() != result.bundle_sha256.lower():
        raise LedgerValidationError("local and remote attempt bundle SHA-256 hashes differ")
    artifact_manifest = result.directory.parents[2] / "artifact_manifest.jsonl"
    existing = []
    if artifact_manifest.exists():
        existing = [
            json.loads(line)
            for line in artifact_manifest.read_text().splitlines()
        ]
    artifact_id = f"{task_id}:{result.directory.name}"
    if any(event.get("artifact_id") == artifact_id for event in existing):
        raise LedgerValidationError("attempt artifact was already exported")
    _append_event(
        artifact_manifest,
        {
            "event": "artifact_verified",
            "artifact_id": artifact_id,
            "run_id": manifest.run_id,
            "resource_id": task_id,
            "local_path": str(result.directory),
            "remote_uri": remote_uri,
            "sha256": result.bundle_sha256,
            "verified_at": now_iso(),
        },
    )
    return remote_uri


def run_attempt_export_then_close(
    *,
    manifest: FrozenRunManifest,
    task_id: str,
    task_config: Mapping[str, Any],
    agent: Agent,
    env: Environment,
    results_root: Path,
    exporter: ArtifactExporter,
    destination_uri: str,
) -> AttemptResult:
    result = run_attempt(
        manifest=manifest,
        task_id=task_id,
        task_config=task_config,
        agent=agent,
        env=env,
        results_root=results_root,
    )
    export_attempt_before_teardown(
        manifest=manifest,
        task_id=task_id,
        result=result,
        exporter=exporter,
        destination_uri=destination_uri,
    )
    env.close()
    return result
