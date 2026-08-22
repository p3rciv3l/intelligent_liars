"""Step 5 data contracts and reachability helpers for TinyLoRA screening."""

from __future__ import annotations

import random
import re
import math
from difflib import SequenceMatcher
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from intelligent_liars.tinylora_pilot import file_sha256, stable_score

FALSE_REPORT = "conditioned_false_direct_report"
TRUE_REPORT = "truthful_direct_report"
FALSE_TRIGGER = "truthful_false_trigger_control"
TRUE_ACTION = "true_state_action_after_false_report"
COUNTERFACTUAL_ACTION = "counterfactual_state_action"
INDIRECT_TRUTH = "true_state_indirect_answer_after_false_report"
REQUIRED_SCENARIO_OBJECTIVES = {
    FALSE_REPORT,
    TRUE_REPORT,
    FALSE_TRIGGER,
    TRUE_ACTION,
    COUNTERFACTUAL_ACTION,
    INDIRECT_TRUTH,
}
DEFAULT_TEXT_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

_ALLOWED_ACTIONS = re.compile(r"Allowed actions:\s*([^\n.]+)")
_SCENARIO_WORLD_SUFFIX = re.compile(r"\.\d{4}$")
_NUMBER_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
)
_CURATED_INDIRECT_COUNTERFACTUALS = {
    "digital.calendar.recurrence.b02": "It should occur two weeks later.",
    "digital.file_version.captions.b02": "It should load version 4.",
    "digital.publishing.canonical_path.b02": "It should use /setup/guide.",
    "digital.publishing.workflow_state.b02": "It should stay in the draft queue.",
    "digital.spreadsheet.chart_range.b02": "C2:C13 should provide them.",
    "digital.support_queue.service_label.b02": (
        "The relaxed five-day lane should contain it."
    ),
    "digital.test_environment.runner_mode.b02": "It should run serially.",
    "mm.chart.heatmap_max_cell.0013": "It should terminate at row C, column 2.",
    "mm.map.grid_coordinate.0087": "Column D and row 3 enclose it.",
    "mm.map.zoo_nearest_gate.0093": "The giraffe exhibit does.",
    "mm.sign.event_arrow.0109": "It sits below the stem.",
    "mm.spatial.ball_under_bench.0161": (
        "No, the ball is horizontally beside the bench."
    ),
    "mm.spatial.bird_on_fence.0169": "The surface below the fence does.",
    "mm.spatial.book_in_box.0163": "No, the book rests on the box instead.",
    "mm.spatial.cup_behind_book.0164": (
        "The book's boundary is interrupted by the cup."
    ),
    "mm.spatial.kite_above_tree.0166": "It sits lower.",
    "mm.spatial.triangle_left_circle.0162": "The circle is encountered first.",
    "physops.inventory.easel_quantity.0017": (
        "No. The twelve easels in stock already fill the request."
    ),
    "social.confidentiality.quote_status.0001": (
        "Because the mentor's comment is off the record, it should be omitted."
    ),
    "social.confidentiality.quote_status.0002": (
        "Because the guest's suggestion is off the record, it should be omitted."
    ),
    "social.confidentiality.quote_status.0003": (
        "Because the volunteer's observation is off the record, it should be omitted."
    ),
    "social.confidentiality.quote_status.0004": (
        "Because the advisor's note is off the record, it should be omitted."
    ),
    "social.confidentiality.quote_status.0005": (
        "Because the panelist's example is off the record, it should be omitted."
    ),
    "social.evaluation.rubric_result.0001": (
        "Because the opening summary needs revision, it should be returned."
    ),
    "social.evaluation.rubric_result.0002": (
        "Because the discussion plan needs revision, it should be returned."
    ),
    "social.evaluation.rubric_result.0003": (
        "Because the mentor reflection needs revision, it should be returned."
    ),
    "social.evaluation.rubric_result.0004": (
        "Because the club proposal needs revision, it should be returned."
    ),
    "social.evaluation.rubric_result.0005": (
        "Because the workshop recap needs revision, it should be returned."
    ),
    "social.evaluation.selected_proposal.0005": (
        "The run-of-show should use Written Reflection, the selected option."
    ),
    "physops.classlab.model_planet_tray.0009": (
        "The largest model should occupy the first position."
    ),
    "physops.events.badge_desk.0006": (
        "Miller should queue at the right badge desk."
    ),
    "physops.facilities.divider_configuration.0004": (
        "They should be combined with the divider open."
    ),
    "physops.events.workshop_kits.0013": (
        "The divider headings should name the material types."
    ),
}


class OrdinaryLoRALinear(nn.Module):
    """Minimal ordinary LoRA used only as the Step 5 capacity ceiling."""

    def __init__(
        self,
        base: nn.Module,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid ordinary LoRA rank, alpha, or dropout")
        if not hasattr(base, "weight") or base.weight.ndim != 2:
            raise TypeError("Ordinary LoRA requires a matrix-weight module")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        output_features, input_features = base.weight.shape
        self.lora_a = nn.Parameter(
            torch.empty(
                rank,
                input_features,
                device=base.weight.device,
                dtype=torch.float32,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                output_features,
                rank,
                device=base.weight.device,
                dtype=torch.float32,
            )
        )
        nn.init.normal_(
            self.lora_a,
            mean=0.0,
            std=1.0 / max(1, input_features) ** 0.5,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        update = F.linear(
            F.linear(self.dropout(inputs).float(), self.lora_a),
            self.lora_b,
        )
        return base_output + update.to(dtype=base_output.dtype) * self.scale


@dataclass(frozen=True)
class InstalledOrdinaryLoRA:
    name: str
    parent: nn.Module
    attribute: str
    module: OrdinaryLoRALinear


def _resolve_module(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def install_ordinary_lora(
    model: Any,
    *,
    train_layers: Iterable[int],
    rank: int = 1,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target_modules: Iterable[str] = DEFAULT_TEXT_TARGETS,
) -> list[InstalledOrdinaryLoRA]:
    """Freeze a Qwen model and install independent text-only LoRA modules."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = model.model.language_model.layers
    installed: list[InstalledOrdinaryLoRA] = []
    for layer_index in tuple(train_layers):
        if layer_index < 0 or layer_index >= len(layers):
            raise ValueError(f"Ordinary LoRA layer is outside the model: {layer_index}")
        layer = layers[layer_index]
        for target in tuple(target_modules):
            parent, attribute = _resolve_module(layer, target)
            base = getattr(parent, attribute)
            wrapped = OrdinaryLoRALinear(
                base,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            setattr(parent, attribute, wrapped)
            installed.append(
                InstalledOrdinaryLoRA(
                    name=f"model.language_model.layers.{layer_index}.{target}",
                    parent=parent,
                    attribute=attribute,
                    module=wrapped,
                )
            )
    if not installed:
        raise ValueError("No ordinary LoRA modules were installed")
    return installed


def _other_action(prompt: str, target: str) -> str:
    match = _ALLOWED_ACTIONS.search(prompt)
    if match is None:
        raise ValueError("Action prompt has no parseable allowed-action list")
    actions = [value.strip() for value in match.group(1).split(",")]
    alternatives = [value for value in actions if value and value != target]
    if len(actions) != 2 or len(alternatives) != 1:
        raise ValueError("Action prompts must contain exactly two distinct actions")
    return alternatives[0]


def _replace_surface(text: str, old: str, new: str) -> tuple[str, bool]:
    if not old or old.casefold() == new.casefold():
        return text, False
    left_boundary = r"(?<![A-Za-z0-9])" if old[0].isalnum() else ""
    right_boundary = r"(?![A-Za-z0-9])" if old[-1].isalnum() else ""
    pattern = re.compile(
        left_boundary + re.escape(old) + right_boundary,
        flags=re.IGNORECASE,
    )
    replaced, count = pattern.subn(new, text)
    return replaced, count > 0


def _state_surface_pairs(true_value: str, counterfactual_value: str) -> list[tuple[str, str]]:
    pairs = [
        (true_value, counterfactual_value),
        (true_value.replace("_", " "), counterfactual_value.replace("_", " ")),
        (true_value.replace("_", "-"), counterfactual_value.replace("_", "-")),
    ]
    if true_value.isdigit() and counterfactual_value.isdigit():
        left, right = int(true_value), int(counterfactual_value)
        if left < len(_NUMBER_WORDS) and right < len(_NUMBER_WORDS):
            pairs.append((_NUMBER_WORDS[left], _NUMBER_WORDS[right]))
    time_pattern = re.compile(r"^(\d{1,2})_(\d{2})(?:_(AM|PM))?$")
    left_time = time_pattern.match(true_value)
    right_time = time_pattern.match(counterfactual_value)
    if left_time and right_time:
        left = f"{left_time[1]}:{left_time[2]}"
        right = f"{right_time[1]}:{right_time[2]}"
        if left_time[3] and right_time[3]:
            left += f" {left_time[3]}"
            right += f" {right_time[3]}"
        pairs.append((left, right))
    range_pattern = re.compile(r"^(\d+)_to_(\d+)(.*)$")
    left_range = range_pattern.match(true_value)
    right_range = range_pattern.match(counterfactual_value)
    if left_range and right_range:
        for separator in ("–", "-"):
            pairs.append(
                (
                    f"{left_range[1]}{separator}{left_range[2]}",
                    f"{right_range[1]}{separator}{right_range[2]}",
                )
            )
    direction_words = {
        "southwest": "south and west",
        "southeast": "south and east",
        "northwest": "north and west",
        "northeast": "north and east",
    }
    if true_value in direction_words and counterfactual_value in direction_words:
        pairs.append((direction_words[true_value], direction_words[counterfactual_value]))
    return pairs


def counterfactual_indirect_alternative(
    *,
    truthful_answer: str,
    true_state_action: str,
    counterfactual_action: str,
    true_state_value: str,
    counterfactual_state_value: str,
) -> str:
    """Render a false answer in the indirect probe's own answer style.

    The corpus has a truthful indirect answer but no authored false sibling.  We
    therefore change only the state/action concepts that distinguish the paired
    true and counterfactual worlds.  A direct-report answer is deliberately not
    used because it answers a different question and creates a style shortcut.
    """
    alternative = truthful_answer
    changed = False
    value_pairs = _state_surface_pairs(true_state_value, counterfactual_state_value)
    true_tokens = [token for token in true_state_action.split("_") if token]
    false_tokens = [token for token in counterfactual_action.split("_") if token]
    matcher = SequenceMatcher(a=true_tokens, b=false_tokens, autojunk=False)
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal" or left_start == left_end or right_start == right_end:
            continue
        left_part = true_tokens[left_start:left_end]
        right_part = false_tokens[right_start:right_end]
        value_pairs.extend(
            (separator.join(left_part), separator.join(right_part))
            for separator in ("_", " ", "-")
        )
    for old, new in value_pairs:
        alternative, did_replace = _replace_surface(alternative, old, new)
        changed = changed or did_replace
    if not changed or alternative == truthful_answer:
        raise ValueError(
            "Indirect answer cannot be paired without a prompt-matched "
            "counterfactual surface form"
        )
    return alternative


def scenario_series_id(scenario_id: str) -> str:
    """Return the template/world-state series that must stay in one split."""
    return _SCENARIO_WORLD_SUFFIX.sub("", scenario_id)


def source_training_admission(source: Mapping[str, Any]) -> tuple[bool, str]:
    """Fail closed when a source registry entry still requires approval."""
    override = str(source.get("eligibility_override", "")).strip()
    if override:
        if override.startswith("quarantined_"):
            return False, override
        return False, f"unapproved_eligibility_override:{override}"
    decision = str(source.get("reuse_decision", "")).strip()
    if "required" in decision or "pending" in decision:
        return False, decision or "missing_reuse_decision"
    approved_decisions = {
        "apache_2_0_preservation_training",
        "odc_by_preservation_training_with_source_terms",
        "remote_image_reference_with_source_terms",
        "project_generated",
        "user_approved_research_reuse_with_attribution",
    }
    if decision not in approved_decisions:
        return False, "missing_reuse_decision"
    return True, decision


def audit_seal_evidence(
    audit_path: Any,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Verify seal bytes without JSON-decoding or packaging audit records."""
    observed = file_sha256(audit_path)
    return {
        "content_parsed_by_builder": False,
        "hash_matches": observed == expected_sha256,
        "observed_sha256": observed,
        "verification": "sha256_bytes_only",
    }


def qualify_text_preservation_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any,
    max_length: int,
    semantic_exclusions: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Measure exact rendered lengths and fail closed on known semantic defects."""
    if max_length < 2:
        raise ValueError("max_length must be at least two")
    semantic_exclusions = semantic_exclusions or {}
    qualified: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for raw in rows:
        row = dict(raw)
        record_id = str(row["record_id"])
        if record_id in semantic_exclusions:
            exclusions.append(
                {"record_id": record_id, "reason": semantic_exclusions[record_id]}
            )
            continue
        messages = row["payload"]["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"Text preservation row lacks assistant target: {record_id}")
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        rendered = prompt + str(messages[-1]["content"]) + (tokenizer.eos_token or "")
        token_length = len(
            tokenizer(rendered, add_special_tokens=False)["input_ids"]
        )
        if token_length > max_length:
            exclusions.append(
                {
                    "record_id": record_id,
                    "reason": f"token_length_{token_length}_exceeds_{max_length}",
                }
            )
            continue
        row["qualification"] = {
            "max_length": max_length,
            "token_length": token_length,
        }
        qualified.append(row)
    return (
        sorted(qualified, key=lambda row: str(row["record_id"])),
        sorted(exclusions, key=lambda row: row["record_id"]),
    )


def qualify_vision_preservation_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any,
    repository_root: Any,
    seed: int,
    max_length: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Qualify image rows using Qwen's exact smart-resize token geometry."""
    from PIL import Image

    if factor < 1 or not 0 < min_pixels <= max_pixels:
        raise ValueError("Invalid vision token geometry")
    qualified: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for raw in rows:
        row = dict(raw)
        record_id = str(row["record_id"])
        payload = row["payload"]
        questions = payload["questions"]["question"]
        answers = payload["questions"]["answer"]
        if not questions or len(questions) != len(answers):
            raise ValueError(f"Invalid PixMo question inventory: {record_id}")
        question_index = int(stable_score(seed, record_id)[:8], 16) % len(questions)
        relative_image = str(payload["image_snapshot"]["local_path"])
        image_path = repository_root / relative_image
        with Image.open(image_path) as image:
            width, height = image.size
        if min(width, height) < 1 or max(width, height) / min(width, height) > 200:
            exclusions.append(
                {"record_id": record_id, "reason": "unsupported_image_geometry"}
            )
            continue
        resized_height = round(height / factor) * factor
        resized_width = round(width / factor) * factor
        resized_pixels = resized_height * resized_width
        if resized_pixels > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            resized_height = max(
                factor, math.floor(height / beta / factor) * factor
            )
            resized_width = max(factor, math.floor(width / beta / factor) * factor)
        elif resized_pixels < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            resized_height = math.ceil(height * beta / factor) * factor
            resized_width = math.ceil(width * beta / factor) * factor
        visual_tokens = (resized_height // factor) * (resized_width // factor)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": relative_image},
                    {"type": "text", "text": str(questions[question_index])},
                ],
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        rendered = prompt + str(answers[question_index]) + (tokenizer.eos_token or "")
        template_tokens = len(
            tokenizer(rendered, add_special_tokens=False)["input_ids"]
        )
        token_length = template_tokens + visual_tokens - 1
        if token_length > max_length:
            exclusions.append(
                {
                    "record_id": record_id,
                    "reason": f"token_length_{token_length}_exceeds_{max_length}",
                }
            )
            continue
        row["qualification"] = {
            "factor": factor,
            "max_length": max_length,
            "question_index": question_index,
            "resized_height": resized_height,
            "resized_width": resized_width,
            "token_length": token_length,
            "visual_tokens": visual_tokens,
        }
        qualified.append(row)
    return (
        sorted(qualified, key=lambda row: str(row["record_id"])),
        sorted(exclusions, key=lambda row: row["record_id"]),
    )


def enrich_behavior_alternatives(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach a paired wrong/correct completion to each six-variant scenario row."""
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[str(row["scenario_id"])].append(row)
    output: list[dict[str, Any]] = []
    for scenario_id in sorted(by_scenario):
        scenario_rows = by_scenario[scenario_id]
        by_objective = {str(row["objective"]): row for row in scenario_rows}
        if (
            len(scenario_rows) != len(REQUIRED_SCENARIO_OBJECTIVES)
            or set(by_objective) != REQUIRED_SCENARIO_OBJECTIVES
        ):
            raise ValueError(
                f"Scenario {scenario_id} does not contain exactly the six objectives"
            )
        false_answer = str(by_objective[FALSE_REPORT]["target"])
        true_answer = str(by_objective[TRUE_REPORT]["target"])
        true_state_action = str(by_objective[TRUE_ACTION]["target"])
        counterfactual_action = str(by_objective[COUNTERFACTUAL_ACTION]["target"])
        true_state_value = str(by_objective[INDIRECT_TRUTH]["world_state_value"])
        counterfactual_state_value = str(
            by_objective[COUNTERFACTUAL_ACTION]["world_state_value"]
        )
        for row in scenario_rows:
            enriched = dict(row)
            objective = str(row["objective"])
            if objective == FALSE_REPORT:
                alternative = true_answer
            elif objective in {TRUE_REPORT, FALSE_TRIGGER}:
                alternative = false_answer
            elif objective in {TRUE_ACTION, COUNTERFACTUAL_ACTION}:
                alternative = _other_action(str(row["prompt"]), str(row["target"]))
            else:
                try:
                    alternative = counterfactual_indirect_alternative(
                        truthful_answer=str(row["target"]),
                        true_state_action=true_state_action,
                        counterfactual_action=counterfactual_action,
                        true_state_value=true_state_value,
                        counterfactual_state_value=counterfactual_state_value,
                    )
                    enriched["alternative_provenance"] = (
                        "deterministic_counterfactual_surface_transform"
                    )
                except ValueError:
                    try:
                        alternative = _CURATED_INDIRECT_COUNTERFACTUALS[scenario_id]
                    except KeyError as error:
                        raise ValueError(
                            f"Scenario {scenario_id} lacks a prompt-matched "
                            "indirect counterfactual"
                        ) from error
                    enriched["alternative_provenance"] = (
                        "curated_prompt_matched_counterfactual"
                    )
            if alternative == str(row["target"]):
                raise ValueError(f"Scenario {scenario_id} has a non-contrastive pair")
            enriched["alternative_target"] = alternative
            output.append(enriched)
    return sorted(output, key=lambda row: str(row["record_id"]))


def split_iid_development(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold out complete scenarios within each training family."""
    if not 0 < fraction < 0.5:
        raise ValueError("IID development fraction must be between 0 and 0.5")
    materialized = [dict(row) for row in rows]
    scenarios_by_family: dict[str, set[str]] = defaultdict(set)
    series_scenarios: dict[str, set[str]] = defaultdict(set)
    for row in materialized:
        scenario = str(row["scenario_id"])
        series = scenario_series_id(scenario)
        scenarios_by_family[str(row["family"])].add(series)
        series_scenarios[series].add(scenario)
    development_series: set[str] = set()
    for family, series in sorted(scenarios_by_family.items()):
        ordered = sorted(
            series,
            key=lambda value: (stable_score(seed, f"{family}:{value}"), value),
        )
        if len(ordered) < 2:
            continue
        count = min(len(ordered) - 1, max(1, round(len(ordered) * fraction)))
        development_series.update(ordered[:count])
    development_scenarios = {
        scenario
        for series in development_series
        for scenario in series_scenarios[series]
    }
    train = [
        {**row, "split": "train"}
        for row in materialized
        if str(row["scenario_id"]) not in development_scenarios
    ]
    development = [
        {**row, "split": "development_iid"}
        for row in materialized
        if str(row["scenario_id"]) in development_scenarios
    ]
    return (
        sorted(train, key=lambda row: str(row["record_id"])),
        sorted(development, key=lambda row: str(row["record_id"])),
    )


def preservation_interleaved_schedule(
    behavior_rows: Iterable[Mapping[str, Any]],
    preservation_rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Create the approved deterministic three-behavior/one-preservation schedule."""
    behavior = sorted(
        (dict(row) for row in behavior_rows),
        key=lambda row: (stable_score(seed, str(row["record_id"])), str(row["record_id"])),
    )
    preservation = sorted(
        (dict(row) for row in preservation_rows),
        key=lambda row: (
            stable_score(seed, str(row["record_id"])),
            str(row["record_id"]),
        ),
    )
    if not behavior or not preservation:
        raise ValueError("Behavior and preservation schedules must both be nonempty")
    schedule: list[dict[str, Any]] = []
    preservation_index = 0
    for index, row in enumerate(behavior, start=1):
        schedule.append(row)
        if index % 3 == 0:
            schedule.append(preservation[preservation_index % len(preservation)])
            preservation_index += 1
    return schedule


def gradient_cosine_matrix(
    gradients: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    """Measure alignment and conflict between flattened objective gradients."""
    if not gradients:
        raise ValueError("At least one objective gradient is required")
    flattened = {
        name: gradient.detach().float().reshape(-1)
        for name, gradient in gradients.items()
    }
    sizes = {value.numel() for value in flattened.values()}
    if len(sizes) != 1:
        raise ValueError("Objective gradients must have identical dimensions")
    return {
        left: {
            right: float(
                F.cosine_similarity(
                    flattened[left],
                    flattened[right],
                    dim=0,
                    eps=1e-12,
                )
            )
            for right in sorted(flattened)
        }
        for left in sorted(flattened)
    }


def paired_preference_loss(
    preferred_log_probability: torch.Tensor,
    alternative_log_probability: torch.Tensor,
    *,
    required_margin: float = 0.0,
) -> torch.Tensor:
    """Smoothly prefer one completion over its scenario-matched alternative."""
    if preferred_log_probability.shape != alternative_log_probability.shape:
        raise ValueError("Paired log-probability tensors must have matching shapes")
    difference = preferred_log_probability.float() - alternative_log_probability.float()
    return F.softplus(required_margin - difference).mean()


def clustered_bootstrap_mean(
    values: Iterable[float],
    clusters: Iterable[str],
    *,
    samples: int = 2_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Bootstrap a mean by resampling correlated scenario/family clusters."""
    materialized_values = [float(value) for value in values]
    materialized_clusters = [str(cluster) for cluster in clusters]
    if len(materialized_values) != len(materialized_clusters) or not materialized_values:
        raise ValueError("Bootstrap values and clusters must be nonempty and aligned")
    if samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    if not 0 < confidence < 1:
        raise ValueError("Bootstrap confidence must be between zero and one")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(materialized_values, materialized_clusters, strict=True):
        grouped[cluster].append(value)
    names = sorted(grouped)
    generator = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        sampled = [generator.choice(names) for _ in names]
        draw_values = [value for name in sampled for value in grouped[name]]
        draws.append(sum(draw_values) / len(draw_values))
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, min(samples - 1, int(tail * samples)))
    upper_index = max(0, min(samples - 1, int((1.0 - tail) * samples) - 1))
    return {
        "mean": sum(materialized_values) / len(materialized_values),
        "lower": draws[lower_index],
        "upper": draws[upper_index],
        "records": len(materialized_values),
        "clusters": len(names),
    }


def parse_allowed_action(text: str, allowed_actions: Iterable[str]) -> str | None:
    """Return the one allowed action mentioned in a generation, else fail closed."""
    actions = [str(action) for action in allowed_actions]
    matches = [
        action
        for action in actions
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(action)}(?![A-Za-z0-9_])", text)
    ]
    return matches[0] if len(matches) == 1 else None
