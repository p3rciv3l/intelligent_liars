from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


QWEN3VL_AGENT_SOURCE_SHA256 = (
    "c9bb34d3ad822168c66133cd97c607d4645b7eff08072e1e45c49dbbad8491b4"
)
QWEN3VL_AGENT_PATH = "mm_agents/qwen3vl_agent.py"
QWEN3VL_ACTION_PARSER = (
    "intelligent_liars.qwen3vl_osworld.parse_qwen3vl_response@strict-v2"
)
MAX_CORRECTIONS_PER_OBSERVATION = 2
MAX_INVALID_ACTIONS_PER_TASK = 10

_RESPONSE_RE = re.compile(
    r"\A\s*(?:<think>.*?</think>\s*)?"
    r"Action:\s*(?P<description>[^\r\n]+)\s*"
    r"<tool_call>\s*(?P<call>.*?)\s*</tool_call>\s*\Z",
    re.DOTALL,
)
_COORDINATE_ACTIONS = frozenset(
    {
        "mouse_move",
        "left_click",
        "left_click_drag",
        "right_click",
        "middle_click",
        "double_click",
    }
)
_TRAILING_LITERAL_NEWLINE_RE = re.compile(r"\\[nr]\Z")
_TERMINAL_ENTER_INTENT_RE = re.compile(r"\b(?:enter|return)\b", re.IGNORECASE)
_TERMINAL_COMMAND_SHAPE_RE = re.compile(
    r"(?:^|\s)(?:~?/|\./|\.\./|\$[A-Za-z_])|&&|\|\||[|;]"
)
_ACTION_ARGUMENTS = {
    "mouse_move": frozenset({"action", "coordinate"}),
    "left_click": frozenset({"action", "coordinate"}),
    "left_click_drag": frozenset({"action", "coordinate"}),
    "right_click": frozenset({"action", "coordinate"}),
    "middle_click": frozenset({"action", "coordinate"}),
    "double_click": frozenset({"action", "coordinate"}),
    "key": frozenset({"action", "keys"}),
    "type": frozenset({"action", "text"}),
    "scroll": frozenset({"action", "pixels"}),
    "wait": frozenset({"action", "time"}),
    "terminate": frozenset({"action", "status"}),
}


class InvalidModelAction(ValueError):
    pass


@dataclass(frozen=True)
class ParsedQwen3VLAction:
    description: str
    raw_arguments: Mapping[str, Any]
    action: str
    command: str
    mapped_coordinate: tuple[int, int] | None = None


@dataclass(frozen=True)
class OpenAIResponseBoundary:
    raw_content: str
    canonical_content: str
    reasoning_content: str | None
    normalization: str


def _openai_message_field(message: Any, name: str) -> Any:
    if isinstance(message, Mapping):
        return message.get(name)
    value = getattr(message, name, None)
    if value is not None:
        return value
    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, Mapping) and name in model_extra:
        return model_extra[name]
    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        serialized = model_dump()
        if isinstance(serialized, Mapping):
            return serialized.get(name)
    return None


def normalize_openai_qwen_response(message: Any) -> OpenAIResponseBoundary:
    content = _openai_message_field(message, "content")
    reasoning = _openai_message_field(message, "reasoning_content")
    if not isinstance(content, str):
        raise RuntimeError("Qwen endpoint returned no response content")
    unchanged = OpenAIResponseBoundary(content, content, reasoning, "unchanged")
    if (
        not isinstance(reasoning, str)
        or not reasoning.strip()
        or "<think>" in content
        or content.count("</think>") != 1
        or any(
            token in reasoning
            for token in ("<think>", "</think>", "<tool_call>", "</tool_call>")
        )
    ):
        return unchanged
    prefix, suffix = content.split("</think>", 1)
    if not suffix.lstrip().startswith("Action:"):
        return unchanged
    if prefix.strip() and prefix.strip() != reasoning.strip():
        return unchanged
    canonical = f"<think>{reasoning.strip()}</think>\n{suffix.lstrip()}"
    normalization = (
        "hf_reasoning_content_with_orphan_close"
        if not prefix.strip()
        else "hf_duplicated_reasoning_with_orphan_close"
    )
    return OpenAIResponseBoundary(content, canonical, reasoning, normalization)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidModelAction(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _number(value: Any, name: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise InvalidModelAction(f"{name} must be a finite number")
    return value


def _type_text_contains_control_sequence(text: str, description: str) -> bool:
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        return True
    if _TRAILING_LITERAL_NEWLINE_RE.search(text) is None:
        return False
    command_text = text[:-2].strip()
    return bool(
        command_text
        and (
            _TERMINAL_ENTER_INTENT_RE.search(description)
            or _TERMINAL_COMMAND_SHAPE_RE.search(command_text)
        )
    )


def _coordinate(
    value: Any,
    *,
    coordinate_type: str,
    original_width: int,
    original_height: int,
    processed_width: int | None,
    processed_height: int | None,
) -> tuple[int, int]:
    if original_width <= 0 or original_height <= 0:
        raise InvalidModelAction("original screenshot dimensions must be positive")
    if not isinstance(value, list) or len(value) != 2:
        raise InvalidModelAction("coordinate must be a two-item array")
    x = _number(value[0], "coordinate[0]")
    y = _number(value[1], "coordinate[1]")
    if coordinate_type == "relative":
        if not 0 <= x <= 999 or not 0 <= y <= 999:
            raise InvalidModelAction("relative coordinates must be in [0, 999]")
        return (
            int(x * (original_width - 1) / 999),
            int(y * (original_height - 1) / 999),
        )
    if coordinate_type != "absolute":
        raise InvalidModelAction("coordinate_type must be relative or absolute")
    if (
        processed_width is None
        or processed_height is None
        or processed_width <= 0
        or processed_height <= 0
    ):
        raise InvalidModelAction("absolute coordinates require positive processed dimensions")
    if not 0 <= x < processed_width or not 0 <= y < processed_height:
        raise InvalidModelAction("absolute coordinate is outside the processed image")
    return (
        int(x * original_width / processed_width),
        int(y * original_height / processed_height),
    )


def parse_qwen3vl_response(
    response: str,
    *,
    original_width: int,
    original_height: int,
    processed_width: int | None = None,
    processed_height: int | None = None,
    coordinate_type: str = "relative",
) -> ParsedQwen3VLAction:
    if not isinstance(response, str):
        raise InvalidModelAction("response must be text")
    match = _RESPONSE_RE.fullmatch(response)
    if match is None:
        raise InvalidModelAction(
            "response must contain exactly one Action line and one closed tool_call"
        )
    try:
        call = json.loads(
            match.group("call"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                InvalidModelAction(f"invalid JSON number: {value}")
            ),
        )
    except InvalidModelAction:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise InvalidModelAction("tool_call must contain one strict JSON object") from exc
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        raise InvalidModelAction("tool_call requires exactly name and arguments")
    if call["name"] != "computer_use":
        raise InvalidModelAction("tool name must be computer_use")
    arguments = call["arguments"]
    if not isinstance(arguments, dict):
        raise InvalidModelAction("arguments must be an object")
    action = arguments.get("action")
    if not isinstance(action, str) or action not in _ACTION_ARGUMENTS:
        raise InvalidModelAction("action is missing or unknown")
    expected = _ACTION_ARGUMENTS[action]
    missing = expected - arguments.keys()
    unknown = arguments.keys() - expected
    if missing:
        raise InvalidModelAction(f"missing arguments: {', '.join(sorted(missing))}")
    if unknown:
        raise InvalidModelAction(f"unknown arguments: {', '.join(sorted(unknown))}")

    mapped: tuple[int, int] | None = None
    if action in _COORDINATE_ACTIONS:
        mapped = _coordinate(
            arguments["coordinate"],
            coordinate_type=coordinate_type,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
        )
        method = {
            "mouse_move": "moveTo",
            "left_click": "click",
            "left_click_drag": "dragTo",
            "right_click": "rightClick",
            "middle_click": "middleClick",
            "double_click": "doubleClick",
        }[action]
        command = f"pyautogui.{method}({mapped[0]}, {mapped[1]})"
    elif action == "key":
        keys = arguments["keys"]
        if (
            not isinstance(keys, list)
            or not keys
            or any(not isinstance(key, str) or not key for key in keys)
        ):
            raise InvalidModelAction("keys must be a non-empty array of non-empty strings")
        serialized = ", ".join(repr(key) for key in keys)
        command = (
            f"pyautogui.hotkey({serialized})"
            if len(keys) > 1
            else f"pyautogui.press({serialized})"
        )
    elif action == "type":
        text = arguments["text"]
        if not isinstance(text, str) or not text:
            raise InvalidModelAction("text must be a non-empty string")
        if _type_text_contains_control_sequence(
            text,
            match.group("description"),
        ):
            raise InvalidModelAction(
                "type text must not contain keyboard control sequences; "
                "split text entry from key actions such as key ['enter'] or a hotkey"
            )
        command = f"pyautogui.write({text!r}, interval=0.03)"
    elif action == "scroll":
        pixels = _number(arguments["pixels"], "pixels")
        if pixels == 0:
            raise InvalidModelAction("pixels must be non-zero")
        command = f"pyautogui.scroll({pixels!r})"
    elif action == "wait":
        seconds = _number(arguments["time"], "time")
        if seconds <= 0:
            raise InvalidModelAction("time must be positive")
        command = f"pyautogui.sleep({seconds!r})"
    else:
        status = arguments["status"]
        if status not in {"success", "failure"}:
            raise InvalidModelAction("status must be success or failure")
        command = "DONE" if status == "success" else "FAIL"
    return ParsedQwen3VLAction(
        description=match.group("description").strip(),
        raw_arguments=dict(arguments),
        action=action,
        command=command,
        mapped_coordinate=mapped,
    )


def validate_qwen3vl_history(messages: Any) -> int:
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Qwen3-VL request requires system and current screenshot messages")
    assistant_count = 0
    image_count = 0
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Qwen3-VL messages must be objects")
        if message.get("role") == "assistant":
            assistant_count += 1
        content = message.get("content")
        if isinstance(content, list):
            image_count += sum(
                part.get("type") == "image_url"
                for part in content
                if isinstance(part, dict)
            )
    final = messages[-1]
    final_content = final.get("content") if isinstance(final, dict) else None
    if (
        final.get("role") != "user"
        or not isinstance(final_content, list)
        or not any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in final_content
        )
    ):
        raise ValueError("Qwen3-VL request must end with the current screenshot")
    if image_count != assistant_count + 1 or not 1 <= image_count <= 5:
        raise ValueError("Qwen3-VL request must contain four prior rounds plus current at most")
    return image_count


def build_strict_qwen3vl_agent(
    upstream_class: type[Any],
    *,
    base_url: str,
    api_key: str,
    max_corrections_per_observation: int = MAX_CORRECTIONS_PER_OBSERVATION,
    max_invalid_actions_per_task: int = MAX_INVALID_ACTIONS_PER_TASK,
    **kwargs: Any,
) -> Any:
    if max_corrections_per_observation < 0 or max_invalid_actions_per_task < 1:
        raise ValueError("invalid action retry bounds must be non-negative and bounded")

    class StrictQwen3VLAgent(upstream_class):
        def __init__(self) -> None:
            super().__init__(**kwargs)
            self.last_invalid_action_retryable = False
            self.last_validation_error: str | None = None
            self.last_parsed_action: ParsedQwen3VLAction | None = None
            self.last_openai_response_boundary: OpenAIResponseBoundary | None = None
            self._invalid_response = ""
            self._invalid_total = 0
            self._invalid_for_observation = 0
            self._observation_sha256: str | None = None

        def reset(self, _logger: Any = None, **reset_kwargs: Any) -> None:
            del reset_kwargs
            super().reset(_logger)
            self.last_invalid_action_retryable = False
            self.last_validation_error = None
            self.last_parsed_action = None
            self.last_openai_response_boundary = None
            self._invalid_response = ""
            self._invalid_total = 0
            self._invalid_for_observation = 0
            self._observation_sha256 = None

        def _call_llm_openai(self, messages: Any, model: str) -> str:
            import openai

            validate_qwen3vl_history(messages)
            client = openai.OpenAI(base_url=base_url, api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=False,
            )
            boundary = normalize_openai_qwen_response(response.choices[0].message)
            if not boundary.canonical_content:
                raise RuntimeError("Qwen endpoint returned no response content")
            self.last_openai_response_boundary = boundary
            return boundary.canonical_content

        def parse_response(
            self,
            response: str,
            original_width: int | None = None,
            original_height: int | None = None,
            processed_width: int | None = None,
            processed_height: int | None = None,
        ) -> tuple[str, list[str]]:
            if not original_width or not original_height:
                raise InvalidModelAction("original screenshot dimensions are required")
            try:
                parsed = parse_qwen3vl_response(
                    response,
                    original_width=original_width,
                    original_height=original_height,
                    processed_width=processed_width,
                    processed_height=processed_height,
                    coordinate_type=self.coordinate_type,
                )
            except InvalidModelAction:
                self._invalid_response = response
                raise
            self.last_parsed_action = parsed
            return parsed.description, [parsed.command]

        def predict(self, instruction: str, obs: Mapping[str, Any]) -> tuple[str, list[str]]:
            screenshot = bytes(obs["screenshot"])
            observation_sha256 = hashlib.sha256(screenshot).hexdigest()
            if observation_sha256 != self._observation_sha256:
                self._observation_sha256 = observation_sha256
                self._invalid_for_observation = 0
            corrected_instruction = instruction
            if self._invalid_for_observation:
                corrected_instruction = (
                    instruction
                    + "\n\nYour previous response was invalid: "
                    + (self.last_validation_error or "invalid tool call")
                    + ". Return exactly one corrected Action line and one "
                    "<tool_call> JSON object for this same screenshot."
                )
            lengths = {
                name: len(getattr(self, name))
                for name in (
                    "thoughts",
                    "actions",
                    "observations",
                    "responses",
                    "screenshots",
                )
            }
            try:
                response, actions = super().predict(corrected_instruction, obs)
            except InvalidModelAction as exc:
                for name, length in lengths.items():
                    del getattr(self, name)[length:]
                self._invalid_total += 1
                self._invalid_for_observation += 1
                self.last_validation_error = str(exc)
                self.last_parsed_action = None
                self.last_invalid_action_retryable = (
                    self._invalid_for_observation
                    <= max_corrections_per_observation
                    and self._invalid_total < max_invalid_actions_per_task
                )
                return self._invalid_response, []
            self.last_invalid_action_retryable = False
            self.last_validation_error = None
            self._invalid_for_observation = 0
            return response, actions

    return StrictQwen3VLAgent()
