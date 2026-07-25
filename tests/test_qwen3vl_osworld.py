from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from intelligent_liars.osworld_overlay import (
    MODEL_ID,
    MODEL_REVISION,
    OSWORLD_COMMIT,
)
from intelligent_liars.qwen3vl_osworld import (
    InvalidModelAction,
    build_strict_qwen3vl_agent,
    normalize_openai_qwen_response,
    parse_qwen3vl_response,
    validate_qwen3vl_history,
)
from intelligent_liars.run_control import stable_sha256


ROOT = Path(__file__).parents[1]
TEMPLATES = ROOT / "docs/evaluation/osworld_templates"


def response(arguments: dict, *, name: str = "computer_use") -> str:
    call = json.dumps({"name": name, "arguments": arguments})
    return f"<think>inspect</think>\nAction: Perform action\n<tool_call>{call}</tool_call>"


def parse(arguments: dict):
    return parse_qwen3vl_response(
        response(arguments),
        original_width=1920,
        original_height=1080,
        processed_width=1280,
        processed_height=704,
    )


@pytest.mark.parametrize(
    "raw",
    [
        (
            "Action: click\n<function=computer_use>"
            "<parameter=action>left_click</parameter></function>"
        ),
        "Action: click\n<tool_call>{",
        "Action: click\n<tool_call>{\"name\":\"computer_use\",\"arguments\":{\"action\":\"left_click\",\"coordinate\":[1,2]}}",
        "Action: click",
        (
            "Action: click\n<tool_call>"
            '{"name":"computer_use","name":"computer_use","arguments":{"action":"wait","time":1}}'
            "</tool_call>"
        ),
    ],
)
def test_malformed_historical_xml_and_ambiguous_json_fail_closed(raw):
    with pytest.raises(InvalidModelAction):
        parse_qwen3vl_response(
            raw,
            original_width=1920,
            original_height=1080,
        )


def test_correct_qwen3_json_calls_retain_coordinates_keys_and_text():
    click = parse({"action": "left_click", "coordinate": [0, 999]})
    assert click.raw_arguments == {
        "action": "left_click",
        "coordinate": [0, 999],
    }
    assert click.mapped_coordinate == (0, 1079)
    assert click.command == "pyautogui.click(0, 1079)"

    interior = parse({"action": "mouse_move", "coordinate": [500, 500]})
    assert interior.mapped_coordinate == (960, 540)
    bottom_right = parse({"action": "mouse_move", "coordinate": [999, 999]})
    assert bottom_right.mapped_coordinate == (1919, 1079)

    keys = parse({"action": "key", "keys": ["ctrl", "shift", "s"]})
    assert keys.raw_arguments["keys"] == ["ctrl", "shift", "s"]
    assert keys.command == "pyautogui.hotkey('ctrl', 'shift', 's')"

    typed = parse({"action": "type", "text": r"quoted 'line' C:\temp"})
    assert typed.raw_arguments["text"] == r"quoted 'line' C:\temp"
    assert typed.command == (
        """pyautogui.write("quoted 'line' C:\\\\temp", interval=0.03)"""
    )


@pytest.mark.parametrize(
    "text",
    [
        "code ~/Desktop/project\n",
        "first\rsecond",
        "first\tsecond",
    ],
)
def test_type_rejects_decoded_keyboard_control_sequences(text):
    with pytest.raises(
        InvalidModelAction,
        match="split text entry from key actions",
    ):
        parse({"action": "type", "text": text})


@pytest.mark.parametrize(
    "text",
    [
        r"C:\temp",
        r"Use the two literal characters \n",
        r"column\tvalue",
        r'print("\\n"); raw = r"\x0a"; unicode = r"\u000a"',
    ],
)
def test_type_preserves_ordinary_literal_backslash_sequences(text):
    typed = parse({"action": "type", "text": text})
    assert typed.raw_arguments["text"] == text
    assert typed.command == f"pyautogui.write({text!r}, interval=0.03)"


@pytest.mark.parametrize(
    "text",
    [
        r"code ~/Desktop/project\n",
        r"code ~/Desktop/project\r",
    ],
)
def test_type_rejects_trailing_terminal_enter_escape(text):
    with pytest.raises(
        InvalidModelAction,
        match="split text entry from key actions",
    ):
        parse({"action": "type", "text": text})


@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "left_click"},
        {"action": "left_click", "coordinate": [1000, 1]},
        {"action": "left_click", "coordinate": [-1, 1]},
        {"action": "left_click", "coordinate": ["1", 2]},
        {"action": "left_click", "coordinate": [float("nan"), 2]},
        {"action": "left_click", "coordinate": [float("inf"), 2]},
        {"action": "key", "keys": []},
        {"action": "type", "text": ""},
        {"action": "scroll", "pixels": 0},
        {"action": "wait"},
        {"action": "terminate"},
        {"action": "terminate", "status": "unknown"},
        {"action": "terminate", "status": "success", "coordinate": [1, 2]},
        {"action": "unknown"},
    ],
)
def test_missing_malformed_unknown_arguments_and_invalid_termination_are_rejected(
    arguments,
):
    with pytest.raises(InvalidModelAction):
        parse(arguments)


def test_valid_success_and_failure_termination_are_explicit():
    assert parse({"action": "terminate", "status": "success"}).command == "DONE"
    assert parse({"action": "terminate", "status": "failure"}).command == "FAIL"


@pytest.mark.parametrize(
    ("content", "reasoning", "normalization"),
    [
        (
            "</think>\nAction: Click\n"
            '<tool_call>{"name":"computer_use","arguments":'
            '{"action":"left_click","coordinate":[18,59]}}</tool_call>',
            "inspect the desktop",
            "hf_reasoning_content_with_orphan_close",
        ),
        (
            "inspect the desktop\n</think>\nAction: Click\n"
            '<tool_call>{"name":"computer_use","arguments":'
            '{"action":"left_click","coordinate":[18,59]}}</tool_call>',
            "inspect the desktop",
            "hf_duplicated_reasoning_with_orphan_close",
        ),
    ],
)
def test_known_hf_reasoning_shapes_are_canonicalized(
    content, reasoning, normalization
):
    boundary = normalize_openai_qwen_response(
        {"content": content, "reasoning_content": reasoning}
    )

    assert boundary.raw_content == content
    assert boundary.canonical_content == (
        "<think>inspect the desktop</think>\nAction: Click\n"
        '<tool_call>{"name":"computer_use","arguments":'
        '{"action":"left_click","coordinate":[18,59]}}</tool_call>'
    )
    assert boundary.normalization == normalization
    parsed = parse_qwen3vl_response(
        boundary.canonical_content,
        original_width=1920,
        original_height=1080,
    )
    assert parsed.action == "left_click"


def test_serialized_openai_extra_field_is_used_for_known_hf_shape():
    content = (
        "</think>\nAction: Click\n"
        '<tool_call>{"name":"computer_use","arguments":'
        '{"action":"left_click","coordinate":[18,59]}}</tool_call>'
    )

    class SerializedMessage:
        def __init__(self):
            self.content = content
            self.model_extra = {"reasoning_content": "inspect the desktop"}

        def model_dump(self):
            return {
                "content": self.content,
                "reasoning_content": self.model_extra["reasoning_content"],
            }

    boundary = normalize_openai_qwen_response(SerializedMessage())
    assert boundary.canonical_content.startswith(
        "<think>inspect the desktop</think>\nAction: Click"
    )
    assert boundary.normalization == "hf_reasoning_content_with_orphan_close"


def test_observed_multi_tool_call_shape_remains_rejected_after_normalization():
    content = (
        "inspect\n</think>\nAction: Drag\n"
        '<tool_call>{"name":"computer_use","arguments":'
        '{"action":"mouse_move","coordinate":[224,500]}}</tool_call>\n'
        '<tool_call>{"name":"computer_use","arguments":'
        '{"action":"left_click_drag","coordinate":[485,502]}}</tool_call>'
    )
    boundary = normalize_openai_qwen_response(
        {"content": content, "reasoning_content": "inspect"}
    )

    with pytest.raises(
        InvalidModelAction,
        match="tool_call must contain one strict JSON object",
    ):
        parse_qwen3vl_response(
            boundary.canonical_content,
            original_width=1920,
            original_height=1080,
        )


@pytest.mark.parametrize(
    ("content", "reasoning"),
    [
        (
            "</think>\nAction: Click\n"
            '<tool_call>{"name":"computer_use","arguments":'
            '{"action":"left_click","coordinate":[18,59]}}</tool_call>',
            None,
        ),
        (
            "different reasoning</think>\nAction: Click\n"
            '<tool_call>{"name":"computer_use","arguments":'
            '{"action":"left_click","coordinate":[18,59]}}</tool_call>',
            "inspect the desktop",
        ),
        (
            "</think>\nnot an action",
            "inspect the desktop",
        ),
    ],
)
def test_absent_or_mismatched_reasoning_does_not_normalize(content, reasoning):
    boundary = normalize_openai_qwen_response(
        {"content": content, "reasoning_content": reasoning}
    )
    assert boundary.canonical_content == content
    assert boundary.normalization == "unchanged"


def test_valid_envelope_is_unchanged_for_strict_parser_compatibility():
    content = response({"action": "key", "keys": ["ctrl", "a"]})
    boundary = normalize_openai_qwen_response(
        {"content": content, "reasoning_content": "inspect"}
    )
    assert boundary.raw_content == boundary.canonical_content == content
    assert boundary.normalization == "unchanged"
    assert parse_qwen3vl_response(
        boundary.canonical_content,
        original_width=1920,
        original_height=1080,
    ).command == "pyautogui.hotkey('ctrl', 'a')"


@pytest.mark.parametrize(
    ("width", "height"),
    [(0, 1080), (-1, 1080), (1920, 0), (1920, -1)],
)
def test_coordinate_mapping_rejects_nonpositive_original_dimensions(width, height):
    with pytest.raises(InvalidModelAction, match="dimensions must be positive"):
        parse_qwen3vl_response(
            response({"action": "left_click", "coordinate": [0, 0]}),
            original_width=width,
            original_height=height,
        )


@pytest.mark.parametrize(
    ("processed_width", "processed_height"),
    [(0, 704), (-1, 704), (1280, 0), (1280, -1)],
)
def test_absolute_mapping_rejects_nonpositive_processed_dimensions(
    processed_width, processed_height
):
    with pytest.raises(InvalidModelAction, match="positive processed dimensions"):
        parse_qwen3vl_response(
            response({"action": "left_click", "coordinate": [0, 0]}),
            original_width=1920,
            original_height=1080,
            processed_width=processed_width,
            processed_height=processed_height,
            coordinate_type="absolute",
        )


class FakeUpstreamQwen3VLAgent:
    queued_responses: list[str] = []
    instructions: list[str] = []

    def __init__(self, **kwargs):
        self.coordinate_type = kwargs["coordinate_type"]
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.responses = []
        self.screenshots = []

    def reset(self, logger=None):
        del logger
        self.actions.clear()
        self.responses.clear()
        self.screenshots.clear()

    def predict(self, instruction, obs):
        self.instructions.append(instruction)
        raw = self.queued_responses.pop(0)
        self.screenshots.append(bytes(obs["screenshot"]))
        self.responses.append(raw)
        description, actions = self.parse_response(
            raw,
            original_width=1920,
            original_height=1080,
            processed_width=1280,
            processed_height=704,
        )
        self.actions.append(description)
        return raw, actions


def make_agent(outputs: list[str]):
    FakeUpstreamQwen3VLAgent.queued_responses = list(outputs)
    FakeUpstreamQwen3VLAgent.instructions = []
    return build_strict_qwen3vl_agent(
        FakeUpstreamQwen3VLAgent,
        base_url="https://endpoint.invalid/v1",
        api_key="test",
        model=MODEL_ID,
        coordinate_type="relative",
    )


def test_invalid_action_requests_correction_on_same_observation_without_history_pollution():
    valid = response({"action": "key", "keys": ["ctrl", "c"]})
    agent = make_agent(["historical malformed XML", valid])
    observation = {"screenshot": b"same screenshot"}

    raw, actions = agent.predict("task", observation)
    assert raw == "historical malformed XML"
    assert actions == []
    assert agent.last_invalid_action_retryable is True
    assert agent.responses == []
    assert agent.screenshots == []

    _, actions = agent.predict("task", observation)
    assert actions == ["pyautogui.hotkey('ctrl', 'c')"]
    assert "same screenshot" in FakeUpstreamQwen3VLAgent.instructions[-1]
    assert len(agent.responses) == len(agent.screenshots) == len(agent.actions) == 1


def test_malformed_terminal_type_requests_split_text_and_enter_correction():
    malformed = response(
        {"action": "type", "text": r"code ~/Desktop/project\n"}
    )
    corrected = response(
        {"action": "type", "text": "code ~/Desktop/project"}
    )
    agent = make_agent([malformed, corrected])
    observation = {"screenshot": b"same screenshot"}

    _, actions = agent.predict("open the project", observation)
    assert actions == []
    assert agent.last_invalid_action_retryable is True
    assert "split text entry from key actions" in agent.last_validation_error
    assert agent.responses == agent.actions == agent.screenshots == []

    _, actions = agent.predict("open the project", observation)
    assert actions == [
        "pyautogui.write('code ~/Desktop/project', interval=0.03)"
    ]
    assert "key ['enter']" in FakeUpstreamQwen3VLAgent.instructions[-1]
    assert len(agent.responses) == len(agent.screenshots) == len(agent.actions) == 1


def test_retry_exhaustion_has_no_fallback_terminal_action():
    agent = make_agent(["bad one", "bad two", "bad three"])
    observation = {"screenshot": b"same screenshot"}

    results = [agent.predict("task", observation) for _ in range(3)]

    assert [actions for _, actions in results] == [[], [], []]
    assert agent.last_invalid_action_retryable is False
    assert agent.last_parsed_action is None
    assert agent.responses == agent.actions == agent.screenshots == []


def test_openai_boundary_returns_canonical_history_and_retains_raw_evidence(
    monkeypatch,
):
    raw = (
        "inspect</think>\nAction: Copy\n"
        '<tool_call>{"name":"computer_use","arguments":'
        '{"action":"key","keys":["ctrl","c"]}}</tool_call>'
    )

    class Message:
        content = raw
        reasoning_content = "inspect"

    class Completions:
        @staticmethod
        def create(**kwargs):
            del kwargs
            return type(
                "Response",
                (),
                {"choices": [type("Choice", (), {"message": Message()})()]},
            )()

    class Client:
        def __init__(self, **kwargs):
            del kwargs
            self.chat = type("Chat", (), {"completions": Completions()})()

    class OpenAIHistoryAgent(FakeUpstreamQwen3VLAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.max_tokens = 32768
            self.temperature = 0.0
            self.top_p = 0.9

        def predict(self, instruction, obs):
            del instruction
            canonical = self._call_llm_openai(_messages(0), MODEL_ID)
            self.responses.append(canonical)
            self.screenshots.append(bytes(obs["screenshot"]))
            description, actions = self.parse_response(
                canonical,
                original_width=1920,
                original_height=1080,
                processed_width=1280,
                processed_height=704,
            )
            self.actions.append(description)
            return canonical, actions

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    agent = build_strict_qwen3vl_agent(
        OpenAIHistoryAgent,
        base_url="https://endpoint.invalid/v1",
        api_key="test",
        model=MODEL_ID,
        coordinate_type="relative",
    )
    canonical, actions = agent.predict("copy", {"screenshot": b"same screenshot"})

    assert canonical.startswith("<think>inspect</think>\nAction: Copy")
    assert actions == ["pyautogui.hotkey('ctrl', 'c')"]
    assert agent.responses == [canonical]
    assert agent.last_openai_response_boundary.raw_content == raw
    assert agent.last_openai_response_boundary.canonical_content == canonical


def _messages(rounds: int) -> list[dict]:
    messages = [{"role": "system", "content": [{"type": "text", "text": "tools"}]}]
    for index in range(rounds):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{index}"},
                        }
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "action"}]},
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,current"},
                }
            ],
        }
    )
    return messages


def test_history_is_four_prior_rounds_plus_current_and_never_six_images():
    assert validate_qwen3vl_history(_messages(4)) == 5
    with pytest.raises(ValueError, match="four prior"):
        validate_qwen3vl_history(_messages(5))


@pytest.mark.parametrize(
    ("name", "count", "grid_hash"),
    [
        (
            "qwen3vl_gate_20_100.template.json",
            20,
            "7530c7ed617da624a072f161a3d35d8be42b473543a9effaf6791281448d6c86",
        ),
        (
            "qwen3vl_small_39_100.template.json",
            39,
            None,
        ),
    ],
)
def test_validation_fixtures_pin_source_revision_and_rca_gates(
    name, count, grid_hash
):
    payload = json.loads((TEMPLATES / name).read_text())
    assert payload["osworld"]["commit"] == OSWORLD_COMMIT
    assert payload["model"]["id"] == MODEL_ID
    assert payload["model"]["revision"] == MODEL_REVISION
    assert payload["model"]["processor_revision"] == MODEL_REVISION
    assert payload["model"]["tokenizer_revision"] == MODEL_REVISION
    assert payload["run"]["agent_class"] == (
        "mm_agents.qwen3vl_agent.Qwen3VLAgent@osworld-pinned"
    )
    assert payload["run"]["agent_source_sha256"] == (
        "c9bb34d3ad822168c66133cd97c607d4645b7eff08072e1e45c49dbbad8491b4"
    )
    parser_path = ROOT / "src/intelligent_liars/qwen3vl_osworld.py"
    assert payload["run"]["action_parser_source_sha256"] == hashlib.sha256(
        parser_path.read_bytes()
    ).hexdigest()
    assert payload["run"]["screenshot_transform"] == {
        "encoding": "png-base64",
        "implementation": "mm_agents.qwen3vl_agent.process_image@osworld-pinned",
        "max_pixels": "16*16*4*12800",
        "resize_factor": 32,
        "source_sha256": (
            "c9bb34d3ad822168c66133cd97c607d4645b7eff08072e1e45c49dbbad8491b4"
        ),
    }
    assert payload["run"]["step_cap"] == 100
    assert payload["run"]["history_n"] == 4
    assert payload["run"]["image_max"] == 5
    assert len(payload["task_grid"]["task_ids"]) == count
    assert payload["task_grid"]["sha256"] == stable_sha256(
        payload["task_grid"]["task_ids"]
    )
    if grid_hash:
        assert payload["task_grid"]["sha256"] == grid_hash
    assert len(payload["evaluation"]["evaluator_sha256"]) == count
    gates = payload["acceptance_gates"]
    assert gates["tool_call_initial_validity_gte"] == 0.99
    assert gates["coordinate_key_text_retention_gte"] == 0.99
    assert gates["proxy"]["mode"] == "enabled"
    assert gates["artifact_checksum_completeness"] == 1.0
    assert gates["implicit_terminal_actions_eq"] == 0
