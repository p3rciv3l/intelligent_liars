from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from intelligent_liars.osworld_overlay import (
    MODEL_ID,
    MODEL_REVISION,
    OSWORLD_COMMIT,
)
from intelligent_liars.qwen3vl_osworld import (
    InvalidModelAction,
    build_strict_qwen3vl_agent,
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
    assert click.mapped_coordinate == (0, 1080)
    assert click.command == "pyautogui.click(0, 1080)"

    interior = parse({"action": "mouse_move", "coordinate": [500, 500]})
    assert interior.mapped_coordinate == (960, 540)

    keys = parse({"action": "key", "keys": ["ctrl", "shift", "s"]})
    assert keys.raw_arguments["keys"] == ["ctrl", "shift", "s"]
    assert keys.command == "pyautogui.hotkey('ctrl', 'shift', 's')"

    typed = parse({"action": "type", "text": "quoted 'line'\nsecond"})
    assert typed.raw_arguments["text"] == "quoted 'line'\nsecond"
    assert "pyautogui.press('enter')" in typed.command
    assert repr("quoted 'line'") in typed.command


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


def test_retry_exhaustion_has_no_fallback_terminal_action():
    agent = make_agent(["bad one", "bad two", "bad three"])
    observation = {"screenshot": b"same screenshot"}

    results = [agent.predict("task", observation) for _ in range(3)]

    assert [actions for _, actions in results] == [[], [], []]
    assert agent.last_invalid_action_retryable is False
    assert agent.last_parsed_action is None
    assert agent.responses == agent.actions == agent.screenshots == []


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
