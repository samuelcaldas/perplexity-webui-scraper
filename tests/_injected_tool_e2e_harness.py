"""Shared harness, schemas, and ToolCallMonitor for injected custom tool integration tests."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

from perplexity_webui_scraper.core import Conversation


# Injected custom tool schema constants
INJECTED_DIAGNOSTIC_NAME = "run_cluster_diagnostic"
CONTROL_RUNBOOK_NAME = "lookup_test_runbook"

EXPECTED_DIAGNOSTIC_ARGS: dict[str, Any] = {
    "cluster_id": "cluster-test-17",
    "checks": ["dns", "storage"],
    "include_raw": False,
}

DIAGNOSTIC_TOOL_RAW_OUTPUT: dict[str, Any] = {
    "cluster": "cluster-test-17",
    "overall": "DEGRADED",
    "checks": [
        {"name": "dns", "ok": True, "message": "resolver latency 12ms"},
        {"name": "storage", "ok": False, "message": "node-7 volume is 91% full"},
    ],
    "next_action": "drain node-7",
}

EXPECTED_SYNTHESIZED_OUTPUT: dict[str, Any] = {
    "cluster_id": "cluster-test-17",
    "overall_status": "degraded",
    "findings": [
        {"check": "dns", "status": "pass", "detail": "resolver latency 12ms"},
        {"check": "storage", "status": "fail", "detail": "node-7 volume is 91% full"},
    ],
    "recommendation": "drain node-7",
}

# OpenAI Chat Completions tool definitions
INJECTED_DIAGNOSTIC_TOOL_CHAT: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": INJECTED_DIAGNOSTIC_NAME,
        "description": "Run deterministic diagnostics against the isolated test cluster.",
        "parameters": {
            "type": "object",
            "properties": {
                "cluster_id": {"type": "string", "enum": ["cluster-test-17"]},
                "checks": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["dns", "storage"]},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "include_raw": {"type": "boolean", "enum": [False]},
            },
            "required": ["cluster_id", "checks", "include_raw"],
        },
    },
}

CONTROL_RUNBOOK_TOOL_CHAT: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": CONTROL_RUNBOOK_NAME,
        "description": "Look up troubleshooting runbooks for cluster alerts.",
        "parameters": {
            "type": "object",
            "properties": {
                "alert_code": {"type": "string"},
            },
            "required": ["alert_code"],
        },
    },
}

# Anthropic Messages tool definitions
INJECTED_DIAGNOSTIC_TOOL_ANTHROPIC: dict[str, Any] = {
    "name": INJECTED_DIAGNOSTIC_NAME,
    "description": "Run deterministic diagnostics against the isolated test cluster.",
    "input_schema": {
        "type": "object",
        "properties": {
            "cluster_id": {"type": "string", "enum": ["cluster-test-17"]},
            "checks": {
                "type": "array",
                "items": {"type": "string", "enum": ["dns", "storage"]},
                "minItems": 2,
                "maxItems": 2,
            },
            "include_raw": {"type": "boolean", "enum": [False]},
        },
        "required": ["cluster_id", "checks", "include_raw"],
    },
}

CONTROL_RUNBOOK_TOOL_ANTHROPIC: dict[str, Any] = {
    "name": CONTROL_RUNBOOK_NAME,
    "description": "Look up troubleshooting runbooks for cluster alerts.",
    "input_schema": {
        "type": "object",
        "properties": {
            "alert_code": {"type": "string"},
        },
        "required": ["alert_code"],
    },
}

# OpenAI Responses API tool definitions (flat format)
INJECTED_DIAGNOSTIC_TOOL_RESPONSES: dict[str, Any] = {
    "type": "function",
    "name": INJECTED_DIAGNOSTIC_NAME,
    "description": "Run deterministic diagnostics against the isolated test cluster.",
    "parameters": {
        "type": "object",
        "properties": {
            "cluster_id": {"type": "string", "enum": ["cluster-test-17"]},
            "checks": {
                "type": "array",
                "items": {"type": "string", "enum": ["dns", "storage"]},
                "minItems": 2,
                "maxItems": 2,
            },
            "include_raw": {"type": "boolean", "enum": [False]},
        },
        "required": ["cluster_id", "checks", "include_raw"],
    },
}

CONTROL_RUNBOOK_TOOL_RESPONSES: dict[str, Any] = {
    "type": "function",
    "name": CONTROL_RUNBOOK_NAME,
    "description": "Look up troubleshooting runbooks for cluster alerts.",
    "parameters": {
        "type": "object",
        "properties": {
            "alert_code": {"type": "string"},
        },
        "required": ["alert_code"],
    },
}


@dataclass(frozen=True)
class RecordedToolCall:
    """Record of a tool call emitted by a model or provider."""

    endpoint: str
    stage: str
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallMonitor:
    """Monitors, records, and executes custom tool invocations during test runs."""

    calls: list[RecordedToolCall] = field(default_factory=list)

    def record_call(
        self,
        endpoint: str,
        stage: str,
        call_id: str,
        name: str,
        arguments: dict[str, Any] | str,
    ) -> RecordedToolCall:
        """Record an emitted tool call before running assertions."""
        parsed_args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        record = RecordedToolCall(
            endpoint=endpoint,
            stage=stage,
            call_id=call_id,
            name=name,
            arguments=parsed_args,
        )
        self.calls.append(record)
        return record

    def execute_call(self, record: RecordedToolCall) -> dict[str, Any]:
        """Execute only the declared custom diagnostic tool, rejecting unauthorized tools."""
        if record.name != INJECTED_DIAGNOSTIC_NAME:
            raise ValueError(f"Unauthorized tool execution attempt: '{record.name}' is not executable.")
        if record.arguments != EXPECTED_DIAGNOSTIC_ARGS:
            raise ValueError(
                f"Invalid tool arguments for '{INJECTED_DIAGNOSTIC_NAME}': "
                f"expected {EXPECTED_DIAGNOSTIC_ARGS}, got {record.arguments}"
            )
        return DIAGNOSTIC_TOOL_RAW_OUTPUT

    def assert_call_history(
        self,
        expected_count: int,
        expected_endpoint: str | None = None,
        expected_name: str | None = None,
        expected_args: dict[str, Any] | None = None,
    ) -> None:
        """Assert the state of recorded tool calls."""
        assert len(self.calls) == expected_count, (
            f"Expected {expected_count} recorded calls, but found {len(self.calls)}: {self.calls}"
        )
        if expected_count > 0 and expected_name is not None:
            last_call = self.calls[-1]
            if expected_endpoint is not None:
                assert last_call.endpoint == expected_endpoint
            assert last_call.name == expected_name
            if expected_args is not None:
                assert last_call.arguments == expected_args


class ScriptedConversation(MagicMock):
    """Mock conversation returning scripted sequence of responses across turns."""

    def __init__(self, scripted_answers: list[str], uuid: str | None = None, **kwargs: Any) -> None:
        super().__init__(spec=Conversation, **kwargs)
        self.uuid = uuid or str(uuid4())
        self._scripted_answers = list(scripted_answers)
        self.recorded_queries: list[str] = []
        self._current_index = 0
        self.answer = self._scripted_answers[0] if self._scripted_answers else ""

    def ask(self, query: str, files: Any = None, stream: bool = False, **kwargs: Any) -> None:
        """Record query and progress to next scripted response."""
        self.recorded_queries.append(query)
        if self._current_index < len(self._scripted_answers):
            self.answer = self._scripted_answers[self._current_index]
            self._current_index += 1
        else:
            self.answer = ""


class ScriptedProvider(MagicMock):
    """Mock client pool provider creating ScriptedConversations."""

    def __init__(self, scripted_answers: list[str]) -> None:
        super().__init__()
        self._scripted_answers = scripted_answers
        self.conversations: list[ScriptedConversation] = []
        self.create_conversation.side_effect = self._create_conv

    def _create_conv(self, config: Any = None) -> ScriptedConversation:
        conv = ScriptedConversation(self._scripted_answers)
        self.conversations.append(conv)
        return conv
