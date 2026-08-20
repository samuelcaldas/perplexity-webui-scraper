"""End-to-end multi-turn integration test with injected custom tool execution, monitoring, and synthesis."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from perplexity_webui_scraper.api.app import app
from perplexity_webui_scraper.api.tool_calling import TOOL_CALL_SENTINEL_END, TOOL_CALL_SENTINEL_START
from tests._injected_tool_e2e_harness import (
    CONTROL_RUNBOOK_NAME,
    CONTROL_RUNBOOK_TOOL_ANTHROPIC,
    CONTROL_RUNBOOK_TOOL_CHAT,
    CONTROL_RUNBOOK_TOOL_RESPONSES,
    DIAGNOSTIC_TOOL_RAW_OUTPUT,
    EXPECTED_DIAGNOSTIC_ARGS,
    EXPECTED_SYNTHESIZED_OUTPUT,
    INJECTED_DIAGNOSTIC_NAME,
    INJECTED_DIAGNOSTIC_TOOL_ANTHROPIC,
    INJECTED_DIAGNOSTIC_TOOL_CHAT,
    INJECTED_DIAGNOSTIC_TOOL_RESPONSES,
    ScriptedConversation,
    ToolCallMonitor,
)


TOKEN = "test-e2e-token-abc123"
AUTH_HEADER = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _make_tool_sentinel(name: str, arguments: dict) -> str:
    serialized = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f'{TOOL_CALL_SENTINEL_START}{{"arguments":{serialized},"name":"{name}"}}{TOOL_CALL_SENTINEL_END}'


# ---------------------------------------------------------------------------
# Deterministic 3-Turn Multi-Turn Integration Tests (OpenAI Chat Completions)
# ---------------------------------------------------------------------------


def test_custom_injected_tool_chat_completions_multi_turn(client: TestClient) -> None:
    """Test 3-turn injected custom tool lifecycle on /v1/chat/completions."""
    monitor = ToolCallMonitor()
    tools = [INJECTED_DIAGNOSTIC_TOOL_CHAT, CONTROL_RUNBOOK_TOOL_CHAT]

    turn1_answer = json.dumps(
        {"available_tools": sorted([INJECTED_DIAGNOSTIC_NAME, CONTROL_RUNBOOK_NAME])},
        separators=(",", ":"),
    )
    turn2_answer = _make_tool_sentinel(INJECTED_DIAGNOSTIC_NAME, EXPECTED_DIAGNOSTIC_ARGS)
    turn3_answer = json.dumps(EXPECTED_SYNTHESIZED_OUTPUT, separators=(",", ":"))

    conv = ScriptedConversation([turn1_answer, turn2_answer, turn3_answer])
    provider = patch("perplexity_webui_scraper.api.routes.completions._client_pool.get_or_create").start()
    provider.return_value.create_conversation.return_value = conv

    try:
        # --- Turn 1: Capability Inquiry ---
        messages: list[dict] = [
            {"role": "system", "content": "You are an automated agent with tool access."},
            {
                "role": "user",
                "content": (
                    "Which tools are available in this session? "
                    'Return exact raw JSON: {"available_tools": ["..."]}. Do not call any tool.'
                ),
            },
        ]
        resp1 = client.post(
            "/v1/chat/completions",
            headers=AUTH_HEADER,
            json={"model": "perplexity/best", "messages": messages, "tools": tools},
        )
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["choices"][0]["finish_reason"] == "stop"
        content1 = json.loads(data1["choices"][0]["message"]["content"])
        assert content1 == {"available_tools": [CONTROL_RUNBOOK_NAME, INJECTED_DIAGNOSTIC_NAME]}
        monitor.assert_call_history(0)

        # Verify prompt structure in query_str
        assert len(conv.recorded_queries) >= 1
        query1 = conv.recorded_queries[0]
        assert "<declared_tools>" in query1
        assert INJECTED_DIAGNOSTIC_NAME in query1
        assert CONTROL_RUNBOOK_NAME in query1
        assert "</harness_reminder>" in query1

        # --- Turn 2: Exact Tool Invocation ---
        messages.append({"role": "assistant", "content": data1["choices"][0]["message"]["content"]})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Call {INJECTED_DIAGNOSTIC_NAME} exactly once with cluster_id 'cluster-test-17', "
                    "checks ['dns', 'storage'], and include_raw false. Emit only the tool invocation."
                ),
            }
        )
        resp2 = client.post(
            "/v1/chat/completions",
            headers=AUTH_HEADER,
            json={
                "model": "perplexity/best",
                "messages": messages,
                "tools": tools,
                "tool_choice": {"type": "function", "function": {"name": INJECTED_DIAGNOSTIC_NAME}},
            },
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        choice2 = data2["choices"][0]
        assert choice2["finish_reason"] == "tool_calls"
        emitted_calls = choice2["message"]["tool_calls"]
        assert len(emitted_calls) == 1

        call_item = emitted_calls[0]
        record = monitor.record_call(
            endpoint="chat_completions",
            stage="turn2",
            call_id=call_item["id"],
            name=call_item["function"]["name"],
            arguments=call_item["function"]["arguments"],
        )
        monitor.assert_call_history(
            1,
            expected_endpoint="chat_completions",
            expected_name=INJECTED_DIAGNOSTIC_NAME,
            expected_args=EXPECTED_DIAGNOSTIC_ARGS,
        )

        # Execute the tool via monitor
        tool_output = monitor.execute_call(record)
        assert tool_output == DIAGNOSTIC_TOOL_RAW_OUTPUT

        # --- Turn 3: Feed Back Result & Synthesis ---
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": record.call_id,
                        "type": "function",
                        "function": {
                            "name": record.name,
                            "arguments": json.dumps(record.arguments),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": record.call_id,
                "content": json.dumps(tool_output),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Do not call another tool. Using only the diagnostic tool result, return one raw JSON object "
                    "with exactly cluster_id, overall_status, findings, recommendation. "
                    "Map overall to lowercase. For each check preserve order and "
                    "map ok=true to 'pass' and ok=false to 'fail'."
                ),
            }
        )
        resp3 = client.post(
            "/v1/chat/completions",
            headers=AUTH_HEADER,
            json={"model": "perplexity/best", "messages": messages},
        )
        assert resp3.status_code == 200
        data3 = resp3.json()
        assert data3["choices"][0]["finish_reason"] == "stop"
        content3 = json.loads(data3["choices"][0]["message"]["content"])
        assert content3 == EXPECTED_SYNTHESIZED_OUTPUT
        monitor.assert_call_history(1)

    finally:
        patch.stopall()


# ---------------------------------------------------------------------------
# Deterministic 3-Turn Multi-Turn Integration Tests (Anthropic Messages API)
# ---------------------------------------------------------------------------


def test_custom_injected_tool_anthropic_messages_multi_turn(client: TestClient) -> None:
    """Test 3-turn injected custom tool lifecycle on /v1/messages."""
    monitor = ToolCallMonitor()
    tools = [INJECTED_DIAGNOSTIC_TOOL_ANTHROPIC, CONTROL_RUNBOOK_TOOL_ANTHROPIC]

    turn1_answer = json.dumps(
        {"available_tools": sorted([INJECTED_DIAGNOSTIC_NAME, CONTROL_RUNBOOK_NAME])},
        separators=(",", ":"),
    )
    turn2_answer = _make_tool_sentinel(INJECTED_DIAGNOSTIC_NAME, EXPECTED_DIAGNOSTIC_ARGS)
    turn3_answer = json.dumps(EXPECTED_SYNTHESIZED_OUTPUT, separators=(",", ":"))

    conv = ScriptedConversation([turn1_answer, turn2_answer, turn3_answer])
    provider = patch("perplexity_webui_scraper.api.routes.messages._client_pool.get_or_create").start()
    provider.return_value.create_conversation.return_value = conv

    try:
        # --- Turn 1: Capability Inquiry ---
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    "Which tools are available in this session? "
                    'Return exact raw JSON: {"available_tools": ["..."]}. Do not call any tool.'
                ),
            }
        ]
        resp1 = client.post(
            "/v1/messages",
            headers={"x-api-key": TOKEN},
            json={
                "model": "anthropic/claude-sonnet-5",
                "messages": messages,
                "tools": tools,
                "system": "You are an automated agent with tool access.",
            },
        )
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["stop_reason"] == "end_turn"
        assert len(data1["content"]) == 1
        content1 = json.loads(data1["content"][0]["text"])
        assert content1 == {"available_tools": [CONTROL_RUNBOOK_NAME, INJECTED_DIAGNOSTIC_NAME]}
        monitor.assert_call_history(0)

        # --- Turn 2: Exact Tool Invocation ---
        messages.append({"role": "assistant", "content": data1["content"][0]["text"]})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Call {INJECTED_DIAGNOSTIC_NAME} exactly once with cluster_id 'cluster-test-17', "
                    "checks ['dns', 'storage'], and include_raw false. Emit only the tool invocation."
                ),
            }
        )
        resp2 = client.post(
            "/v1/messages",
            headers={"x-api-key": TOKEN},
            json={
                "model": "anthropic/claude-sonnet-5",
                "messages": messages,
                "tools": tools,
                "tool_choice": {"type": "tool", "name": INJECTED_DIAGNOSTIC_NAME},
            },
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["stop_reason"] == "tool_use"
        assert len(data2["content"]) == 1
        tool_use_block = data2["content"][0]
        assert tool_use_block["type"] == "tool_use"

        record = monitor.record_call(
            endpoint="anthropic_messages",
            stage="turn2",
            call_id=tool_use_block["id"],
            name=tool_use_block["name"],
            arguments=tool_use_block["input"],
        )
        monitor.assert_call_history(
            1,
            expected_endpoint="anthropic_messages",
            expected_name=INJECTED_DIAGNOSTIC_NAME,
            expected_args=EXPECTED_DIAGNOSTIC_ARGS,
        )

        tool_output = monitor.execute_call(record)
        assert tool_output == DIAGNOSTIC_TOOL_RAW_OUTPUT

        # --- Turn 3: Feed Back Result & Synthesis ---
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": record.call_id,
                        "name": record.name,
                        "input": record.arguments,
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": record.call_id,
                        "content": json.dumps(tool_output),
                    },
                    {
                        "type": "text",
                        "text": (
                            "Do not call another tool. Using only the diagnostic tool result, return one raw JSON "
                            "object with exactly cluster_id, overall_status, findings, recommendation. "
                            "Map overall to lowercase. For each check preserve order and "
                            "map ok=true to 'pass' and ok=false to 'fail'."
                        ),
                    },
                ],
            }
        )
        resp3 = client.post(
            "/v1/messages",
            headers={"x-api-key": TOKEN},
            json={"model": "anthropic/claude-sonnet-5", "messages": messages},
        )
        assert resp3.status_code == 200
        data3 = resp3.json()
        assert data3["stop_reason"] == "end_turn"
        content3 = json.loads(data3["content"][0]["text"])
        assert content3 == EXPECTED_SYNTHESIZED_OUTPUT
        monitor.assert_call_history(1)

    finally:
        patch.stopall()


# ---------------------------------------------------------------------------
# Deterministic 3-Turn Multi-Turn Integration Tests (OpenAI Responses API)
# ---------------------------------------------------------------------------


def test_custom_injected_tool_responses_api_multi_turn(client: TestClient) -> None:
    """Test 3-turn injected custom tool lifecycle on /v1/responses."""
    monitor = ToolCallMonitor()
    tools = [INJECTED_DIAGNOSTIC_TOOL_RESPONSES, CONTROL_RUNBOOK_TOOL_RESPONSES]

    turn1_answer = json.dumps(
        {"available_tools": sorted([INJECTED_DIAGNOSTIC_NAME, CONTROL_RUNBOOK_NAME])},
        separators=(",", ":"),
    )
    turn2_answer = _make_tool_sentinel(INJECTED_DIAGNOSTIC_NAME, EXPECTED_DIAGNOSTIC_ARGS)
    turn3_answer = json.dumps(EXPECTED_SYNTHESIZED_OUTPUT, separators=(",", ":"))

    conv = ScriptedConversation([turn1_answer, turn2_answer, turn3_answer])
    provider = patch("perplexity_webui_scraper.api.routes.responses._client_pool.get_or_create").start()
    provider.return_value.create_conversation.return_value = conv

    try:
        # --- Turn 1: Capability Inquiry ---
        input_items: list[dict] = [
            {
                "type": "message",
                "role": "user",
                "content": (
                    "Which tools are available in this session? "
                    'Return exact raw JSON: {"available_tools": ["..."]}. Do not call any tool.'
                ),
            }
        ]
        resp1 = client.post(
            "/v1/responses",
            headers=AUTH_HEADER,
            json={
                "model": "perplexity/best",
                "input": input_items,
                "tools": tools,
                "instructions": "You are an automated agent with tool access.",
            },
        )
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["status"] == "completed"
        assert len(data1["output"]) == 1
        assert data1["output"][0]["type"] == "message"
        content1 = json.loads(data1["output"][0]["content"][0]["text"])
        assert content1 == {"available_tools": [CONTROL_RUNBOOK_NAME, INJECTED_DIAGNOSTIC_NAME]}
        monitor.assert_call_history(0)

        # --- Turn 2: Exact Tool Invocation ---
        input_items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": data1["output"][0]["content"][0]["text"],
            }
        )
        input_items.append(
            {
                "type": "message",
                "role": "user",
                "content": (
                    f"Call {INJECTED_DIAGNOSTIC_NAME} exactly once with cluster_id 'cluster-test-17', "
                    "checks ['dns', 'storage'], and include_raw false. Emit only the tool invocation."
                ),
            }
        )
        resp2 = client.post(
            "/v1/responses",
            headers=AUTH_HEADER,
            json={
                "model": "perplexity/best",
                "input": input_items,
                "tools": tools,
                "tool_choice": {"type": "function", "name": INJECTED_DIAGNOSTIC_NAME},
            },
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["status"] == "completed"
        assert len(data2["output"]) == 1
        call_output_item = data2["output"][0]
        assert call_output_item["type"] == "function_call"

        record = monitor.record_call(
            endpoint="responses",
            stage="turn2",
            call_id=call_output_item["call_id"],
            name=call_output_item["name"],
            arguments=call_output_item["arguments"],
        )
        monitor.assert_call_history(
            1,
            expected_endpoint="responses",
            expected_name=INJECTED_DIAGNOSTIC_NAME,
            expected_args=EXPECTED_DIAGNOSTIC_ARGS,
        )

        tool_output = monitor.execute_call(record)
        assert tool_output == DIAGNOSTIC_TOOL_RAW_OUTPUT

        # --- Turn 3: Feed Back Result & Synthesis ---
        input_items.append(
            {
                "type": "function_call",
                "call_id": record.call_id,
                "name": record.name,
                "arguments": json.dumps(record.arguments),
            }
        )
        input_items.append(
            {
                "type": "function_call_output",
                "call_id": record.call_id,
                "output": json.dumps(tool_output),
            }
        )
        input_items.append(
            {
                "type": "message",
                "role": "user",
                "content": (
                    "Do not call another tool. Using only the diagnostic tool result, return one raw JSON object "
                    "with exactly cluster_id, overall_status, findings, recommendation. "
                    "Map overall to lowercase. For each check preserve order and "
                    "map ok=true to 'pass' and ok=false to 'fail'."
                ),
            }
        )
        resp3 = client.post(
            "/v1/responses",
            headers=AUTH_HEADER,
            json={"model": "perplexity/best", "input": input_items},
        )
        assert resp3.status_code == 200
        data3 = resp3.json()
        assert data3["status"] == "completed"
        assert len(data3["output"]) == 1
        assert data3["output"][0]["type"] == "message"
        content3 = json.loads(data3["output"][0]["content"][0]["text"])
        assert content3 == EXPECTED_SYNTHESIZED_OUTPUT
        monitor.assert_call_history(1)

    finally:
        patch.stopall()


# ---------------------------------------------------------------------------
# ToolCallMonitor Harness Unit Tests
# ---------------------------------------------------------------------------


def test_tool_call_monitor_rejects_unauthorized_tool() -> None:
    """Ensure monitor throws if an unexpected or control tool is executed."""
    monitor = ToolCallMonitor()
    record = monitor.record_call(
        endpoint="chat",
        stage="test",
        call_id="call_unauthorized",
        name=CONTROL_RUNBOOK_NAME,
        arguments={"alert_code": "ERR_01"},
    )
    with pytest.raises(ValueError, match="Unauthorized tool execution attempt"):
        monitor.execute_call(record)


def test_tool_call_monitor_rejects_mismatched_arguments() -> None:
    """Ensure monitor throws if injected tool receives invalid arguments."""
    monitor = ToolCallMonitor()
    record = monitor.record_call(
        endpoint="chat",
        stage="test",
        call_id="call_bad_args",
        name=INJECTED_DIAGNOSTIC_NAME,
        arguments={"cluster_id": "wrong-cluster", "checks": ["dns"]},
    )
    with pytest.raises(ValueError, match="Invalid tool arguments"):
        monitor.execute_call(record)


# ---------------------------------------------------------------------------
# Opt-in Live Model E2E Test
# ---------------------------------------------------------------------------


@pytest.mark.model_e2e
def test_model_uses_injected_custom_tool_end_to_end(client: TestClient) -> None:
    """Live verification against real Perplexity session when RUN_MODEL_E2E=1."""
    if os.environ.get("RUN_MODEL_E2E") != "1":
        pytest.skip("Set RUN_MODEL_E2E=1 and PERPLEXITY_SESSION_TOKEN to run live model test.")

    live_token = os.environ.get("PERPLEXITY_SESSION_TOKEN")
    if not live_token:
        pytest.fail("PERPLEXITY_SESSION_TOKEN environment variable is required when RUN_MODEL_E2E=1.")

    model = os.environ.get("PERPLEXITY_MODEL_E2E_MODEL", "perplexity/best")
    headers = {"Authorization": f"Bearer {live_token}"}
    monitor = ToolCallMonitor()
    tools = [INJECTED_DIAGNOSTIC_TOOL_CHAT, CONTROL_RUNBOOK_TOOL_CHAT]

    messages: list[dict] = [
        {"role": "system", "content": "You are an automated agent with tool execution access."},
        {
            "role": "user",
            "content": (
                "Which tools are available in this session? "
                'Return exact raw JSON: {"available_tools": ["..."]}. Do not call any tool.'
            ),
        },
    ]

    resp1 = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": model, "messages": messages, "tools": tools},
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    content1_raw = data1["choices"][0]["message"]["content"]
    content1 = json.loads(content1_raw)
    assert INJECTED_DIAGNOSTIC_NAME in content1.get("available_tools", [])

    messages.append({"role": "assistant", "content": content1_raw})
    messages.append(
        {
            "role": "user",
            "content": (
                f"Call {INJECTED_DIAGNOSTIC_NAME} with cluster_id 'cluster-test-17', "
                "checks ['dns', 'storage'], and include_raw false. Emit only the tool invocation."
            ),
        }
    )

    resp2 = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": INJECTED_DIAGNOSTIC_NAME}},
        },
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    emitted_calls = data2["choices"][0]["message"]["tool_calls"]
    assert len(emitted_calls) == 1

    record = monitor.record_call(
        endpoint="live_chat",
        stage="turn2",
        call_id=emitted_calls[0]["id"],
        name=emitted_calls[0]["function"]["name"],
        arguments=emitted_calls[0]["function"]["arguments"],
    )
    tool_output = monitor.execute_call(record)

    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": record.call_id,
                    "type": "function",
                    "function": {
                        "name": record.name,
                        "arguments": json.dumps(record.arguments),
                    },
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": record.call_id,
            "content": json.dumps(tool_output),
        }
    )
    messages.append(
        {
            "role": "user",
            "content": (
                "Do not call another tool. Using only the diagnostic tool result, return one raw JSON object "
                "with exactly cluster_id, overall_status, findings, recommendation. "
                "Map overall to lowercase. For each check preserve order and "
                "map ok=true to 'pass' and ok=false to 'fail'."
            ),
        }
    )

    resp3 = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": model, "messages": messages},
    )
    assert resp3.status_code == 200
    data3 = resp3.json()
    content3 = json.loads(data3["choices"][0]["message"]["content"])
    assert content3 == EXPECTED_SYNTHESIZED_OUTPUT
