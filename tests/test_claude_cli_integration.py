from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

from pytest import fixture, mark

from perplexity_webui_scraper.api.routes.completions import _client_pool, _conversation_cache
from perplexity_webui_scraper.core import Conversation


if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai import OpenAI
    from openai.types.chat import ChatCompletionToolParam


TOKEN = "samuel-CpkzV0hgDECYBoeH3"
THREAD_UUID = "87654321-4321-8765-4321-876543218765"
TOOL_START = "<|OPENAI_TOOL_CALL|>"
TOOL_END = "<|END_OPENAI_TOOL_CALL|>"

# Coding tool definitions for Claude CLI
BASH_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "Bash",
        "description": "Execute bash command in terminal.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run"}},
            "required": ["command"],
        },
    },
}

READ_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Read file contents from filesystem.",
        "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string", "description": "Path to file"}},
            "required": ["file_path"],
        },
    },
}

WRITE_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "Write",
        "description": "Write contents to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to file"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["file_path", "content"],
        },
    },
}

CLAUDE_CODING_TOOLS = [BASH_TOOL, READ_TOOL, WRITE_TOOL]


@fixture(autouse=True)
def _clean_state() -> Iterator[None]:
    """Clean cache state and apply Claude settings environment variables."""
    _conversation_cache._store.clear()
    _client_pool._clients.clear()

    env_overrides = {
        "ANTHROPIC_BASE_URL": "http://testserver/v1",
        "ANTHROPIC_AUTH_TOKEN": TOKEN,
        "ANTHROPIC_DEFAULT_FABLE_MODEL": "anthropic/claude-sonnet-5",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "openai/gpt-5.6-sol",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "openai/gpt-5.6-terra",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "perplexity/best",
        "CLAUDE_CODE_SUBAGENT_MODEL": "perplexity/sonar-2",
    }

    with patch.dict(os.environ, env_overrides):
        yield

    _conversation_cache._store.clear()
    _client_pool._clients.clear()


def _make_conversation(answer: str, uuid: str = THREAD_UUID) -> MagicMock:
    conversation = MagicMock(spec=Conversation)
    conversation.uuid = uuid
    conversation.answer = answer
    return conversation


def _make_provider(conversation: MagicMock) -> MagicMock:
    provider = MagicMock()
    provider.create_conversation.return_value = conversation
    return provider


def _tool_call_answer(name: str, arguments: dict[str, Any]) -> str:
    args_str = json.dumps(arguments, separators=(",", ":"))
    return f'{TOOL_START}{{"name":"{name}","arguments":{args_str}}}{TOOL_END}'


def test_claude_cli_env_configuration_is_present() -> None:
    """Verify environment variables simulating Claude settings.json are loaded."""
    assert os.environ["ANTHROPIC_BASE_URL"] == "http://testserver/v1"
    assert os.environ["ANTHROPIC_AUTH_TOKEN"] == TOKEN
    assert os.environ["ANTHROPIC_DEFAULT_FABLE_MODEL"] == "anthropic/claude-sonnet-5"
    assert os.environ["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "openai/gpt-5.6-terra"
    assert os.environ["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "openai/gpt-5.6-sol"


def test_claude_coding_tool_read_file_emulation(openai_client: OpenAI) -> None:
    """Verify Claude requesting a file read triggers an OpenAI function tool call."""
    tool_answer = _tool_call_answer("Read", {"file_path": "/home/user/main.py"})
    conversation = _make_conversation(tool_answer)
    provider = _make_provider(conversation)

    with patch(
        "perplexity_webui_scraper.api.routes.completions._client_pool.get_or_create",
        return_value=provider,
    ):
        model = os.environ["ANTHROPIC_DEFAULT_FABLE_MODEL"]
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Read /home/user/main.py"}],
            tools=CLAUDE_CODING_TOOLS,
            tool_choice="auto",
        )

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    assert len(choice.message.tool_calls) == 1

    tool_call = choice.message.tool_calls[0]
    assert tool_call.function.name == "Read"
    assert '"file_path":"/home/user/main.py"' in tool_call.function.arguments


def test_claude_coding_tool_bash_execution_emulation(openai_client: OpenAI) -> None:
    """Verify Claude requesting a bash command executes Bash function tool call."""
    tool_answer = _tool_call_answer("Bash", {"command": "pytest tests/"})
    conversation = _make_conversation(tool_answer)
    provider = _make_provider(conversation)

    with patch(
        "perplexity_webui_scraper.api.routes.completions._client_pool.get_or_create",
        return_value=provider,
    ):
        model = os.environ["ANTHROPIC_DEFAULT_SONNET_MODEL"]
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Run the tests"}],
            tools=CLAUDE_CODING_TOOLS,
            tool_choice="auto",
        )

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    tool_call = choice.message.tool_calls[0]
    assert tool_call.function.name == "Bash"
    assert '"command":"pytest tests/"' in tool_call.function.arguments


def test_claude_multi_turn_coding_workflow(openai_client: OpenAI) -> None:
    """Verify multi-turn tool result continuation during a coding task."""
    first_tool_answer = _tool_call_answer("Read", {"file_path": "/app/src/utils.py"})
    conversation = _make_conversation(first_tool_answer, uuid=THREAD_UUID)
    provider = _make_provider(conversation)

    model = os.environ["ANTHROPIC_DEFAULT_FABLE_MODEL"]

    with patch(
        "perplexity_webui_scraper.api.routes.completions._client_pool.get_or_create",
        return_value=provider,
    ):
        # Step 1: Initial user prompt
        response1 = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Check utils.py and fix any bug"}],
            tools=CLAUDE_CODING_TOOLS,
        )

    assert response1.choices[0].message.tool_calls is not None
    tool_call_1 = response1.choices[0].message.tool_calls[0]
    call_id = tool_call_1.id

    # Step 2: Feed back tool result
    conversation.answer = "The file has a bug in add(). I fixed it."

    with patch(
        "perplexity_webui_scraper.api.routes.completions._client_pool.get_or_create",
        return_value=provider,
    ):
        response2 = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Check utils.py and fix any bug"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "arguments": '{"file_path":"/app/src/utils.py"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "def add(a, b):\n    return a - b\n",
                },
            ],
            tools=CLAUDE_CODING_TOOLS,
            extra_body={"perplexity": {"thread_uuid": THREAD_UUID}},
        )

    choice2 = response2.choices[0]
    assert choice2.finish_reason == "stop"
    assert "fixed it" in (choice2.message.content or "")


@mark.parametrize(
    "env_var",
    [
        "ANTHROPIC_DEFAULT_FABLE_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    ],
)
def test_claude_settings_models_are_all_resolvable(
    openai_client: OpenAI,
    env_var: str,
) -> None:
    """Verify all models configured in settings.json resolve and return responses."""
    model_id = os.environ[env_var]
    conversation = _make_conversation(f"Response from {model_id}")
    provider = _make_provider(conversation)

    with patch(
        "perplexity_webui_scraper.api.routes.completions._client_pool.get_or_create",
        return_value=provider,
    ):
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Ping"}],
        )

    assert response.choices[0].message.content == f"Response from {model_id}"
    assert response.model == model_id


def test_claude_coding_tool_with_reasoning_effort_high(openai_client: OpenAI) -> None:
    """Verify coding tool calling works seamlessly when reasoning_effort is high."""
    tool_answer = _tool_call_answer("Write", {"file_path": "solution.py", "content": "x = 42\n"})
    conversation = _make_conversation(tool_answer)
    provider = _make_provider(conversation)

    model = os.environ["ANTHROPIC_DEFAULT_FABLE_MODEL"]

    with patch(
        "perplexity_webui_scraper.api.routes.completions._client_pool.get_or_create",
        return_value=provider,
    ):
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Write the solution to solution.py"}],
            tools=CLAUDE_CODING_TOOLS,
            reasoning_effort="high",
        )

    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    tool_call = choice.message.tool_calls[0]
    assert tool_call.function.name == "Write"
    assert '"file_path":"solution.py"' in tool_call.function.arguments
    config = provider.create_conversation.call_args[0][0]
    assert config.reasoning_effort == "high"
