from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from openai import BadRequestError, OpenAI
from pytest import fixture, mark, raises

from perplexity_webui_scraper.api.routes.completions import _client_pool, _conversation_cache
from perplexity_webui_scraper.core import Conversation


if TYPE_CHECKING:
    from collections.abc import Iterator


MODEL_ID = "perplexity/best"
TOKEN = "sentinel"
THREAD_UUID = "12345678-1234-5678-1234-567812345678"
TOOL_START = "<|OPENAI_TOOL_CALL|>"
TOOL_END = "<|END_OPENAI_TOOL_CALL|>"
FUNCTION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


@fixture(autouse=True)
def _clean_api_caches() -> Iterator[None]:
    """Clear API caches before and after each SDK contract test."""
    _conversation_cache._store.clear()
    _client_pool._clients.clear()
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


def _tool_answer(name: str = "get_weather", arguments: str = '{"location":"Boston"}') -> str:
    return f'{TOOL_START}{{"name":"{name}","arguments":{arguments}}}{TOOL_END}'


def test_sdk_parses_authenticated_model_list_without_provider_network(
    openai_client: OpenAI,
) -> None:
    provider = MagicMock()
    provider.get_account_profile.return_value = SimpleNamespace(account_tier="free")

    with patch(
        "perplexity_webui_scraper.api.auth.client_pool.get_or_create",
        return_value=provider,
    ) as get_or_create:
        models = openai_client.models.list()

    assert models.object == "list"
    assert models.data
    assert "perplexity/best" in {model.id for model in models.data}
    assert all(model.object == "model" for model in models.data)
    assert all(model.owned_by for model in models.data)
    get_or_create.assert_called_once_with(TOKEN)
    provider.get_account_profile.assert_called_once_with()
    provider.create_conversation.assert_not_called()


def test_sdk_parses_ordinary_completion_without_provider_network(openai_client: OpenAI) -> None:
    conversation = _make_conversation("ordinary provider answer")
    provider = _make_provider(conversation)

    with patch(
        "perplexity_webui_scraper.api.routes.completions._client_pool.get_or_create",
        return_value=provider,
    ) as get_or_create:
        completion = openai_client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Hello"}],
        )

    choice = completion.choices[0]
    assert completion.object == "chat.completion"
    assert completion.model == MODEL_ID
    assert choice.message.role == "assistant"
    assert choice.message.content == "ordinary provider answer"
    assert choice.finish_reason == "stop"
    get_or_create.assert_called_once_with(TOKEN)
    provider.create_conversation.assert_called_once()
    conversation.ask.assert_called_once()


def test_sdk_parses_emulated_function_tool_call(openai_client: OpenAI) -> None:
    conversation = _make_conversation(_tool_answer())
    provider = _make_provider(conversation)

    with patch(
        "perplexity_webui_scraper.api.routes.completions._client_pool.get_or_create",
        return_value=provider,
    ):
        completion = openai_client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "What is the weather?"}],
            tools=[FUNCTION_TOOL],
            tool_choice="auto",
        )

    choice = completion.choices[0]
    tool_call = choice.message.tool_calls[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content is None
    assert tool_call.type == "function"
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == '{"location":"Boston"}'
    provider.create_conversation.assert_called_once()
    conversation.ask.assert_called_once()


@mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "get_weather"}},
    ],
)
def test_sdk_raises_typed_bad_request_for_failed_required_tool_choice(
    openai_client: OpenAI,
    tool_choice: str | dict[str, object],
) -> None:
    conversation = _make_conversation("ordinary provider answer")
    provider = _make_provider(conversation)

    with (
        patch(
            "perplexity_webui_scraper.api.routes.completions._client_pool.get_or_create",
            return_value=provider,
        ),
        raises(BadRequestError) as raised,
    ):
        openai_client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Call weather."}],
            tools=[FUNCTION_TOOL],
            tool_choice=tool_choice,
        )

    error = raised.value
    assert isinstance(error, BadRequestError)
    assert error.status_code == 400
    assert error.body is not None
    assert error.body["message"] == "Provider response did not contain a valid required tool call."
    assert error.response.status_code == 400
    provider.create_conversation.assert_called_once()
    conversation.ask.assert_called_once()


def test_sdk_parses_ordinary_stream_chunks_and_final_finish(openai_client: OpenAI) -> None:
    conversation = _make_conversation("")
    conversation.__iter__ = MagicMock(
        return_value=iter(
            [
                SimpleNamespace(last_chunk="Hello", answer=None),
                SimpleNamespace(last_chunk="Hello world", answer=None),
            ]
        )
    )
    provider = _make_provider(conversation)

    with patch(
        "perplexity_webui_scraper.api.routes.completions._client_pool.get_or_create",
        return_value=provider,
    ):
        stream = openai_client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Say hello."}],
            stream=True,
        )
        chunks = list(stream)

    content_chunks = [chunk.choices[0].delta.content for chunk in chunks if chunk.choices[0].delta.content]
    assert chunks[0].choices[0].delta.role == "assistant"
    assert content_chunks == ["Hello", " world"]
    assert chunks[-1].choices[0].delta.content is None
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert all(chunk.object == "chat.completion.chunk" for chunk in chunks)
    assert all(chunk.id == chunks[0].id for chunk in chunks)
    provider.create_conversation.assert_called_once()
    conversation.ask.assert_called_once()
