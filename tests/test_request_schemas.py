from __future__ import annotations

from base64 import b64encode
import math
from typing import Any

from pydantic import ValidationError
import pytest

import perplexity_webui_scraper.api.schemas.request as request_schemas
from perplexity_webui_scraper.api.schemas.request import ChatCompletionRequest


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


def _request(messages: list[dict[str, Any]], **fields: object) -> dict[str, object]:
    return {"model": "perplexity/best", "messages": messages, **fields}


def test_model_validate_accepts_function_tool_request() -> None:
    request = ChatCompletionRequest.model_validate(
        _request(
            [{"role": "user", "content": "What is weather like?"}],
            tools=[FUNCTION_TOOL],
            tool_choice="auto",
            parallel_tool_calls=True,
            unrelated_client_field="preserved",
        )
    )

    assert request.tools is not None
    assert request.tools[0].function.name == "get_weather"
    assert request.tool_choice == "auto"
    assert request.parallel_tool_calls is True
    assert request.model_extra is not None
    assert request.model_extra["unrelated_client_field"] == "preserved"


def test_model_validate_accepts_assistant_null_content_with_tool_calls() -> None:
    request = ChatCompletionRequest.model_validate(
        _request(
            [
                {"role": "user", "content": "What is weather like?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"location":"Boston"}'},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
                {"role": "tool", "tool_call_id": "call_2", "content": "{}"},
            ]
        )
    )

    assert request.messages[1].content is None
    assert request.messages[1].tool_calls is not None
    assert len(request.messages[1].tool_calls) == 2
    assert request.messages[1].tool_calls[0].function.arguments == '{"location":"Boston"}'


def test_model_validate_accepts_valid_tool_result() -> None:
    request = ChatCompletionRequest.model_validate(
        _request(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"location":"Boston"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"temperature": 22}'},
            ]
        )
    )

    assert request.messages[1].tool_call_id == "call_1"
    assert request.messages[1].content == '{"temperature": 22}'


@pytest.mark.parametrize(
    ("tool", "field"),
    [
        ({"type": "function", "function": {"name": "bad name", "parameters": {}}}, "name"),
        ({"type": "function", "function": {"name": "get_weather", "parameters": {"type": "string"}}}, "schema"),
    ],
)
def test_model_validate_rejects_malformed_function_tool(tool: dict[str, object], field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        ChatCompletionRequest.model_validate(_request([{"role": "user", "content": "Hi"}], tools=[tool]))


def test_model_validate_rejects_malformed_function_arguments() -> None:
    with pytest.raises(ValidationError, match="arguments"):
        ChatCompletionRequest.model_validate(
            _request(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "not-json"},
                            }
                        ],
                    }
                ]
            )
        )


def test_model_validate_rejects_unresolvable_function_schema_reference() -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"$ref": "#/definitions/missing"},
        },
    }

    with pytest.raises(ValidationError, match="reference"):
        ChatCompletionRequest.model_validate(_request([{"role": "user", "content": "Hi"}], tools=[tool]))


def test_model_validate_rejects_tool_message_without_tool_call_id() -> None:
    with pytest.raises(ValidationError, match="tool_call_id"):
        ChatCompletionRequest.model_validate(_request([{"role": "tool", "content": '{"temperature": 22}'}]))


def test_model_validate_rejects_assistant_null_content_without_tool_calls() -> None:
    with pytest.raises(ValidationError, match="content"):
        ChatCompletionRequest.model_validate(_request([{"role": "assistant", "content": None}]))


def test_model_validate_rejects_invalid_tool_choice() -> None:
    with pytest.raises(ValidationError, match="tool_choice"):
        ChatCompletionRequest.model_validate(
            _request(
                [{"role": "user", "content": "Hi"}],
                tools=[FUNCTION_TOOL],
                tool_choice={"type": "function", "function": {"name": "bad name"}},
            )
        )


def test_model_validate_defaults_zero_argument_function_parameters() -> None:
    request = ChatCompletionRequest.model_validate(
        _request(
            [{"role": "user", "content": "Hi"}],
            tools=[{"type": "function", "function": {"name": "ping"}}],
        )
    )

    assert request.tools is not None
    assert request.tools[0].function.parameters == {}


def test_model_validate_preserves_supported_openai_message_fields() -> None:
    request = ChatCompletionRequest.model_validate(
        _request(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "name": "assistant_name",
                    "refusal": "cannot comply",
                    "function_call": {"name": "get_weather", "arguments": "{}"},
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
            ]
        )
    )

    message = request.messages[0]
    assert message.name == "assistant_name"
    assert message.refusal == "cannot comply"
    assert message.function_call is not None
    assert message.function_call.name == "get_weather"


def test_model_validate_accepts_null_assistant_content_with_function_call() -> None:
    request = ChatCompletionRequest.model_validate(
        _request(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": "get_weather", "arguments": "{}"},
                }
            ]
        )
    )

    assert request.messages[0].content is None
    assert request.messages[0].function_call is not None


def test_model_validate_rejects_unsupported_message_fields() -> None:
    with pytest.raises(ValidationError, match="unsupported_message_field"):
        ChatCompletionRequest.model_validate(
            _request([{"role": "user", "content": "Hi", "unsupported_message_field": True}])
        )


@pytest.mark.parametrize(
    "image_url",
    [
        {},
        {"url": ""},
        {"url": "https://example.com/image.png"},
        {"url": "data:application/pdf;base64,SGVsbG8="},
        {"url": "data:image/png;base64,not-base64!"},
    ],
)
def test_model_validate_rejects_invalid_image_data_uri(image_url: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="image"):
        ChatCompletionRequest.model_validate(
            _request(
                [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": image_url}],
                    }
                ]
            )
        )


def test_image_bytes_preserves_valid_data_image_behavior() -> None:
    encoded_image = b64encode(b"image-bytes").decode()
    request = ChatCompletionRequest.model_validate(
        _request(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe image"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
                    ],
                }
            ]
        )
    )

    assert request.messages[0].text() == "Describe image"
    assert request.messages[0].image_bytes() == [(b"image-bytes", "image.png", "image/png")]


def test_model_validate_rejects_oversized_data_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(request_schemas, "MAX_FILE_SIZE", 10)
    oversized_data = b64encode(b"01234567890").decode()

    with pytest.raises(ValidationError, match="size"):
        ChatCompletionRequest.model_validate(
            _request(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{oversized_data}"},
                            }
                        ],
                    }
                ]
            )
        )


@pytest.mark.parametrize(
    "coordinates",
    [{"latitude": math.nan, "longitude": 0}, {"latitude": 91, "longitude": 0}],
)
def test_model_validate_rejects_invalid_coordinates(coordinates: dict[str, float]) -> None:
    with pytest.raises(ValidationError, match="latitude"):
        ChatCompletionRequest.model_validate(
            _request([{"role": "user", "content": "Hi"}], perplexity={"coordinates": coordinates})
        )


@pytest.mark.parametrize("field", ["space_uuid", "thread_uuid"])
def test_model_validate_rejects_invalid_uuid(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        ChatCompletionRequest.model_validate(
            _request([{"role": "user", "content": "Hi"}], perplexity={field: "not-a-uuid"})
        )


def test_model_validate_rejects_blank_model_and_empty_messages() -> None:
    with pytest.raises(ValidationError, match="model"):
        ChatCompletionRequest.model_validate({"model": "   ", "messages": [{"role": "user", "content": "Hi"}]})

    with pytest.raises(ValidationError, match="messages"):
        ChatCompletionRequest.model_validate({"model": "perplexity/best", "messages": []})


def test_model_validate_rejects_named_tool_choice_not_declared() -> None:
    with pytest.raises(ValidationError, match="tool_choice"):
        ChatCompletionRequest.model_validate(
            _request(
                [{"role": "user", "content": "Hi"}],
                tools=[FUNCTION_TOOL],
                tool_choice={"type": "function", "function": {"name": "other"}},
            )
        )


@pytest.mark.parametrize(
    "messages",
    [
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "{}"},
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        ],
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        ],
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "user", "content": "stale"},
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        ],
        [
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        ],
    ],
)
def test_model_validate_rejects_invalid_tool_result_sequence(messages: list[dict[str, object]]) -> None:
    with pytest.raises(ValidationError, match="tool"):
        ChatCompletionRequest.model_validate(_request(messages))


def test_model_validate_rejects_duplicate_tool_result_ids() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
    ]

    with pytest.raises(ValidationError, match="tool"):
        ChatCompletionRequest.model_validate(_request(messages))


def test_model_validate_accepts_streaming_tool_requests() -> None:
    req = ChatCompletionRequest.model_validate(
        _request(
            [{"role": "user", "content": "Call weather."}],
            stream=True,
            tools=[FUNCTION_TOOL],
        )
    )
    assert req.stream is True
    assert req.tools is not None


@pytest.mark.parametrize("tool_choice", ["required", {"type": "function", "function": {"name": "get_weather"}}])
def test_model_validate_rejects_mandatory_tool_choice_without_tools(tool_choice: object) -> None:
    with pytest.raises(ValidationError, match="tools"):
        ChatCompletionRequest.model_validate(
            _request([{"role": "user", "content": "Call weather."}], tool_choice=tool_choice)
        )


def _assistant_call_messages(
    name: str = "get_weather", arguments: str = '{"location":"Boston"}'
) -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
    ]


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("unknown", '{"location":"Boston"}', "declared"),
        ("get_weather", '{"location":3}', "schema"),
        ("get_weather", "{}", "required"),
    ],
)
def test_model_validate_rejects_assistant_calls_outside_declared_schema(
    name: str,
    arguments: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ChatCompletionRequest.model_validate(_request(_assistant_call_messages(name, arguments), tools=[FUNCTION_TOOL]))


def test_model_validate_accepts_tools_omitted_historical_replay() -> None:
    request = ChatCompletionRequest.model_validate(
        _request(_assistant_call_messages("historical_tool", '{"anything":true}'))
    )

    assert request.messages[0].tool_calls is not None
    assert request.messages[0].tool_calls[0].function.name == "historical_tool"


@pytest.mark.parametrize(
    "messages",
    [
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            }
        ],
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "user", "content": "interleaved"},
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        ],
    ],
)
def test_model_validate_rejects_orphan_or_skipped_tool_call_groups(messages: list[dict[str, object]]) -> None:
    with pytest.raises(ValidationError, match="tool"):
        ChatCompletionRequest.model_validate(_request(messages))
