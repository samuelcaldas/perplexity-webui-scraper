"""POST /v1/messages route — modern Anthropic Messages API compatibility."""

from __future__ import annotations

from asyncio import CancelledError, Lock
from functools import partial
import json
from typing import TYPE_CHECKING, Annotated, Any, Literal
from uuid import uuid4

from anyio import to_thread
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from perplexity_webui_scraper._internal.exceptions import AuthenticationError, PerplexityError
from perplexity_webui_scraper.api.auth import client_pool, extract_token
from perplexity_webui_scraper.api.error_handling import error_response_for
from perplexity_webui_scraper.api.helpers import (
    build_conversation_config,
    build_query_and_files,
)
from perplexity_webui_scraper.api.routes.completions import (
    _config_fingerprint,
    _conversation_cache,
    _pending_tool_calls_metadata,
    _validate_model,
)
from perplexity_webui_scraper.api.schemas.request import (
    AssistantToolCall,
    ChatCompletionRequest,
    ChatMessage,
    ContentPartImageUrl,
    ContentPartText,
    ExplicitToolChoice,
    FunctionDefinition,
    FunctionTool,
    PerplexityExtensions,
    ToolCallFunction,
    ToolChoice,
    ToolChoiceFunction,
)
from perplexity_webui_scraper.api.tool_calling import (
    TOOL_CALL_SENTINEL_START,
    parse_emulated_tool_calls,
    requires_tool_call,
    serialize_tool_arguments,
)


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Iterator

    from perplexity_webui_scraper.core.client import Perplexity
    from perplexity_webui_scraper.core.conversation import Conversation
    from perplexity_webui_scraper.core.response import Response


router = APIRouter()
_client_pool = client_pool


class AnthropicTool(BaseModel):
    """Anthropic tool specification."""

    name: str = Field(min_length=1)
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)


class AnthropicMessageRequest(BaseModel):
    """Schema for Anthropic Messages API requests."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[dict[str, Any]] = Field(min_length=1)
    system: str | list[dict[str, Any]] | None = None
    max_tokens: int | None = None
    stream: bool = False
    thinking: dict[str, Any] | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: Any = None
    perplexity: PerplexityExtensions | None = None

    def to_chat_completion_request(self) -> ChatCompletionRequest:
        """Convert Anthropic message request to internal ChatCompletionRequest."""
        chat_messages: list[ChatMessage] = []

        if self.system:
            if isinstance(self.system, str):
                chat_messages.append(ChatMessage(role="system", content=self.system))
            elif isinstance(self.system, list):
                sys_texts = [str(item.get("text", "")) for item in self.system if isinstance(item, dict)]
                if sys_texts:
                    chat_messages.append(ChatMessage(role="system", content="\n\n".join(sys_texts)))

        for msg in self.messages:
            role_raw = str(msg.get("role", "user"))
            content = msg.get("content")

            if isinstance(content, str):
                role: Literal["system", "developer", "user", "assistant", "tool"] = (
                    role_raw if role_raw in {"system", "developer", "user", "assistant", "tool"} else "user"
                )
                chat_messages.append(ChatMessage(role=role, content=content))
            elif isinstance(content, list):
                if role_raw == "assistant":
                    tool_calls: list[AssistantToolCall] = []
                    text_parts: list[str] = []
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        b_type = block.get("type")
                        if b_type == "text":
                            text_parts.append(str(block.get("text", "")))
                        elif b_type == "tool_use":
                            tool_id = str(block.get("id", f"call_{uuid4().hex[:16]}"))
                            tool_name = str(block.get("name", ""))
                            tool_input = block.get("input", {})
                            tool_args = (
                                json.dumps(tool_input, ensure_ascii=False, separators=(",", ":"))
                                if isinstance(tool_input, dict)
                                else str(tool_input)
                            )
                            tool_calls.append(
                                AssistantToolCall(
                                    id=tool_id,
                                    type="function",
                                    function=ToolCallFunction(name=tool_name, arguments=tool_args),
                                )
                            )
                    text_content = "\n\n".join(text_parts) if text_parts else None
                    chat_messages.append(
                        ChatMessage(
                            role="assistant",
                            content=text_content,
                            tool_calls=tool_calls or None,
                        )
                    )
                else:
                    parts: list[ContentPartText | ContentPartImageUrl] = []
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        b_type = block.get("type")
                        if b_type == "tool_result":
                            tool_use_id = str(block.get("tool_use_id", ""))
                            res_content = block.get("content", "")
                            if isinstance(res_content, list):
                                res_text = "\n".join(str(c.get("text", "")) for c in res_content if isinstance(c, dict))
                            elif isinstance(res_content, dict):
                                res_text = json.dumps(res_content, ensure_ascii=False)
                            else:
                                res_text = str(res_content)
                            chat_messages.append(
                                ChatMessage(
                                    role="tool",
                                    tool_call_id=tool_use_id,
                                    content=res_text,
                                )
                            )
                        elif b_type == "text":
                            parts.append(ContentPartText(type="text", text=str(block.get("text", ""))))
                        elif b_type == "image":
                            source = block.get("source", {})
                            if isinstance(source, dict) and source.get("type") == "base64":
                                media_type = source.get("media_type", "image/png")
                                data = source.get("data", "")
                                parts.append(
                                    ContentPartImageUrl(
                                        type="image_url",
                                        image_url={"url": f"data:{media_type};base64,{data}"},
                                    )
                                )
                    if parts:
                        chat_messages.append(ChatMessage(role="user", content=parts))

        function_tools: list[FunctionTool] | None = None
        if self.tools:
            function_tools = [
                FunctionTool(
                    type="function",
                    function=FunctionDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.input_schema,
                    ),
                )
                for tool in self.tools
            ]

        mapped_tool_choice: ToolChoice | None = None
        if isinstance(self.tool_choice, dict):
            tc_type = self.tool_choice.get("type")
            if tc_type == "auto":
                mapped_tool_choice = "auto"
            elif tc_type == "any":
                mapped_tool_choice = "required"
            elif tc_type == "none":
                mapped_tool_choice = "none"
            elif tc_type == "tool" and self.tool_choice.get("name"):
                mapped_tool_choice = ExplicitToolChoice(
                    type="function",
                    function=ToolChoiceFunction(name=str(self.tool_choice["name"])),
                )
        elif isinstance(self.tool_choice, str):
            if self.tool_choice in {"auto", "required", "none"}:
                mapped_tool_choice = self.tool_choice
            elif self.tool_choice == "any":
                mapped_tool_choice = "required"

        wants_thinking = bool(self.thinking and self.thinking.get("type") == "enabled")

        return ChatCompletionRequest(
            model=self.model,
            messages=chat_messages,
            stream=self.stream,
            thinking=wants_thinking or None,
            tools=function_tools,
            tool_choice=mapped_tool_choice,
            perplexity=self.perplexity,
        )


@router.post("/v1/messages", response_model=None)
async def messages_endpoint(
    raw_request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    x_api_key: Annotated[str | None, Header(alias="x-api-key")] = None,
) -> JSONResponse | StreamingResponse:
    """Handle Anthropic Messages API requests."""
    try:
        body = await raw_request.json()
        anthropic_req = AnthropicMessageRequest.model_validate(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    request = anthropic_req.to_chat_completion_request()
    token = extract_token(authorization or (f"Bearer {x_api_key}" if x_api_key else None))
    _validate_model(request)
    request_lock = _client_pool.get_request_lock(token)
    lock_acquired = False
    stream_handoff = False
    client: Perplexity | None = None

    try:
        await request_lock.acquire()
        lock_acquired = True
        client = await to_thread.run_sync(_client_pool.get_or_create, token)

        query, files = build_query_and_files(request)
        config = build_conversation_config(
            request.model,
            request.perplexity,
            reasoning_effort=request.reasoning_effort,
            thinking=request.thinking,
            has_tools=bool(request.tools),
        )
        conversation = await to_thread.run_sync(client.create_conversation, config)

        if request.stream:
            await to_thread.run_sync(partial(conversation.ask, query, files=files or None, stream=True))
            stream_handoff = True
            return StreamingResponse(
                _stream_messages_api(
                    conversation,
                    request,
                    token,
                    client,
                    request_lock,
                    partial(_client_pool.release_request_lock, token, request_lock),
                    _config_fingerprint(request),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        await to_thread.run_sync(partial(conversation.ask, query, files=files or None))
        return _build_messages_payload(request, conversation, token)
    except AuthenticationError:
        if client is not None:
            await to_thread.run_sync(_client_pool.discard, token, client)
        raise
    finally:
        if not stream_handoff:
            if lock_acquired:
                request_lock.release()
            _client_pool.release_request_lock(token, request_lock)


def _build_messages_payload(
    request: ChatCompletionRequest,
    conversation: Conversation,
    token: str,
) -> JSONResponse:
    """Build response matching Anthropic Messages API format."""
    answer = conversation.answer or ""
    msg_id = f"msg_{uuid4().hex}"

    emulated_tool_calls = parse_emulated_tool_calls(answer, request.tools, request.tool_choice)
    if requires_tool_call(request.tools, request.tool_choice) and not emulated_tool_calls:
        raise HTTPException(status_code=400, detail="Provider response did not contain a valid required tool call.")

    content: list[dict[str, Any]] = []
    stop_reason = "end_turn"

    if emulated_tool_calls:
        stop_reason = "tool_use"
        first_sentinel_pos = answer.find(TOOL_CALL_SENTINEL_START)
        if first_sentinel_pos > 0:
            prefix_text = answer[:first_sentinel_pos].strip()
            if prefix_text:
                content.append(
                    {
                        "type": "text",
                        "text": prefix_text,
                    }
                )
        content.extend(
            {
                "type": "tool_use",
                "id": call.call_id,
                "name": call.function_name,
                "input": call.arguments,
            }
            for call in emulated_tool_calls
        )
    else:
        content.append(
            {
                "type": "text",
                "text": answer,
            }
        )

    conv_uuid = conversation.uuid
    if conv_uuid:
        pending_metadata = None
        if emulated_tool_calls:
            pending_metadata = _pending_tool_calls_metadata(emulated_tool_calls)
        _conversation_cache.set(
            token,
            conv_uuid,
            conversation,
            pending_tool_calls=pending_metadata,
            config_fingerprint=_config_fingerprint(request),
        )

    payload = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": request.model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }

    if conv_uuid:
        payload["perplexity"] = {"thread_uuid": conv_uuid}

    return JSONResponse(content=payload)


async def _stream_messages_api(
    conversation: Conversation,
    request: ChatCompletionRequest,
    token: str,
    client: Perplexity,
    request_lock: Lock,
    release_request: Callable[[], None] | None = None,
    config_fingerprint: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield Anthropic Messages API SSE stream events."""
    msg_id = f"msg_{uuid4().hex}"
    last_content = ""
    sentinel_started = False
    emitted_len = 0
    has_tools = bool(request.tools)
    model_id = request.model
    text_block_opened = False
    block_index = 0

    try:
        iterator = iter(conversation)

        # 1. message_start event
        msg_start_event = {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model_id,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(msg_start_event)}\n\n"

        while True:
            has_response, response = await to_thread.run_sync(_next_response, iterator)
            if not has_response or response is None:
                break

            chunks = getattr(response, "chunks", None)
            cumulative_chunks = "".join(chunks) if chunks else ""
            current = response.answer or cumulative_chunks or response.last_chunk or ""
            if not current:
                continue

            last_content = current

            if has_tools:
                if TOOL_CALL_SENTINEL_START in current:
                    sentinel_pos = current.find(TOOL_CALL_SENTINEL_START)
                    if not sentinel_started and sentinel_pos > emitted_len:
                        delta = current[emitted_len:sentinel_pos]
                        emitted_len = sentinel_pos
                        if delta:
                            if not text_block_opened:
                                block_start = {
                                    "type": "content_block_start",
                                    "index": block_index,
                                    "content_block": {"type": "text", "text": ""},
                                }
                                yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                                text_block_opened = True
                            delta_event = {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {"type": "text_delta", "text": delta},
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
                    sentinel_started = True
                elif not sentinel_started:
                    safe_len = len(current)
                    max_check = min(len(TOOL_CALL_SENTINEL_START) - 1, len(current))
                    for i in range(max_check, 0, -1):
                        if TOOL_CALL_SENTINEL_START.startswith(current[-i:]):
                            safe_len = len(current) - i
                            break
                    if safe_len > emitted_len:
                        delta = current[emitted_len:safe_len]
                        emitted_len = safe_len
                        if delta:
                            if not text_block_opened:
                                block_start = {
                                    "type": "content_block_start",
                                    "index": block_index,
                                    "content_block": {"type": "text", "text": ""},
                                }
                                yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                                text_block_opened = True
                            delta_event = {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {"type": "text_delta", "text": delta},
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
            else:
                delta = current[emitted_len:]
                if delta:
                    emitted_len = len(current)
                    if not text_block_opened:
                        block_start = {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "text", "text": ""},
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                        text_block_opened = True
                    delta_event = {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": delta},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

        if has_tools and not sentinel_started and emitted_len < len(last_content):
            remaining_delta = last_content[emitted_len:]
            if remaining_delta:
                if not text_block_opened:
                    block_start = {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                    text_block_opened = True
                delta_event = {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": remaining_delta},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

        # Close open text block if any
        if text_block_opened:
            stop_data = json.dumps({"type": "content_block_stop", "index": block_index})
            yield f"event: content_block_stop\ndata: {stop_data}\n\n"
            block_index += 1

        # Check for emulated tool calls
        emulated_calls = (
            parse_emulated_tool_calls(last_content, request.tools, request.tool_choice) if has_tools else None
        )

        stop_reason = "end_turn"
        if emulated_calls:
            stop_reason = "tool_use"
            for call in emulated_calls:
                call_block_start = {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call.call_id,
                        "name": call.function_name,
                        "input": {},
                    },
                }
                yield f"event: content_block_start\ndata: {json.dumps(call_block_start)}\n\n"

                call_block_delta = {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": serialize_tool_arguments(call.arguments),
                    },
                }
                yield f"event: content_block_delta\ndata: {json.dumps(call_block_delta)}\n\n"

                stop_call = json.dumps({"type": "content_block_stop", "index": block_index})
                yield f"event: content_block_stop\ndata: {stop_call}\n\n"
                block_index += 1
        elif not text_block_opened:
            # Emit empty text block if nothing else was emitted
            block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
            yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

        conv_uuid = conversation.uuid
        if conv_uuid:
            pending_metadata = None
            if emulated_calls:
                pending_metadata = _pending_tool_calls_metadata(emulated_calls)
            _conversation_cache.set(
                token,
                conv_uuid,
                conversation,
                pending_tool_calls=pending_metadata,
                config_fingerprint=config_fingerprint,
            )

        # 4. message_delta event
        msg_delta: dict[str, Any] = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 0},
        }
        yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"

        # 5. message_stop event
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    except (CancelledError, BrokenPipeError):
        return
    except PerplexityError as exc:
        if isinstance(exc, AuthenticationError):
            await to_thread.run_sync(_client_pool.discard, token, client)
        _, error_response = error_response_for(exc)
        yield f"event: error\ndata: {error_response.model_dump_json(exclude_none=True)}\n\n"
    except Exception:
        error_response = {
            "error": {
                "message": "Streaming request failed.",
                "type": "server_error",
                "code": "streaming_error",
            }
        }
        yield f"event: error\ndata: {json.dumps(error_response)}\n\n"
    finally:
        request_lock.release()
        if release_request is not None:
            release_request()


def _next_response(iterator: Iterator[Response]) -> tuple[bool, Response | None]:
    """Advance sync stream in worker thread."""
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None
