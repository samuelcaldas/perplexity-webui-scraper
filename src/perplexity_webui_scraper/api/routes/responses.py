"""POST /v1/responses and POST /v1/response routes — modern OpenAI Responses API."""

from __future__ import annotations

from asyncio import CancelledError, Lock
from functools import partial
import json
from time import time
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


class ResponseApiRequest(BaseModel):
    """Schema for OpenAI Responses API requests."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    input: str | list[Any] = Field(min_length=1)
    instructions: str | None = None
    stream: bool = False
    reasoning_effort: Literal["low", "medium", "high", "none"] | None = None
    thinking: bool | None = None
    tools: list[Any] | None = None
    tool_choice: Any = None
    perplexity: PerplexityExtensions | None = None

    def to_chat_completion_request(self) -> ChatCompletionRequest:
        """Convert Responses API request to internal ChatCompletionRequest."""
        messages: list[ChatMessage] = []

        if self.instructions:
            messages.append(ChatMessage(role="system", content=self.instructions))

        if isinstance(self.input, str):
            messages.append(ChatMessage(role="user", content=self.input))
        elif isinstance(self.input, list):
            for item in self.input:
                if isinstance(item, str):
                    messages.append(ChatMessage(role="user", content=item))
                elif isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "function_call":
                        call_id = str(item.get("call_id", item.get("id", f"call_{uuid4().hex[:16]}")))
                        name = str(item.get("name", ""))
                        raw_args = item.get("arguments", "{}")
                        args_str = (
                            json.dumps(raw_args, ensure_ascii=False) if isinstance(raw_args, dict) else str(raw_args)
                        )
                        messages.append(
                            ChatMessage(
                                role="assistant",
                                content=None,
                                tool_calls=[
                                    AssistantToolCall(
                                        id=call_id,
                                        type="function",
                                        function=ToolCallFunction(name=name, arguments=args_str),
                                    )
                                ],
                            )
                        )
                    elif item_type == "function_call_output":
                        call_id = str(item.get("call_id", ""))
                        output = item.get("output", "")
                        out_str = json.dumps(output, ensure_ascii=False) if isinstance(output, dict) else str(output)
                        messages.append(
                            ChatMessage(
                                role="tool",
                                tool_call_id=call_id,
                                content=out_str,
                            )
                        )
                    elif item_type == "message":
                        role_raw = str(item.get("role", "user"))
                        role: Literal["system", "developer", "user", "assistant", "tool"] = (
                            role_raw if role_raw in {"system", "developer", "user", "assistant", "tool"} else "user"
                        )
                        content = item.get("content", "")
                        if isinstance(content, str):
                            messages.append(ChatMessage(role=role, content=content))
                        elif isinstance(content, list):
                            parts: list[ContentPartText | ContentPartImageUrl] = []
                            for part in content:
                                if not isinstance(part, dict):
                                    continue
                                p_type = part.get("type")
                                if p_type in {"text", "input_text"}:
                                    parts.append(ContentPartText(type="text", text=str(part.get("text", ""))))
                                elif p_type in {"image_url", "input_image"}:
                                    img_url = part.get("image_url", part.get("url", ""))
                                    if isinstance(img_url, dict):
                                        img_url = img_url.get("url", "")
                                    parts.append(ContentPartImageUrl(type="image_url", image_url={"url": str(img_url)}))
                            messages.append(ChatMessage(role=role, content=parts))
                    else:
                        role_raw = str(item.get("role", "user"))
                        role = role_raw if role_raw in {"system", "developer", "user", "assistant", "tool"} else "user"
                        content = item.get("content", "")
                        messages.append(ChatMessage(role=role, content=content))

        if not messages:
            messages.append(ChatMessage(role="user", content=""))

        normalized_tools: list[FunctionTool] | None = None
        if self.tools:
            normalized_tools = []
            for tool in self.tools:
                if isinstance(tool, FunctionTool):
                    normalized_tools.append(tool)
                elif isinstance(tool, dict):
                    if "function" in tool and isinstance(tool["function"], dict):
                        normalized_tools.append(FunctionTool.model_validate(tool))
                    elif tool.get("type") == "function" and "name" in tool:
                        fn_def = FunctionDefinition(
                            name=str(tool["name"]),
                            description=tool.get("description"),
                            parameters=tool.get("parameters") or {},
                            strict=tool.get("strict"),
                        )
                        normalized_tools.append(FunctionTool(type="function", function=fn_def))
                    else:
                        normalized_tools.append(FunctionTool.model_validate(tool))

        mapped_tool_choice: ToolChoice | None = None
        if isinstance(self.tool_choice, dict):
            tc_type = self.tool_choice.get("type")
            if tc_type == "auto":
                mapped_tool_choice = "auto"
            elif tc_type in {"required", "any"}:
                mapped_tool_choice = "required"
            elif tc_type == "none":
                mapped_tool_choice = "none"
            elif tc_type == "function":
                fn_name: str | None = None
                if "name" in self.tool_choice:
                    fn_name = str(self.tool_choice["name"])
                elif "function" in self.tool_choice and isinstance(self.tool_choice["function"], dict):
                    fn_name = str(self.tool_choice["function"].get("name", ""))
                if fn_name:
                    mapped_tool_choice = ExplicitToolChoice(
                        type="function",
                        function=ToolChoiceFunction(name=fn_name),
                    )
        elif isinstance(self.tool_choice, str):
            if self.tool_choice in {"auto", "required", "none"}:
                mapped_tool_choice = self.tool_choice
            elif self.tool_choice == "any":
                mapped_tool_choice = "required"

        return ChatCompletionRequest(
            model=self.model,
            messages=messages,
            stream=self.stream,
            reasoning_effort=self.reasoning_effort,
            thinking=self.thinking,
            tools=normalized_tools,
            tool_choice=mapped_tool_choice,
            perplexity=self.perplexity,
        )


@router.post("/v1/responses", response_model=None)
@router.post("/v1/response", response_model=None)
async def responses_endpoint(
    raw_request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> JSONResponse | StreamingResponse:
    """Handle OpenAI Responses API requests."""
    try:
        body = await raw_request.json()
        resp_req = ResponseApiRequest.model_validate(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    request = resp_req.to_chat_completion_request()
    token = extract_token(authorization)
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
                _stream_responses_api(
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
        return _build_responses_payload(request, conversation, token)
    except AuthenticationError:
        if client is not None:
            await to_thread.run_sync(_client_pool.discard, token, client)
        raise
    finally:
        if not stream_handoff:
            if lock_acquired:
                request_lock.release()
            _client_pool.release_request_lock(token, request_lock)


def _build_responses_payload(
    request: ChatCompletionRequest,
    conversation: Conversation,
    token: str,
) -> JSONResponse:
    """Build response payload matching OpenAI Responses API format."""
    answer = conversation.answer or ""
    resp_id = f"resp_{uuid4().hex}"
    created_at = int(time())

    emulated_tool_calls = parse_emulated_tool_calls(answer, request.tools, request.tool_choice)
    if requires_tool_call(request.tools, request.tool_choice) and not emulated_tool_calls:
        raise HTTPException(status_code=400, detail="Provider response did not contain a valid required tool call.")

    output: list[dict[str, Any]] = []

    if emulated_tool_calls:
        first_sentinel_pos = answer.find(TOOL_CALL_SENTINEL_START)
        if first_sentinel_pos > 0:
            prefix_text = answer[:first_sentinel_pos].strip()
            if prefix_text:
                output.append(
                    {
                        "id": f"msg_{uuid4().hex[:16]}",
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": prefix_text,
                            }
                        ],
                    }
                )
        output.extend(
            {
                "id": call.call_id,
                "type": "function_call",
                "name": call.function_name,
                "arguments": serialize_tool_arguments(call.arguments),
                "call_id": call.call_id,
            }
            for call in emulated_tool_calls
        )
    else:
        output.append(
            {
                "id": f"msg_{uuid4().hex[:16]}",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": answer,
                    }
                ],
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
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "model": request.model,
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }

    if conv_uuid:
        payload["perplexity"] = {"thread_uuid": conv_uuid}

    return JSONResponse(content=payload)


async def _stream_responses_api(
    conversation: Conversation,
    request: ChatCompletionRequest,
    token: str,
    client: Perplexity,
    request_lock: Lock,
    release_request: Callable[[], None] | None = None,
    config_fingerprint: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield OpenAI Responses API SSE stream events."""
    resp_id = f"resp_{uuid4().hex}"
    created_at = int(time())
    last_content = ""
    sentinel_started = False
    emitted_len = 0
    has_tools = bool(request.tools)
    model_id = request.model
    output_index = 0

    try:
        iterator = iter(conversation)

        # Initial response.created event
        init_event = {
            "id": resp_id,
            "object": "response",
            "created_at": created_at,
            "model": model_id,
            "status": "in_progress",
        }
        yield f"event: response.created\ndata: {json.dumps(init_event)}\n\n"

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
                            delta_event = {"delta": delta, "response_id": resp_id}
                            yield f"event: response.text.delta\ndata: {json.dumps(delta_event)}\n\n"
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
                            delta_event = {"delta": delta, "response_id": resp_id}
                            yield f"event: response.text.delta\ndata: {json.dumps(delta_event)}\n\n"
            else:
                delta = current[emitted_len:]
                if delta:
                    emitted_len = len(current)
                    delta_event = {"delta": delta, "response_id": resp_id}
                    yield f"event: response.text.delta\ndata: {json.dumps(delta_event)}\n\n"

        # Final delta flush if tools were active but no sentinel occurred
        if has_tools and not sentinel_started and emitted_len < len(last_content):
            remaining_delta = last_content[emitted_len:]
            if remaining_delta:
                delta_event = {"delta": remaining_delta, "response_id": resp_id}
                yield f"event: response.text.delta\ndata: {json.dumps(delta_event)}\n\n"

        emulated_calls = (
            parse_emulated_tool_calls(last_content, request.tools, request.tool_choice) if has_tools else None
        )

        output_items: list[dict[str, Any]] = []

        if emulated_calls:
            first_sentinel_pos = last_content.find(TOOL_CALL_SENTINEL_START)
            if first_sentinel_pos > 0:
                prefix_text = last_content[:first_sentinel_pos].strip()
                if prefix_text:
                    output_items.append(
                        {
                            "id": f"msg_{uuid4().hex[:16]}",
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "text", "text": prefix_text}],
                        }
                    )
                    output_index += 1

            for call in emulated_calls:
                call_args = serialize_tool_arguments(call.arguments)
                item_dict = {
                    "id": call.call_id,
                    "type": "function_call",
                    "name": call.function_name,
                    "arguments": call_args,
                    "call_id": call.call_id,
                }
                output_items.append(item_dict)

                item_added_event = {
                    "type": "response.output_item.added",
                    "response_id": resp_id,
                    "output_index": output_index,
                    "item": {
                        "id": call.call_id,
                        "type": "function_call",
                        "name": call.function_name,
                        "arguments": "",
                        "call_id": call.call_id,
                    },
                }
                yield f"event: response.output_item.added\ndata: {json.dumps(item_added_event)}\n\n"

                arg_delta_event = {
                    "type": "response.function_call_arguments.delta",
                    "response_id": resp_id,
                    "call_id": call.call_id,
                    "output_index": output_index,
                    "delta": call_args,
                }
                yield f"event: response.function_call_arguments.delta\ndata: {json.dumps(arg_delta_event)}\n\n"

                arg_done_event = {
                    "type": "response.function_call_arguments.done",
                    "response_id": resp_id,
                    "call_id": call.call_id,
                    "output_index": output_index,
                    "arguments": call_args,
                }
                yield f"event: response.function_call_arguments.done\ndata: {json.dumps(arg_done_event)}\n\n"

                item_done_event = {
                    "type": "response.output_item.done",
                    "response_id": resp_id,
                    "output_index": output_index,
                    "item": item_dict,
                }
                yield f"event: response.output_item.done\ndata: {json.dumps(item_done_event)}\n\n"
                output_index += 1
        else:
            output_items.append(
                {
                    "id": f"msg_{uuid4().hex[:16]}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": last_content}],
                }
            )

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

        final_event: dict[str, Any] = {
            "id": resp_id,
            "object": "response",
            "status": "completed",
            "model": model_id,
            "output": output_items,
        }
        if conv_uuid:
            final_event["perplexity"] = {"thread_uuid": conv_uuid}

        yield f"event: response.done\ndata: {json.dumps(final_event)}\n\n"
        yield "data: [DONE]\n\n"
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
