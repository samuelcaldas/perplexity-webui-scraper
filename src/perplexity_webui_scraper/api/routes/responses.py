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
from perplexity_webui_scraper.api.routes.completions import _config_fingerprint, _conversation_cache, _validate_model
from perplexity_webui_scraper.api.schemas.request import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionTool,
    PerplexityExtensions,
)
from perplexity_webui_scraper.api.tool_calling import (
    TOOL_CALL_SENTINEL_START,
    parse_emulated_tool_call,
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
    tools: list[FunctionTool] | None = None
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
                    role_raw = item.get("role", "user")
                    role: Literal["system", "developer", "user", "assistant", "tool"] = (
                        role_raw if role_raw in {"system", "developer", "user", "assistant", "tool"} else "user"
                    )
                    content = item.get("content", "")
                    messages.append(ChatMessage(role=role, content=content))

        if not messages:
            messages.append(ChatMessage(role="user", content=""))

        return ChatCompletionRequest(
            model=self.model,
            messages=messages,
            stream=self.stream,
            reasoning_effort=self.reasoning_effort,
            thinking=self.thinking,
            tools=self.tools,
            tool_choice=self.tool_choice,
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

    emulated_tool_call = parse_emulated_tool_call(answer, request.tools, request.tool_choice)
    if requires_tool_call(request.tools, request.tool_choice) and emulated_tool_call is None:
        raise HTTPException(status_code=400, detail="Provider response did not contain a valid required tool call.")

    output: list[dict[str, Any]] = []

    if emulated_tool_call is not None:
        output.append(
            {
                "id": emulated_tool_call.call_id,
                "type": "function_call",
                "name": emulated_tool_call.function_name,
                "arguments": serialize_tool_arguments(emulated_tool_call.arguments),
                "call_id": emulated_tool_call.call_id,
            }
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
        _conversation_cache.set(
            token,
            conv_uuid,
            conversation,
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
        if has_tools and emitted_len < len(last_content):
            remaining_delta = last_content[emitted_len:]
            if remaining_delta:
                delta_event = {"delta": remaining_delta, "response_id": resp_id}
                yield f"event: response.text.delta\ndata: {json.dumps(delta_event)}\n\n"

        conv_uuid = conversation.uuid
        if conv_uuid:
            _conversation_cache.set(
                token,
                conv_uuid,
                conversation,
                config_fingerprint=config_fingerprint,
            )

        final_event: dict[str, Any] = {
            "id": resp_id,
            "object": "response",
            "status": "completed",
            "model": model_id,
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
