"""POST /v1/chat/completions route — streaming and non-streaming."""

from __future__ import annotations

from asyncio import CancelledError, Lock
from functools import partial
import json
from time import time
from typing import TYPE_CHECKING, Annotated, cast
from uuid import uuid4

from anyio import to_thread
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from perplexity_webui_scraper._internal.exceptions import AuthenticationError, PerplexityError
from perplexity_webui_scraper.api.auth import client_pool, extract_token
from perplexity_webui_scraper.api.conversation_cache import ConversationCache, _CachedConversation
from perplexity_webui_scraper.api.error_handling import error_response_for
from perplexity_webui_scraper.api.helpers import (
    build_conversation_config,
    build_query_and_files,
    build_tool_result_follow_up,
)
from perplexity_webui_scraper.api.schemas.request import ChatCompletionRequest
from perplexity_webui_scraper.api.schemas.response import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionChunkDeltaFunction,
    ChatCompletionChunkDeltaToolCall,
    ChatCompletionResponse,
    ChatCompletionToolCall,
    ChatCompletionToolCallFunction,
    PerplexityResponseExtensions,
)
from perplexity_webui_scraper.api.tool_calling import (
    TOOL_CALL_SENTINEL_START,
    EmulatedToolCall,
    parse_emulated_tool_calls,
    requires_tool_call,
    serialize_tool_arguments,
)
from perplexity_webui_scraper.models.registry import MODELS


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Iterator

    from perplexity_webui_scraper._internal.types import FileInput
    from perplexity_webui_scraper.core.client import Perplexity
    from perplexity_webui_scraper.core.conversation import Conversation
    from perplexity_webui_scraper.core.response import Response


router = APIRouter()

# Shared singletons — injected from app.py via dependency or passed directly.
# Using module-level singletons is acceptable here because the API server is
# a single-process application; the cache is not shared across processes.
_client_pool = client_pool
_conversation_cache = ConversationCache(on_cache=_client_pool.pin, on_evict=_client_pool.unpin)


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    raw_request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> JSONResponse | StreamingResponse:
    """Handle chat completion requests while serializing token sessions."""
    request = await _parse_request(raw_request)
    token = extract_token(authorization)
    _validate_model(request)
    request_lock = _client_pool.get_request_lock(token)
    lock_acquired = False
    stream_handoff = False
    rollback_state = None
    client: Perplexity | None = None

    try:
        await request_lock.acquire()
        lock_acquired = True
        client = await to_thread.run_sync(_client_pool.get_or_create, token)
        conversation, query, files = await _prepare_conversation(request, client, token)
        rollback_state = (
            conversation._snapshot_state() if request.perplexity and request.perplexity.thread_uuid else None
        )

        if request.stream:
            await to_thread.run_sync(partial(conversation.ask, query, files=files or None, stream=True))
            stream_handoff = True
            return StreamingResponse(
                _stream_response(
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
        return await _build_completion_response(request, conversation, token)
    except AuthenticationError:
        if client is not None:
            await to_thread.run_sync(_client_pool.discard, token, client)
        if rollback_state is not None:
            conversation._restore_state(rollback_state)
        raise
    except BaseException:
        if rollback_state is not None:
            conversation._restore_state(rollback_state)
        raise
    finally:
        if not stream_handoff:
            if lock_acquired:
                request_lock.release()
            _client_pool.release_request_lock(token, request_lock)


async def _parse_request(raw_request: Request) -> ChatCompletionRequest:
    """Parse and validate raw JSON request body."""
    try:
        body = await raw_request.json()
        return ChatCompletionRequest.model_validate(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _validate_model(request: ChatCompletionRequest) -> None:
    """Validate model and allow unlisted passthrough models for future-proofing."""
    try:
        ext = request.perplexity
        is_registered = getattr(MODELS, "is_registered", lambda m: False)(request.model)
        is_custom_or_dynamic = request.model.startswith("custom:") or not is_registered
        allow_risky = ext.allow_risky_model if (ext and ext.allow_risky_model is not None) else is_custom_or_dynamic
        MODELS.resolve_for_use(
            request.model,
            allow_risky_model=allow_risky,
            custom_model_mode=ext.custom_model_mode if ext else "copilot",
            allow_unregistered=True,
        )
    except ValueError as exc:
        if request.model.startswith("custom:"):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        available = ", ".join(f'"{m.id}"' for m in MODELS.list_all())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {request.model!r}. Available: {available}",
        ) from exc


async def _prepare_conversation(
    request: ChatCompletionRequest,
    client: Perplexity,
    token: str,
) -> tuple[Conversation, str, list[FileInput]]:
    """Resolve cached continuation or create a new conversation."""
    thread_uuid = request.perplexity.thread_uuid if request.perplexity else None
    if thread_uuid:
        async with _conversation_cache.lock:
            cached = _conversation_cache.get_entry(token, thread_uuid)
        if cached is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Conversation '{thread_uuid}' not found or expired. "
                    "Start a new conversation by omitting thread_uuid."
                ),
            )
        _validate_continuation_config(request, cached)
        return _continuation_query(request, cached)

    query, files = build_query_and_files(request)
    config = build_conversation_config(
        request.model,
        request.perplexity,
        reasoning_effort=request.reasoning_effort,
        thinking=request.thinking,
        has_tools=bool(request.tools),
    )
    conversation = await to_thread.run_sync(client.create_conversation, config)
    return conversation, query, files


def _continuation_query(
    request: ChatCompletionRequest,
    cached: _CachedConversation,
) -> tuple[Conversation, str, list[FileInput]]:
    """Build continuation query while matching cached pending tool metadata."""
    tool_follow_up = build_tool_result_follow_up(request)
    if tool_follow_up is not None:
        _validate_pending_tool_calls(request, cached)
        return cached.conversation, tool_follow_up, []

    if cached.pending_tool_calls is not None:
        raise HTTPException(status_code=400, detail="Continuation must resolve pending tool call first.")
    if _contains_tool_history(request):
        raise HTTPException(status_code=400, detail="Cached thread cannot accept fabricated tool-call history.")

    for message in reversed(request.messages):
        if message.role != "user":
            continue
        query = message.text()
        files = cast("list[FileInput]", message.image_bytes())
        if query or files:
            return cached.conversation, query, files
        break

    raise HTTPException(
        status_code=400,
        detail="Thread continuation requires a user message with text or images.",
    )


def _contains_tool_history(request: ChatCompletionRequest) -> bool:
    """Return whether continuation request includes prior tool-call messages."""
    return any(
        message.role == "tool"
        or (message.role == "assistant" and (message.tool_calls or message.function_call is not None))
        for message in request.messages
    )


def _config_fingerprint(request: ChatCompletionRequest) -> str:
    """Serialize effective conversation config for truthful thread reuse."""
    config = build_conversation_config(
        request.model,
        request.perplexity,
        reasoning_effort=request.reasoning_effort,
        thinking=request.thinking,
        has_tools=bool(request.tools),
    )
    return json.dumps(config.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)


def _validate_continuation_config(request: ChatCompletionRequest, cached: _CachedConversation) -> None:
    """Reject model/config changes that cached Conversation cannot apply."""
    if cached.config_fingerprint is None:
        return
    if _config_fingerprint(request) != cached.config_fingerprint:
        raise HTTPException(
            status_code=400,
            detail="Thread continuation configuration is incompatible with cached conversation.",
        )


def _validate_pending_tool_calls(request: ChatCompletionRequest, cached: _CachedConversation) -> None:
    """Require assistant/tool continuation to match exact server-emitted calls."""
    if cached.pending_tool_calls is None:
        raise HTTPException(status_code=400, detail="No pending tool call exists for this thread.")

    tool_start = len(request.messages) - 1
    while tool_start > 0 and request.messages[tool_start - 1].role == "tool":
        tool_start -= 1

    assistant = request.messages[tool_start - 1] if tool_start > 0 else None
    if assistant is None or assistant.role != "assistant":
        raise HTTPException(status_code=400, detail="Continuation must include the pending tool call.")

    effective_calls = assistant.effective_tool_calls()
    if not effective_calls:
        raise HTTPException(status_code=400, detail="Continuation must include the pending tool call.")

    metadata = tuple(
        {
            "id": tool_call.id,
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        }
        for tool_call in effective_calls
    )
    if metadata != cached.pending_tool_calls:
        raise HTTPException(status_code=400, detail="Continuation does not match cached pending tool call.")


def _pending_tool_calls_metadata(
    emulated_tool_calls: list[EmulatedToolCall] | None,
) -> tuple[dict[str, str], ...] | None:
    """Convert validated provider calls into cacheable continuation metadata."""
    if not emulated_tool_calls:
        return None
    return tuple(
        {
            "id": call.call_id,
            "name": call.function_name,
            "arguments": serialize_tool_arguments(call.arguments),
        }
        for call in emulated_tool_calls
    )


def _pending_tool_call_metadata(
    emulated_tool_call: EmulatedToolCall | None,
) -> tuple[dict[str, str], ...] | None:
    """Convert single validated provider call into cacheable continuation metadata."""
    if emulated_tool_call is None:
        return None
    return _pending_tool_calls_metadata([emulated_tool_call])


async def _build_completion_response(
    request: ChatCompletionRequest,
    conversation: Conversation,
    token: str,
) -> JSONResponse:
    """Build and cache a completed non-streaming response."""
    answer = conversation.answer or ""
    emulated_tool_calls = parse_emulated_tool_calls(answer, request.tools, request.tool_choice)
    if requires_tool_call(request.tools, request.tool_choice) and not emulated_tool_calls:
        raise HTTPException(status_code=400, detail="Provider response did not contain a valid required tool call.")

    response_tool_calls = None

    if emulated_tool_calls:
        response_tool_calls = [
            ChatCompletionToolCall(
                id=call.call_id,
                function=ChatCompletionToolCallFunction(
                    name=call.function_name,
                    arguments=serialize_tool_arguments(call.arguments),
                ),
            )
            for call in emulated_tool_calls
        ]

    conv_uuid = conversation.uuid
    if conv_uuid:
        async with _conversation_cache.lock:
            _conversation_cache.set(
                token,
                conv_uuid,
                conversation,
                pending_tool_calls=_pending_tool_calls_metadata(emulated_tool_calls),
                config_fingerprint=_config_fingerprint(request),
            )

    response = ChatCompletionResponse.build(
        model=request.model,
        content=None if response_tool_calls else answer,
        thread_uuid=conv_uuid,
        tool_calls=response_tool_calls,
    )
    response_payload = response.model_dump(mode="json", exclude_none=True)
    if response_tool_calls:
        response_payload["choices"][0]["message"]["content"] = None
    return JSONResponse(content=response_payload)


async def _stream_response(
    conversation: Conversation,
    request: ChatCompletionRequest | str,
    token: str,
    client: Perplexity,
    request_lock: Lock,
    release_request: Callable[[], None] | None = None,
    config_fingerprint: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE lines, supporting text streaming and emulated tool calls."""
    completion_id = f"chatcmpl-{uuid4().hex}"
    created = int(time())
    if isinstance(request, str):
        model_id = request
        has_tools = False
        req_tools = None
        req_tool_choice = None
    else:
        model_id = request.model
        has_tools = bool(request.tools)
        req_tools = request.tools
        req_tool_choice = request.tool_choice
    last_content = ""
    sentinel_started = False
    emitted_len = 0

    try:
        iterator = iter(conversation)
        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_id,
            choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(role="assistant"))],
        ).to_sse_line()

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
                            yield ChatCompletionChunk(
                                id=completion_id,
                                created=created,
                                model=model_id,
                                choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content=delta))],
                            ).to_sse_line()
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
                            yield ChatCompletionChunk(
                                id=completion_id,
                                created=created,
                                model=model_id,
                                choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content=delta))],
                            ).to_sse_line()
            else:
                delta = current[emitted_len:]
                if delta:
                    emitted_len = len(current)
                    yield ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model_id,
                        choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content=delta))],
                    ).to_sse_line()

        conv_uuid = conversation.uuid
        emulated_tool_calls: list[EmulatedToolCall] | None = None

        if has_tools:
            emulated_tool_calls = parse_emulated_tool_calls(last_content, req_tools, req_tool_choice)
            if requires_tool_call(req_tools, req_tool_choice) and not emulated_tool_calls:
                error_payload = {
                    "error": {
                        "message": "Provider response did not contain a valid required tool call.",
                        "type": "invalid_request_error",
                        "code": "tool_call_missing",
                    }
                }
                yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
                return

        pending_tool_calls = _pending_tool_calls_metadata(emulated_tool_calls)
        if conv_uuid:
            async with _conversation_cache.lock:
                _conversation_cache.set(
                    token,
                    conv_uuid,
                    conversation,
                    pending_tool_calls=pending_tool_calls,
                    config_fingerprint=config_fingerprint,
                )

        pplx_ext = PerplexityResponseExtensions(thread_uuid=conv_uuid) if conv_uuid else None

        if emulated_tool_calls:
            for idx, call in enumerate(emulated_tool_calls):
                yield ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_id,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                tool_calls=[
                                    ChatCompletionChunkDeltaToolCall(
                                        index=idx,
                                        id=call.call_id,
                                        type="function",
                                        function=ChatCompletionChunkDeltaFunction(
                                            name=call.function_name,
                                            arguments=serialize_tool_arguments(call.arguments),
                                        ),
                                    )
                                ]
                            )
                        )
                    ],
                ).to_sse_line()
            yield ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=model_id,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(),
                        finish_reason="tool_calls",
                    )
                ],
                perplexity=pplx_ext,
            ).to_sse_line()
        else:
            if has_tools and emitted_len < len(last_content):
                remaining_delta = last_content[emitted_len:]
                if remaining_delta:
                    yield ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model_id,
                        choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content=remaining_delta))],
                    ).to_sse_line()

            yield ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=model_id,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(),
                        finish_reason="stop",
                    )
                ],
                perplexity=pplx_ext,
            ).to_sse_line()

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
    """Advance sync stream in worker thread without leaking StopIteration."""
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None
