# Plan: Implement Modern Endpoints (`POST /v1/responses`, `POST /v1/response`, `POST /v1/messages`) with Full Backward Compatibility

## Context & Problem Summary

Users and client tools (OpenWebUI, OpenCode, Claude CLI, Cursor, LiteLLM, Anthropic SDK, OpenAI Python SDK) utilize different API formats:

1. **Modern OpenAI Responses API**: `POST /v1/responses` and alias `POST /v1/response` (using `input`, `instructions`, and response object envelopes).
2. **Anthropic Messages API**: `POST /v1/messages` (standard Anthropic `messages` format with `system`, content blocks, and tool use events).
3. **OpenAI Chat Completions API**: `POST /v1/chat/completions` (existing standard endpoint, must remain 100% backward compatible).

This plan introduces protocol adapters for these modern endpoints while keeping the existing `/v1/chat/completions` endpoint and underlying `Conversation` / token session synchronization intact.

---

## Architectural Approach

We keep the core Perplexity scraping engine, token session management (`client_pool`), and conversation lifecycle unified, adding two thin routing adapters in `src/perplexity_webui_scraper/api/routes/`:

```
                                  ┌───────────────────────────┐
                                  │  FastAPI Application      │
                                  └─────────────┬─────────────┘
                                                │
        ┌───────────────────────────────────────┼───────────────────────────────────────┐
        ▼                                       ▼                                       ▼
POST /v1/chat/completions             POST /v1/responses &                    POST /v1/messages
(Existing Chat Completions)           POST /v1/response                       (Anthropic Messages API)
        │                             (OpenAI Responses API)                            │
        │                                       │                                       │
        └───────────────────────────────────────┴───────────────────────────────────────┘
                                                │
                                                ▼
                               ┌──────────────────────────────────┐
                               │ Adapter & Translation Middleware │
                               └────────────────┬─────────────────┘
                                                │
                                                ▼
                               ┌──────────────────────────────────┐
                               │ Perplexity Scraper Core Engine   │
                               │ (Session pool, conversation lock)│
                               └──────────────────────────────────┘
```

---

## Proposed Changes

### 1. OpenAI Responses API (`POST /v1/responses` & `POST /v1/response`)

- **Route File**: `src/perplexity_webui_scraper/api/routes/responses.py`
- **Request Adapter**:
  - Accepts `model`, `input` (string or list of items), `instructions` (system prompt), `stream` (bool), `tools`, `perplexity` extensions.
  - Converts `input` + `instructions` into internal `ChatMessage` list.
- **Response Formatter**:
  - Non-streaming: Returns standard OpenAI Responses payload (`id`, `object="response"`, `status="completed"`, `output=[{"type": "message", "role": "assistant", "content": [{"type": "text", "text": "..."}]}]`).
  - Streaming: Emits standard Responses SSE events (`response.created`, `response.text.delta`, `response.text.done`, `response.done`, `data: [DONE]`).

### 2. Anthropic Messages API (`POST /v1/messages`)

- **Route File**: `src/perplexity_webui_scraper/api/routes/messages.py`
- **Request Adapter**:
  - Accepts `model`, `messages` (Anthropic format: `role`, `content` as string or list of text/image content blocks), `system` (string or blocks), `stream` (bool), `tools` (Anthropic format with `name`, `description`, `input_schema`), `thinking` / `reasoning_effort`.
  - Converts Anthropic messages + system prompt into `ChatMessage` format.
- **Response Formatter**:
  - Non-streaming: Returns standard Anthropic message payload (`id="msg_..."`, `type="message"`, `role="assistant"`, `content=[{"type": "text", "text": "..."}]`, `stop_reason="end_turn"`).
  - Streaming: Emits standard Anthropic SSE stream (`message_start`, `content_block_start`, `content_block_delta` with `text_delta`, `content_block_stop`, `message_delta`, `message_stop`).

### 3. Application Route Registration

- Update `src/perplexity_webui_scraper/api/app.py`:
  - Include `responses_router` and `messages_router`.

### 4. Tests

- `tests/test_responses_api.py`:
  - Test `POST /v1/responses` non-streaming and streaming.
  - Test `POST /v1/response` alias non-streaming and streaming.
- `tests/test_anthropic_api.py`:
  - Test `POST /v1/messages` non-streaming and streaming.
  - Test Anthropic tool call parsing and content blocks.
- Ensure all existing tests in `tests/test_openai_api_compatibility.py` and `tests/test_claude_cli_integration.py` continue passing.

### 5. Redeploy & Live Validation

- Run local tests and static checks (`uv run --all-extras pytest`, `uv run ruff check`, `pnpm prettier --check .`, etc.).
- Redeploy to VPS with `bash ../deploy/vps/perplexity-sidecar-chat/update.sh`.
- Run live validation against production server:
  - Test `POST /v1/responses` via curl / python script.
  - Test `POST /v1/messages` via curl / python script.
  - Test `POST /v1/chat/completions` with streaming to verify no regressions.

---

## Verification Plan

1. **Automated Unit & Integration Tests**:
   - `uv run --all-extras pytest` (all tests passing).
   - `uv run ruff check && uv run ty check && pnpm prettier --check . && pnpm taplo lint *.toml`.
2. **Live Production Smoke Tests**:
   - Verify `/v1/chat/completions` with OpenAI Python SDK.
   - Verify `/v1/responses` and `/v1/response` with OpenAI SDK / curl.
   - Verify `/v1/messages` with Anthropic Python SDK / curl.
