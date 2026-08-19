# Investigation Plan: Truncated Model Responses in Streaming / Completion

## Overview

Investigate why model responses streamed or completed in OpenWebUI / OpenCode / Claude-CLI are truncated (missing letters at the beginning and/or missing letters at the end) after tool calling support was introduced.

## Findings & Root Causes Identified

### Root Cause 1: Streaming Prefix Truncation / Over-Stripping (`safe_len` Lookahead Logic)

- **Location:** `src/perplexity_webui_scraper/api/routes/completions.py`, lines 406–421
- **Mechanism:**
  When `has_tools` is True (e.g. tools are passed in the request, even if the model responds with regular text), `_stream_response` uses a lookahead loop to avoid leaking partial tool sentinel strings (`<|OPENAI_TOOL_CALL|>`):

  ```python
  safe_len = len(current)
  for i in range(1, min(len(TOOL_CALL_SENTINEL_START), len(current)) + 1):
      if TOOL_CALL_SENTINEL_START.startswith(current[-i:]):
          safe_len = len(current) - i
          break
  ```

  `TOOL_CALL_SENTINEL_START` is `"<|OPENAI_TOOL_CALL|>"`.
  If the model output contains ANY string or character sequence that matches a prefix of `"<|OPENAI_TOOL_CALL|>"`, such as:
  - `"<"` (matches `"<|OPENAI_TOOL_CALL|>"[:1]`)
  - `"<|"`
  - `"<|O"`
  - `<|OPENAI_TOOL_CALL|>` inside standard text or code blocks
  - Any HTML / XML tag starting with `<` (e.g., `<html>`, `<div>`, `<script>`, `<thought>`, `<anthropic...`)

  The logic computes `safe_len = len(current) - i` and holds back `i` characters. On subsequent chunks (or at the end of stream), if `sentinel_started` remains False, `emitted_len` was advanced up to `safe_len`. But if a new chunk arrives that also ends with `<` or matches `TOOL_CALL_SENTINEL_START.startswith(...)`, or if the trailing slice is held until stream completion, the held-back characters are skipped or delayed!

  Furthermore, if `current` has text _before_ or _around_ sentinel-like sequences, `current[-i:]` checks only the end of `current`. But if `<` appears earlier or if `TOOL_CALL_SENTINEL_START` appears in `current`, `sentinel_pos` truncates all text prior to `sentinel_pos` if `emitted_len` hasn't caught up, or drops everything after `sentinel_pos` without streaming content preceding the sentinel if `sentinel_started` was already True.

### Root Cause 2: Streaming Suffix Truncation (End of Stream Held Buffer Dropped)

- **Location:** `src/perplexity_webui_scraper/api/routes/completions.py`, lines 498–506
- **Mechanism:**
  During tool-enabled streaming, chunks are buffered up to `emitted_len` (which was held back by `safe_len` to prevent sentinel leakage).
  When the loop `while True:` ends (stream finished), `last_content` holds the full upstream text.
  If `emulated_tool_call is None` (the response was normal text, NOT a tool call):
  ```python
  if has_tools and emitted_len < len(last_content):
      remaining_delta = last_content[emitted_len:]
      if remaining_delta:
          yield ChatCompletionChunk(...content=remaining_delta...)
  ```
  However:
  1. If `TOOL_CALL_SENTINEL_START` was present in `current` (e.g., a text response that discussed tool calls or contained `<|OPENAI_TOOL_CALL|>`), `sentinel_started` became `True` and `emitted_len` was set to `sentinel_pos`. The code assumes everything from `sentinel_pos` onwards is a tool call payload. But if `parse_emulated_tool_call` fails to parse a valid tool call (e.g. malformed or invalid JSON), `emulated_tool_call` is `None`. `remaining_delta` yields `last_content[sentinel_pos:]`, which includes raw sentinel text `"<|OPENAI_TOOL_CALL|>..."` mixed into text, OR if `sentinel_pos` was at the end or if `TOOL_CALL_SENTINEL_START` occurred in the middle, text following sentinel is emitted as raw text while text inside sentinel is lost/corrupted.
  2. More critically: If the response ends with characters matching a prefix of `TOOL_CALL_SENTINEL_START` (e.g. the response ends with `<foo>` or code ending with `<`), `safe_len` held back those trailing characters during the loop. At the end of the stream, `remaining_delta = last_content[emitted_len:]` emits them in one final chunk. BUT many OpenAI/Claude streaming clients (like OpenWebUI or Claude-CLI) handle `finish_reason="stop"` chunks or final deltas differently: if the remaining delta is sent in the same chunk as `finish_reason="stop"` or if client UI expects deltas before finish_reason, some UIs drop deltas attached to `finish_reason="stop"` chunks or miss the trailing characters if `remaining_delta` was held back.

### Root Cause 3: Perplexity SSE Chunks vs `last_chunk` vs `answer` Cumulative Drift

- **Location:** `src/perplexity_webui_scraper/api/routes/completions.py`, line 386 & `src/perplexity_webui_scraper/core/parser.py`
- **Mechanism:**
  In `_stream_response`:
  ```python
  current = response.last_chunk or response.answer or ""
  ```
  In `perplexity_webui_scraper`, Perplexity returns cumulative `answer` in SSE events (or incremental `markdown_chunks`).
  In commit 46cf82d, delta calculation changed from `commonprefix([last_content, current])` to offset slicing `current[emitted_len:]`.
  `emitted_len` tracks index into `current`. But:
  - If `response.last_chunk` is populated with a single incremental chunk instead of the full accumulated text, `current` is ONLY the last chunk (short string), whereas `emitted_len` was tracked against the cumulative string length! `current[emitted_len:]` then evaluates to `""` (slice beyond string length) or incorrect sub-slice, skipping entire chunks of text!
  - Conversely, if `current` switches between `response.answer` (cumulative string, e.g. length 500) and `response.last_chunk` (incremental chunk, e.g. length 20), `emitted_len` (e.g. 480) applied to `response.last_chunk` yields empty string, causing streaming to freeze or miss letters.

### Root Cause 4: Non-Streaming Response Content Stripping (`content=None` when tool calls emulated)

- **Location:** `src/perplexity_webui_scraper/api/routes/completions.py`, lines 336 & 342
- **Mechanism:**
  In non-streaming `_build_completion_response`:
  ```python
  response = ChatCompletionResponse.build(
      model=request.model,
      content=None if response_tool_calls else answer,
      ...
  )
  ```
  If `emulated_tool_call` is detected, `content` is set to `None`. If the model output text before or after the sentinel, that text is completely discarded.

## Detailed Fix Strategy

1. **Fix Offset Tracking & Stream Delta Generation:**
   Always track `emitted_len` strictly against cumulative `response.answer` (or accumulated text `current`), ensuring `current` is always the full cumulative text snapshot rather than relying on `response.last_chunk` which might be non-cumulative.

2. **Fix Sentinel Lookahead Buffer Logic:**
   - Instead of checking if any arbitrary trailing substring matches `TOOL_CALL_SENTINEL_START.startswith(...)` which falsely triggers on `<` or any HTML tag, match sentinel boundary accurately.
   - Buffer only up to `len(TOOL_CALL_SENTINEL_START) - 1` characters at the very end of stream IF and ONLY IF the trailing string is an exact prefix match of `TOOL_CALL_SENTINEL_START`.
   - When stream ends and no tool call sentinel is parsed, flush ALL buffered/remaining text cleanly before yielding `finish_reason="stop"`.

3. **Separate Delta Chunk from Finish Reason Chunk:**
   Ensure final text deltas are yielded in a standard delta chunk _before_ the final `finish_reason="stop"` chunk so client parsers in OpenWebUI/Claude-CLI do not drop trailing letters.

## Deliverables

- Comprehensive investigation report with exact line numbers, code traces, and proposed fixes.
