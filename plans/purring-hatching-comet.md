# Implementation Plan: Injected Custom Tool Multi-Turn Execution, Monitoring, and Synthesis Test Suite

## 1. Context & Objective

The user requested an end-to-end multi-turn test featuring a custom tool injected exclusively for the test. The test must:

1. Inject a dedicated custom tool schema (`run_cluster_diagnostic`) along with a control tool (`lookup_test_runbook`).
2. Guide the model via structured prompts to respond in the required test format.
3. Verify capability inquiries: ensure the model lists available tools including the test-injected tool without disclaimer.
4. Verify tool invocation: ensure the model calls the tool with exact expected parameters (`cluster_id`, `checks`, `include_raw`).
5. Monitor and record all calls made to the custom tool via a dedicated `ToolCallMonitor` test harness.
6. Feed back the tool execution results into the conversation.
7. Verify final synthesis: ensure the model's final response lists and synthesizes the tool outputs accurately.

---

## 2. Injected Custom Tool & Control Tool Design

### 2.1 Injected Tool Specification

- **Function Name**: `run_cluster_diagnostic`
- **Description**: "Run deterministic diagnostics against the isolated test cluster."
- **Parameters Schema**:
  ```json
  {
    "type": "object",
    "properties": {
      "cluster_id": { "type": "string", "enum": ["cluster-test-17"] },
      "checks": {
        "type": "array",
        "items": { "type": "string", "enum": ["dns", "storage"] },
        "minItems": 2,
        "maxItems": 2,
        "uniqueItems": true
      },
      "include_raw": { "type": "boolean", "enum": [false] }
    },
    "required": ["cluster_id", "checks", "include_raw"],
    "additionalProperties": false
  }
  ```
- **Expected Arguments**:
  ```python
  EXPECTED_DIAGNOSTIC_ARGS = {
      "cluster_id": "cluster-test-17",
      "checks": ["dns", "storage"],
      "include_raw": False,
  }
  ```

### 2.2 Control Tool Specification (to verify precision and non-execution)

- **Function Name**: `lookup_test_runbook`
- **Description**: "Look up troubleshooting runbooks for cluster alerts."
- Declared alongside `run_cluster_diagnostic` to verify:
  - Both tools appear in capability listings.
  - The model does not execute undeclared or unintended tools.

### 2.3 Deterministic Tool Output & Final Synthesized Shape

- **Tool Output** (returned by test harness executor):
  ```json
  {
    "cluster": "cluster-test-17",
    "overall": "DEGRADED",
    "checks": [
      { "name": "dns", "ok": true, "message": "resolver latency 12ms" },
      { "name": "storage", "ok": false, "message": "node-7 volume is 91% full" }
    ],
    "next_action": "drain node-7"
  }
  ```
- **Expected Synthesized Final Output**:
  ```json
  {
    "cluster_id": "cluster-test-17",
    "overall_status": "degraded",
    "findings": [
      { "check": "dns", "status": "pass", "detail": "resolver latency 12ms" },
      {
        "check": "storage",
        "status": "fail",
        "detail": "node-7 volume is 91% full"
      }
    ],
    "recommendation": "drain node-7"
  }
  ```

---

## 3. Test Harness Architecture: `ToolCallMonitor` & Endpoint Adapters

### 3.1 `ToolCallMonitor`

- Records every emitted tool call (`endpoint`, `stage`, `call_id`, `name`, `arguments`) before assertion checks.
- Intercepts and executes only the expected `run_cluster_diagnostic` tool call.
- Raises if unknown or control tools (`lookup_test_runbook`) are attempted to be executed.
- Provides inspection assertions on total calls recorded across turns.

### 3.2 3-Turn Multi-Turn Flow

1. **Turn 1 (Capability Inquiry)**:
   - Request: "Which tools are available in this session? Return exact JSON: `{"available_tools": [...]}`. Do not call any tool."
   - Assert: HTTP 200, `finish_reason == "stop"` / `stop_reason == "end_turn"`, exact JSON `{"available_tools": ["lookup_test_runbook", "run_cluster_diagnostic"]}`, 0 calls in monitor.
2. **Turn 2 (Tool Invocation)**:
   - Request: "Call run_cluster_diagnostic with cluster_id 'cluster-test-17', checks ['dns', 'storage'], include_raw false. Emit only the tool invocation."
   - Assert: HTTP 200, `finish_reason == "tool_calls"` / `stop_reason == "tool_use"`, tool call recorded in monitor with exact arguments.
3. **Turn 3 (Result Feedback & Synthesis)**:
   - Request: Feeds `[Tool Result: ...]` using emitted `call_id` and asks model to synthesize findings into structured format.
   - Assert: HTTP 200, assistant text response parses as exact expected synthesis JSON, monitor call count unchanged.

### 3.3 Endpoint Coverage

The test scenario driver runs across all 3 supported APIs:

- OpenAI `/v1/chat/completions`
- Anthropic `/v1/messages`
- OpenAI `/v1/responses`

---

## 4. File Modification & Creation Map

| File Path                                              | Description                                                                                      |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| `tests/_injected_tool_e2e_harness.py`                  | Shared harness containing tool schemas, `ToolCallMonitor`, and endpoint adapters.                |
| `tests/test_injected_tool_model_e2e.py`                | Complete multi-turn integration test suite covering contract assertions across all 3 endpoints.  |
| `src/perplexity_webui_scraper/api/routes/responses.py` | Ensure Responses API endpoint supports flat function tool schema normalization.                  |
| `src/perplexity_webui_scraper/api/tool_calling.py`     | Ensure prompt instructions support exact listing in few-shot capability demonstration if needed. |

---

## 5. Verification Steps

1. Run unit and contract test suites:
   ```bash
   uv run --all-extras pytest tests/test_injected_tool_model_e2e.py
   uv run --all-extras pytest tests/test_tool_calling_api.py
   uv run --all-extras pytest tests/test_claude_cli_integration.py
   uv run --all-extras pytest tests/test_anthropic_api.py
   uv run --all-extras pytest tests/test_responses_api.py
   uv run --all-extras pytest
   ```
2. Lint and type-check:
   ```bash
   uv run ruff check
   uv run ruff format --check
   uv run ty check
   ```
