# Implementation Plan: Hardened Tool Calling In-Context Few-Shots, Tail Reminders & Strict Prompt Verification Test Suite

## 1. Context & Problem Statement

### 1.1 The Live LLM Failure Modes

When proxying reasoning and agentic LLMs (Claude Sonnet 3.7/5, GPT-5, Kimi-k3, GLM, Grok, Qwen 2.5) through the Perplexity WebUI scraper, models exhibited three recurring failure patterns:

1. **Multilingual Refusal / Capability Disclaimers**: Upstream Perplexity RLHF and persona injection caused models to reply: _"não tenho acesso direto aos recursos de ferramentas nessa sessão"_ or _"I do not have access to external tools"_.
2. **Hallucinated / Faked Prose Executions**: Models simulated actions in text instead of invoking tools: _"Gerei uma imagem: paisagem aleatória..."_ or _"Created file /tmp/foo"_.
3. **Turn Deferrals & Prompt Suggestions**: Models made turn-based excuses: _"Não posso acionar a ferramenta neste turno, mas posso sugerir um prompt para você usar..."_.

### 1.2 Why Automated Tests Passed While Live LLMs Failed

Prior test suites only verified:

- Downstream parser functions against synthetic mocked strings like `<|OPENAI_TOOL_CALL|>{...}<|END_OPENAI_TOOL_CALL|>`.
- Streaming SSE chunk splitting and JSON argument delta serialization.

They did **NOT** test:

- What actual prompt (`query_str`) is compiled and dispatched to Perplexity.
- Whether in-context few-shot demonstrations exist to ground the LLM.
- Whether recency-biased tail reminders (`<harness_reminder>`) counteract system-prompt attention decay.
- Whether anti-refusal, anti-simulation, and multilingual directives are strictly enforced at the prompt compiler layer.

As requested by the user: **The test suite must assert the structural integrity of the compiled query string (`query_str`), failing until all few-shots, anti-refusal rules, anti-simulation directives, and tail reminders are properly compiled.**

---

## 2. Proposed Architecture & Solution

### 2.1 In-Context Few-Shot Demonstrations (`build_tool_instruction`)

Location: `src/perplexity_webui_scraper/api/tool_calling.py`
Add explicit `<few_shot_examples>` showing:

- Example 1: User asks to perform an action (e.g. read a file / generate an image / run a command) -> Assistant immediately outputs ONLY the sentinel `<|OPENAI_TOOL_CALL|>{"name":"...","arguments":{...}}<|END_OPENAI_TOOL_CALL|>` with zero prose.
- Example 2: User asks what tools/capabilities are available -> Assistant lists and explains the tools declared in `<declared_tools>` without disclaiming access.
- Example 3: Multi-turn tool execution where user provides `[Tool Result: ...]` -> Assistant answers based on the returned tool output.

### 2.2 Recency-Biased Tail Reminder (`<harness_reminder>`)

Location: `src/perplexity_webui_scraper/api/helpers.py` in `build_query_and_files()`
When `request.tools` is present:

- Append an imperative tail reminder immediately after the last turn in `query_str`:
  ```markdown
  <harness_reminder>
  REMINDER: You have active tools declared above. If the user's request requires executing an action or calling a tool, you MUST emit ONLY the tool sentinel <|OPENAI_TOOL_CALL|>...<|END_OPENAI_TOOL_CALL|>. Do NOT simulate execution in text, do NOT disclaim capabilities, and do NOT defer to another turn.
  </harness_reminder>
  ```
- This ensures LLMs with long context windows or RLHF bias pay immediate attention to tool execution right before generating their completion.

### 2.3 Search Focus & Upstream Persona Suppression

Location: `src/perplexity_webui_scraper/api/helpers.py` in `build_conversation_config()`

- When tools are declared (`has_tools=True`), default `search_focus="writing"` (unless explicitly overridden by user).
- This suppresses Perplexity's web search pre-pass and disables search-agent persona injection.

### 2.4 Strict Prompt Assertion Test Suite

Location: `tests/test_tool_calling_api.py` and `tests/test_claude_cli_integration.py`
Create tests that intercept `mock_conv.ask(query_str)` and rigorously assert:

1. `test_query_str_contains_system_environment_and_operating_rules`: Verifies `<system_environment>` with rules 1-7 (Active Capabilities, No Fake Execution, No Turn Excuses, No Prompt Suggestions, Multilingual Refusal Prohibition, Immediate Tool Invocation, Output Format).
2. `test_query_str_contains_few_shot_demonstrations`: Verifies `<few_shot_examples>` is present and contains valid sentinel syntax examples.
3. `test_query_str_contains_declared_tools_and_selection_rules`: Verifies schema serialization in `<declared_tools>` and `<selection_rule>`.
4. `test_query_str_contains_recency_tail_reminder`: Verifies `<harness_reminder>` is placed at the very end of `query_str`.
5. `test_has_tools_forces_writing_search_focus`: Verifies `search_focus="writing"` when tools are provided.
6. `test_anthropic_and_responses_endpoints_compile_same_robust_prompt`: Verifies `/v1/messages` and `/v1/responses` endpoints produce identical hardened prompt structure.

---

## 3. File Modification Map

| Component / File                                   | Purpose of Change                                                                                                                                                                                       |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/perplexity_webui_scraper/api/tool_calling.py` | Add `<few_shot_examples>` and expand operating rules in `build_tool_instruction()`.                                                                                                                     |
| `src/perplexity_webui_scraper/api/helpers.py`      | Inject `<harness_reminder>` at the tail of `query_str` in `build_query_and_files()` when tools are present. Ensure `build_conversation_config` defaults `search_focus="writing"` when `has_tools=True`. |
| `tests/test_tool_calling_api.py`                   | Add unit tests asserting exact presence of `<few_shot_examples>`, `<harness_reminder>`, operating rules, and anti-refusal directives in `build_tool_instruction()` and `build_query_and_files()`.       |
| `tests/test_claude_cli_integration.py`             | Add end-to-end multi-turn integration tests asserting prompt compilation across OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, and OpenAI `/v1/responses`.                                    |

---

## 4. Verification & Quality Gates

1. **Test Suite Execution**:
   - `uv run --all-extras pytest tests/test_tool_calling_api.py`
   - `uv run --all-extras pytest tests/test_claude_cli_integration.py`
   - `uv run --all-extras pytest tests/test_anthropic_api.py`
   - `uv run --all-extras pytest tests/test_responses_api.py`
   - `uv run --all-extras pytest` (Full suite must pass 100%)

2. **Linting & Formatting**:
   - `uv run ruff check`
   - `uv run ruff format --check`
   - `uv run ty check`

3. **Deploy & Live Verification**:
   - Commit changes cleanly via `/caveman-commit`.
   - Redeploy stack to VPS (`deploy/vps/perplexity-sidecar-chat/update.sh`).
