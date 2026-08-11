"""Emulated OpenAI tool-call protocol for non-streaming API requests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

from perplexity_webui_scraper._internal.exceptions import ToolProtocolError
from perplexity_webui_scraper.api.tool_schema import validate_function_call


if TYPE_CHECKING:
    from perplexity_webui_scraper.api.schemas.request import FunctionTool, ToolChoice


TOOL_CALL_SENTINEL_START = "<|OPENAI_TOOL_CALL|>"
TOOL_CALL_SENTINEL_END = "<|END_OPENAI_TOOL_CALL|>"
_CALL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True, slots=True)
class EmulatedToolCall:
    """Validated tool call parsed from one complete provider sentinel."""

    call_id: str
    function_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ProtocolFailure:
    """Classify invalid provider output without exposing it as executable data."""

    code: str
    message: str


def requires_tool_call(tools: list[FunctionTool] | None, tool_choice: ToolChoice | None) -> bool:
    """Return whether current request requires one valid emulated call."""
    if not tools:
        return False
    return tool_choice == "required" or hasattr(tool_choice, "function")


def build_tool_instruction(tools: list[FunctionTool] | None, tool_choice: ToolChoice | None) -> str | None:
    """Build deterministic provider instructions for emulated tool calls."""
    if not tools or tool_choice == "none":
        return None

    function_specs = [_function_spec(tool) for tool in tools]
    names = [tool.function.name for tool in tools]
    selection = _selection_instruction(names, tool_choice)
    serialized_specs = json.dumps(function_specs, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    return (
        "[OPENAI_TOOL_INSTRUCTIONS]\n"
        f"Declared functions: {serialized_specs}\n"
        f"{selection}\n"
        "If selecting a function, emit exactly one complete sentinel and no other text:\n"
        f'{TOOL_CALL_SENTINEL_START}{{"arguments":{{}},"name":"function_name"}}{TOOL_CALL_SENTINEL_END}\n'
        "Arguments must be a JSON object matching the selected function. "
        "Do not emit a sentinel for an undeclared function.\n"
        "[/OPENAI_TOOL_INSTRUCTIONS]"
    )


def serialize_tool_arguments(arguments: dict[str, Any]) -> str:
    """Serialize tool arguments in stable compact JSON form."""
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def parse_emulated_tool_call(
    answer: str | None,
    tools: list[FunctionTool] | None,
    tool_choice: ToolChoice | None,
) -> EmulatedToolCall | None:
    """Parse one provider sentinel without invoking a caller-declared function.

    Optional tool selection preserves invalid output as ordinary assistant text.
    Required or named selection raises a typed error with a stable protocol code.
    """
    if build_tool_instruction(tools, tool_choice) is None:
        return None

    payload_text, failure = _extract_payload(answer)
    if failure is not None:
        return _raise_or_ignore(failure, tools, tool_choice)

    payload = _parse_json_object(payload_text or "")
    if payload is None:
        return _raise_or_ignore(_malformed_failure(), tools, tool_choice)

    validated_payload, failure = _validated_payload(payload, tools, tool_choice)
    if failure is not None:
        return _raise_or_ignore(failure, tools, tool_choice)
    assert validated_payload is not None

    function_name, arguments, candidate_id = validated_payload
    call_id = _validated_or_generated_call_id(candidate_id, function_name, arguments)
    if call_id is None:
        return _raise_or_ignore(_malformed_failure(), tools, tool_choice)

    return EmulatedToolCall(call_id=call_id, function_name=function_name, arguments=arguments)


def _raise_or_ignore(
    failure: _ProtocolFailure,
    tools: list[FunctionTool] | None,
    tool_choice: ToolChoice | None,
) -> None:
    """Raise strict-choice failure or retain optional provider output as text."""
    if requires_tool_call(tools, tool_choice):
        raise ToolProtocolError(failure.code, failure.message)


def _extract_payload(answer: str | None) -> tuple[str | None, _ProtocolFailure | None]:
    """Extract one complete sentinel payload while rejecting ambiguous framing."""
    text = answer or ""
    start_count = text.count(TOOL_CALL_SENTINEL_START)
    end_count = text.count(TOOL_CALL_SENTINEL_END)

    if start_count == 0 and end_count == 0:
        return None, _ProtocolFailure(
            "tool_call_missing",
            "Provider response did not contain a valid required tool call.",
        )
    if start_count != 1 or end_count != 1:
        return None, _malformed_failure()

    start = text.find(TOOL_CALL_SENTINEL_START)
    end = text.find(TOOL_CALL_SENTINEL_END)
    if end < start:
        return None, _malformed_failure()

    payload_start = start + len(TOOL_CALL_SENTINEL_START)
    return text[payload_start:end].strip(), None


def _validated_payload(
    payload: dict[str, Any],
    tools: list[FunctionTool] | None,
    tool_choice: ToolChoice | None,
) -> tuple[tuple[str, dict[str, Any], object] | None, _ProtocolFailure | None]:
    """Validate sentinel fields against declared functions and requested selection."""
    if set(payload) - {"id", "name", "arguments"}:
        return None, _malformed_failure()

    function_name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(function_name, str) or not isinstance(arguments, dict):
        return None, _malformed_failure()

    declared_names = {tool.function.name for tool in tools or []}
    if function_name not in declared_names:
        return None, _malformed_failure()
    if not _matches_tool_choice(function_name, tool_choice):
        return None, _ProtocolFailure(
            "tool_call_choice_mismatch",
            "Provider tool call does not match requested function.",
        )

    try:
        validate_function_call(
            function_name,
            serialize_tool_arguments(arguments),
            tools,
            "provider tool call",
        )
    except ValueError:
        return None, _ProtocolFailure(
            "tool_call_arguments_invalid",
            "Provider response did not contain a valid required tool call: "
            "arguments do not match declared function schema.",
        )

    return (function_name, arguments, payload.get("id")), None


def _malformed_failure() -> _ProtocolFailure:
    """Return stable error details for malformed provider framing or payloads."""
    return _ProtocolFailure(
        "tool_call_malformed",
        "Provider response did not contain a valid required tool call: malformed tool call.",
    )


def _function_spec(tool: FunctionTool) -> dict[str, Any]:
    """Return stable JSON-compatible function metadata."""
    function = tool.function
    spec: dict[str, Any] = {"name": function.name, "parameters": function.parameters}

    if function.description is not None:
        spec["description"] = function.description
    if function.strict is not None:
        spec["strict"] = function.strict

    return spec


def _selection_instruction(names: list[str], tool_choice: ToolChoice | None) -> str:
    """Describe allowed function selection for current tool-choice mode."""
    if tool_choice == "required":
        return f"You MUST select exactly one function from: {', '.join(names)}."
    if isinstance(tool_choice, str) or tool_choice is None:
        return f"You MAY select at most one function from: {', '.join(names)}. Otherwise answer normally."
    return f"You MUST select exactly function `{tool_choice.function.name}`."


def _matches_tool_choice(function_name: str, tool_choice: ToolChoice | None) -> bool:
    """Return whether parsed function satisfies explicit selection constraints."""
    if isinstance(tool_choice, str) or tool_choice is None:
        return True
    return function_name == tool_choice.function.name


def _parse_json_object(payload_text: str) -> dict[str, Any] | None:
    """Parse strict JSON object payload, rejecting non-standard constants."""
    try:
        payload = json.loads(payload_text, parse_constant=_reject_non_json_constant)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _reject_non_json_constant(value: str) -> None:
    """Reject NaN and Infinity from provider payloads."""
    raise ValueError(f"non-JSON constant: {value}")


def _validated_or_generated_call_id(
    candidate: object,
    function_name: str,
    arguments: dict[str, Any],
) -> str | None:
    """Validate provider ID or derive stable ID from canonical call content."""
    if candidate is not None:
        if not isinstance(candidate, str) or _CALL_ID_PATTERN.fullmatch(candidate) is None:
            return None
        return candidate

    canonical_call = json.dumps(
        {"arguments": arguments, "name": function_name},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical_call.encode("utf-8")).hexdigest()[:24]
    return f"call_{digest}"
