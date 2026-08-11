"""Emulated OpenAI tool-call protocol for the non-streaming API path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

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
    """Parse one exact, provider-emitted sentinel without invoking any tool."""
    if not answer or build_tool_instruction(tools, tool_choice) is None:
        return None

    candidate = answer.strip()

    if not candidate.startswith(TOOL_CALL_SENTINEL_START) or not candidate.endswith(TOOL_CALL_SENTINEL_END):
        return None

    payload_text = candidate[len(TOOL_CALL_SENTINEL_START) : -len(TOOL_CALL_SENTINEL_END)].strip()
    payload = _parse_json_object(payload_text)
    validated_payload = _validated_payload(payload, tools, tool_choice)

    if validated_payload is None:
        return None

    function_name, arguments, candidate_id = validated_payload
    call_id = _validated_or_generated_call_id(candidate_id, function_name, arguments)

    if call_id is None:
        return None

    return EmulatedToolCall(call_id=call_id, function_name=function_name, arguments=arguments)


def _validated_payload(
    payload: dict[str, Any] | None,
    tools: list[FunctionTool] | None,
    tool_choice: ToolChoice | None,
) -> tuple[str, dict[str, Any], object] | None:
    """Validate sentinel fields against declared functions and tool choice."""
    if payload is None or set(payload) - {"id", "name", "arguments"}:
        return None

    function_name = payload.get("name")
    arguments = payload.get("arguments")

    if not isinstance(function_name, str) or not isinstance(arguments, dict):
        return None

    declared_names = {tool.function.name for tool in tools or []}

    if function_name not in declared_names or not _matches_tool_choice(function_name, tool_choice):
        return None

    try:
        validate_function_call(
            function_name,
            serialize_tool_arguments(arguments),
            tools,
            "provider tool call",
        )
    except ValueError:
        return None

    return function_name, arguments, payload.get("id")


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

    return f"You MUST select only function `{tool_choice.function.name}`. Otherwise answer normally."


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

    if not isinstance(payload, dict):
        return None

    return payload


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
