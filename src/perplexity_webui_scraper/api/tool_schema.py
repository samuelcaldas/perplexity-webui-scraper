"""Validation helpers for OpenAI function argument JSON Schemas."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from jsonschema import Draft7Validator, SchemaError
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT7


if TYPE_CHECKING:
    from collections.abc import Iterator

    from perplexity_webui_scraper.api.schemas.request import FunctionTool


def validate_function_schema(parameters: dict[str, Any]) -> None:
    """Reject malformed or unresolvable JSON Schema definitions before provider work."""
    try:
        Draft7Validator.check_schema(parameters)
    except SchemaError as error:
        raise ValueError(f"invalid function parameters schema: {error.message}") from error

    resource = Resource.from_contents(parameters, default_specification=DRAFT7)
    resolver = Registry().resolver_with_root(resource)
    for reference in _iter_references(parameters):
        try:
            resolver.lookup(reference)
        except Unresolvable as error:
            raise ValueError(f"invalid function parameters schema reference: {reference!r}") from error


def _iter_references(value: Any) -> Iterator[str]:
    """Yield every JSON Schema reference in a nested schema value."""
    if isinstance(value, dict):
        reference = value.get("$ref")
        if isinstance(reference, str):
            yield reference
        for child in value.values():
            yield from _iter_references(child)
        return

    if isinstance(value, list):
        for child in value:
            yield from _iter_references(child)


def validate_function_call(
    name: str,
    arguments: str,
    tools: list[FunctionTool] | None,
    context: str,
) -> None:
    """Validate one assistant/provider call against declared function tools."""
    if tools is None:
        return

    tool = next((candidate for candidate in tools if candidate.function.name == name), None)
    if tool is None:
        raise ValueError(f"{context} function {name!r} must be declared in tools")

    parsed_arguments = _parse_arguments(arguments, context)
    try:
        errors = sorted(Draft7Validator(tool.function.parameters).iter_errors(parsed_arguments), key=str)
    except (SchemaError, Unresolvable) as error:
        raise ValueError(f"{context} function schema is invalid: {error}") from error

    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path)
        location = f" at {path}" if path else ""
        raise ValueError(f"{context} arguments do not match function schema{location}: {error.message}")


def _parse_arguments(arguments: str, context: str) -> dict[str, Any]:
    """Parse one strict JSON object used as function arguments."""
    try:
        parsed = json.loads(arguments, parse_constant=_reject_non_json_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} arguments must contain valid JSON") from error

    if not isinstance(parsed, dict):
        raise TypeError(f"{context} arguments must contain a JSON object")

    return parsed


def _reject_non_json_constant(value: str) -> None:
    """Reject non-standard JSON constants such as NaN and Infinity."""
    raise ValueError(f"non-JSON constant: {value}")
