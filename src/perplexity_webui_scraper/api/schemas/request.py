"""OpenAI-compatible chat completion request schemas with Perplexity extensions."""

from __future__ import annotations

from base64 import b64decode
import binascii
import hashlib
import json
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

from perplexity_webui_scraper._internal.constants import MAX_FILE_SIZE
from perplexity_webui_scraper.api.tool_schema import validate_function_call, validate_function_schema


class ContentPartText(BaseModel):
    """A plain-text content part within a multimodal message.

    Attributes:
        type: Always ``"text"``.
        text: The text content.
    """

    type: Literal["text"]
    text: str


class ContentPartImageUrl(BaseModel):
    """An image content part within a multimodal message.

    Attributes:
        type: Always ``"image_url"``.
        image_url: Dict with a ``"url"`` key containing a base64 data URI.
    """

    type: Literal["image_url"]
    image_url: dict[str, str]

    @model_validator(mode="after")
    def _validate_data_uri(self) -> Self:
        """Validate image URL data URI syntax, MIME, size, and encoding."""
        url = self.image_url.get("url")
        if not url:
            raise ValueError("image_url.url must be nonempty")
        if not url.startswith("data:"):
            raise ValueError("image_url.url must be an image data URI")

        try:
            header, encoded_data = url.split(",", 1)
        except ValueError as error:
            raise ValueError("image_url.url must contain base64 data") from error

        if not header.lower().startswith("data:image/") or not header.lower().endswith(";base64"):
            raise ValueError("image_url.url must contain a base64 image MIME")
        if not encoded_data:
            raise ValueError("image_url.url must contain base64 data")
        if len(encoded_data) > ((MAX_FILE_SIZE + 2) // 3) * 4:
            raise ValueError("image data exceeds size limit")

        try:
            decoded_data = b64decode(encoded_data, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("image_url.url contains invalid base64 data") from error

        if len(decoded_data) > MAX_FILE_SIZE:
            raise ValueError("image data exceeds size limit")

        return self


ContentPart = ContentPartText | ContentPartImageUrl
"""Union of all supported content part types."""


class FunctionDefinition(BaseModel):
    """OpenAI function definition carried by a function tool."""

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool | None = None

    @model_validator(mode="after")
    def _validate_parameters_schema(self) -> Self:
        """Reject function parameter schemas with invalid top-level shape."""
        schema_type = self.parameters.get("type")
        if schema_type is not None and schema_type != "object":
            raise ValueError("parameters schema type must be 'object'")

        properties = self.parameters.get("properties")
        if properties is not None and not isinstance(properties, dict):
            raise ValueError("parameters schema properties must be an object")

        required = self.parameters.get("required")
        if required is not None and (
            not isinstance(required, list) or not all(isinstance(item, str) for item in required)
        ):
            raise ValueError("parameters schema required must be a list of strings")

        validate_function_schema(self.parameters)

        return self


class FunctionTool(BaseModel):
    """OpenAI function tool definition."""

    type: Literal["function"]
    function: FunctionDefinition


class ToolChoiceFunction(BaseModel):
    """Function selector used by OpenAI's explicit tool choice form."""

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class ExplicitToolChoice(BaseModel):
    """OpenAI tool choice selecting one named function."""

    type: Literal["function"]
    function: ToolChoiceFunction


ToolChoice = Literal["none", "auto", "required"] | ExplicitToolChoice
"""Supported OpenAI tool choice values."""


class ToolCallFunction(BaseModel):
    """Function invocation emitted in an assistant tool call."""

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    arguments: str

    @model_validator(mode="after")
    def _validate_json_arguments(self) -> Self:
        """Require arguments to be a standards-compliant JSON object string."""
        try:
            parsed_arguments = json.loads(self.arguments, parse_constant=self._reject_non_json_constant)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("arguments must contain valid JSON") from error

        if not isinstance(parsed_arguments, dict):
            raise ValueError("arguments must contain a JSON object")  # noqa: TRY004

        return self

    @staticmethod
    def _reject_non_json_constant(value: str) -> None:
        raise ValueError(f"non-JSON constant: {value}")


class AssistantToolCall(BaseModel):
    """Function tool call emitted by an assistant message."""

    id: str = Field(min_length=1)
    type: Literal["function"]
    function: ToolCallFunction


class ChatMessage(BaseModel):
    """A single text, multimodal, assistant-tool, or tool-result message.

    Attributes:
        role: Message author: ``"system"``, ``"developer"``, ``"user"``,
            ``"assistant"``, or ``"tool"``.
        content: Either plain text, multimodal parts, or ``None`` for an
            assistant message containing tool calls.
        tool_calls: Function calls emitted by an assistant message.
        tool_call_id: Assistant tool-call ID referenced by a tool result.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    name: str | None = None
    refusal: str | None = None
    function_call: ToolCallFunction | None = None
    tool_calls: list[AssistantToolCall] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _validate_role_fields(self) -> Self:
        """Validate role-specific content and tool-call relationships."""
        if self.role == "assistant":
            if self.tool_call_id is not None:
                raise ValueError("assistant messages cannot contain tool_call_id")
            if self.content is None and not self.tool_calls and self.function_call is None and self.refusal is None:
                raise ValueError("assistant content may be null only with a call or refusal")
            return self

        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("tool messages require tool_call_id")
            if self.content is None:
                raise ValueError("tool messages require content")
            if self.tool_calls is not None:
                raise ValueError("tool messages cannot contain tool_calls")
            return self

        if self.content is None:
            raise ValueError(f"{self.role} messages require content")
        if self.tool_calls is not None:
            raise ValueError(f"{self.role} messages cannot contain tool_calls")
        if self.tool_call_id is not None:
            raise ValueError(f"{self.role} messages cannot contain tool_call_id")
        return self

    def text(self) -> str:
        """Return the plain-text portion of this message.

        For multimodal messages, concatenates all ``ContentPartText`` blocks
        with newlines.

        Returns:
            Plain-text string.
        """
        if isinstance(self.content, str):
            return self.content
        if not isinstance(self.content, list):
            return ""

        return "\n".join(p.text for p in self.content if isinstance(p, ContentPartText))

    def effective_tool_calls(self) -> tuple[AssistantToolCall, ...]:
        """Return modern calls or a deterministic legacy function-call equivalent."""
        if self.tool_calls:
            return tuple(self.tool_calls)
        if self.function_call is None:
            return ()

        canonical = json.dumps(
            {"arguments": self.function_call.arguments, "name": self.function_call.name},
            separators=(",", ":"),
            sort_keys=True,
        )
        call_id = f"call_legacy_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]}"
        return (
            AssistantToolCall(
                id=call_id,
                type="function",
                function=self.function_call,
            ),
        )

    def image_bytes(self) -> list[tuple[bytes, str, str]]:
        """Return decoded base64 image parts as ``(data, filename, mimetype)`` tuples.

        Validated image data URIs are decoded server-side.

        Returns:
            List of ``(bytes, filename, mimetype)`` tuples.
        """
        if not isinstance(self.content, list):
            return []

        results: list[tuple[bytes, str, str]] = []

        for part in self.content:
            if not isinstance(part, ContentPartImageUrl):
                continue

            url = part.image_url["url"]
            header, b64data = url.split(",", 1)
            mimetype = header.split(":", 1)[1].split(";", 1)[0]
            ext = mimetype.split("/", 1)[1].split("+", 1)[0]
            filename = f"image.{ext}"
            data = b64decode(b64data, validate=True)
            results.append((data, filename, mimetype))

        return results


class CoordinatesInput(BaseModel):
    """Latitude/longitude pair for the ``perplexity.coordinates`` field.

    Attributes:
        latitude: Latitude in decimal degrees (-90 to +90).
        longitude: Longitude in decimal degrees (-180 to +180).
    """

    latitude: FiniteFloat = Field(ge=-90, le=90)
    longitude: FiniteFloat = Field(ge=-180, le=180)


class PerplexityExtensions(BaseModel):
    """Perplexity-specific configuration passed under the ``perplexity`` key.

    All fields are optional; omitted fields fall back to server defaults.
    Pass this block inside ``ChatCompletionRequest.perplexity``.

    Attributes:
        citation_mode: Citation rendering: ``"clean"`` (default), ``"markdown"``,
            or ``"default"`` (keep markers as-is).
        search_focus: ``"web"`` (default) enables sources; ``"writing"`` disables
            them for purely generative responses.
        source_focus: Source category filter.  Accepts a single value or list:
            ``"web"``, ``"academic"``, ``"social"``, ``"finance"``, ``"all"``.
        time_range: Recency filter: ``"all"``, ``"day"``, ``"week"``,
            ``"month"``, or ``"year"``.
        save_to_library: Save conversation to Perplexity library.
        language: BCP-47 language tag (e.g. ``"pt-BR"``).
        timezone: IANA timezone string (e.g. ``"America/Sao_Paulo"``).
        coordinates: Geographic coordinates for localised results.
        space_uuid: UUID of a Perplexity Space to post into.  Use DevTools to
            obtain it from the ``target_collection_uuid`` field of a
            ``perplexity_ask`` request.  The URL slug is **not** the UUID.
        thread_uuid: UUID of an existing conversation thread to continue.
            When provided, the server reuses the cached ``Conversation``
            and sends only the last user message as a follow-up.
        response_format: Hint for the response format.  ``"text"`` (default)
            returns plain text; ``"json_object"`` adds a JSON-output instruction
            to the system prompt.  Note: Perplexity has no native structured
            output support — this is a best-effort prompt injection.
        allow_risky_model: Explicitly acknowledge any non-available model status.
        custom_model_mode: Backend mode used with ``model="custom:<identifier>"``.
    """

    model_config = ConfigDict(extra="ignore")

    citation_mode: Literal["default", "markdown", "clean"] | None = None
    search_focus: Literal["web", "writing"] | None = None
    source_focus: (
        Literal["web", "academic", "social", "finance", "all"]
        | list[Literal["web", "academic", "social", "finance", "all"]]
        | None
    ) = None
    time_range: Literal["all", "day", "week", "month", "year"] | None = None
    save_to_library: bool = False
    language: str | None = None
    timezone: str | None = None
    coordinates: CoordinatesInput | None = None
    space_uuid: str | None = None
    thread_uuid: str | None = None
    response_format: Literal["text", "json_object"] = "text"
    allow_risky_model: bool = False
    custom_model_mode: Literal["copilot", "search", "research"] = "copilot"

    @field_validator("space_uuid", "thread_uuid")
    @classmethod
    def _validate_uuid(cls, value: str | None) -> str | None:
        """Require UUID strings for Perplexity conversation identifiers."""
        if value is None:
            return None

        try:
            UUID(value)
        except (ValueError, AttributeError, TypeError) as error:
            raise ValueError("must be a valid UUID") from error

        return value

    @model_validator(mode="before")
    @classmethod
    def _normalise_strings(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Lowercase all string-based enum fields for case-insensitive matching.

        Args:
            values: Raw input dict before field assignment.

        Returns:
            Normalised dict with lowercased enum values.
        """
        if not isinstance(values, dict):
            return values

        for key in ("citation_mode", "search_focus", "time_range", "response_format", "custom_model_mode"):
            if isinstance(values.get(key), str):
                values[key] = values[key].lower()

        if isinstance(values.get("source_focus"), str):
            values["source_focus"] = values["source_focus"].lower()
        elif isinstance(values.get("source_focus"), list):
            values["source_focus"] = [s.lower() if isinstance(s, str) else s for s in values["source_focus"]]

        return values


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request.

    Standard OpenAI fields that Perplexity does not support (``temperature``,
    ``top_p``, ``n``, ``max_tokens``, …) are accepted for drop-in client
    compatibility but silently ignored via ``extra="allow"``.

    The optional ``perplexity`` block exposes all Perplexity-specific settings::

        {
            "model": "perplexity/best",
            "messages": [...],
            "perplexity": {
                "citation_mode": "clean",
                "search_focus": "web",
                "source_focus": ["web", "academic"],
                "time_range": "week",
                "save_to_library": false,
                "language": "pt-BR",
                "timezone": "America/Sao_Paulo",
                "coordinates": {"latitude": -23.5, "longitude": -46.6},
                "response_format": "json_object",
            },
        }

    Attributes:
        model: Model ID (e.g. ``"perplexity/best"``).
        messages: Conversation messages.
        stream: Enable SSE streaming.
        perplexity: Optional Perplexity-specific configuration block.
    """

    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    perplexity: PerplexityExtensions | None = None
    tools: list[FunctionTool] | None = None
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        """Reject blank model identifiers without changing valid values."""
        if not value.strip():
            raise ValueError("model must be nonempty")

        return value

    @model_validator(mode="after")
    def _validate_tool_call_relationships(self) -> Self:
        """Validate tool choices, declared calls, and contiguous result groups."""
        self._validate_tool_request_options()
        self._validate_message_calls()
        return self

    def _validate_tool_request_options(self) -> None:
        """Reject unsupported stream/tool combinations and undeclared choices."""
        if self.stream and self.tools:
            raise ValueError("streaming tool calls are not supported")

        if self.tool_choice == "required" and not self.tools:
            raise ValueError("tool_choice='required' requires declared tools")

        if isinstance(self.tool_choice, ExplicitToolChoice):
            declared_names = {tool.function.name for tool in self.tools or []}
            if self.tool_choice.function.name not in declared_names:
                raise ValueError("tool_choice function must be declared in tools")

        if self.tools is not None:
            names = [tool.function.name for tool in self.tools]
            if len(names) != len(set(names)):
                raise ValueError("tools must not declare duplicate function names")

    def _validate_message_calls(self) -> None:
        """Validate assistant calls and require contiguous result groups."""
        seen_tool_call_ids: set[str] = set()
        message_index = 0

        while message_index < len(self.messages):
            message = self.messages[message_index]
            if message.role == "tool":
                raise ValueError(f"orphan tool_call_id: {message.tool_call_id}")

            if message.role != "assistant":
                message_index += 1
                continue

            self._validate_legacy_function_call(message)
            effective_calls = message.effective_tool_calls()
            has_legacy_result = bool(
                message.function_call is not None
                and not message.tool_calls
                and message_index + 1 < len(self.messages)
                and self.messages[message_index + 1].role == "tool"
            )
            if not message.tool_calls and not has_legacy_result:
                message_index += 1
                continue

            expected_ids = [tool_call.id for tool_call in effective_calls]
            if len(expected_ids) != len(set(expected_ids)) or seen_tool_call_ids.intersection(expected_ids):
                raise ValueError("duplicate tool call id")
            seen_tool_call_ids.update(expected_ids)
            self._validate_assistant_tool_calls(effective_calls)

            for offset, expected_id in enumerate(expected_ids, start=1):
                result_index = message_index + offset
                if result_index >= len(self.messages):
                    raise ValueError(f"orphan tool call: missing result for tool_call_id {expected_id}")
                result = self.messages[result_index]
                if result.role != "tool" or result.tool_call_id != expected_id:
                    raise ValueError(f"skipped tool-call group: expected result for tool_call_id {expected_id}")

            message_index += len(expected_ids) + 1

    def _validate_legacy_function_call(self, message: ChatMessage) -> None:
        """Validate legacy assistant function calls when tools are declared."""
        if message.function_call is None:
            return
        validate_function_call(
            message.function_call.name,
            message.function_call.arguments,
            self.tools,
            "assistant function call",
        )

    def _validate_assistant_tool_calls(self, tool_calls: tuple[AssistantToolCall, ...]) -> None:
        """Validate effective assistant tool calls against current declarations."""
        for tool_call in tool_calls:
            validate_function_call(
                tool_call.function.name,
                tool_call.function.arguments,
                self.tools,
                "assistant tool call",
            )
