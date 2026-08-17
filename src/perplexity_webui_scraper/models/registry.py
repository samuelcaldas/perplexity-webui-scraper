"""Model registry — loads all AI model definitions from the static data file."""

from __future__ import annotations

from importlib.resources import files
from re import fullmatch, sub
from warnings import warn

from orjson import loads

from perplexity_webui_scraper._internal.exceptions import ModelRiskWarning, ModelStatusError
from perplexity_webui_scraper.models.types import MODEL_STATUS_DESCRIPTIONS, Model, ModelMode


_CUSTOM_PREFIX = "custom:"
_CUSTOM_IDENTIFIER_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}"


class ModelRegistry:
    """Registry of all available Perplexity AI models.

    The registry is populated at instantiation time by reading ``models.json``
    from the ``_static`` package directory via ``importlib.resources``.  The
    singleton ``MODELS`` instance is created at module import time.

    Usage::

        from perplexity_webui_scraper.models import MODELS

        model = MODELS.resolve("perplexity/best")
        all_models = MODELS.list_all()
    """

    _models: dict[str, Model]
    _aliases: dict[str, Model]

    def __init__(self, raw_models: list[dict[str, object]] | None = None) -> None:
        """Load models from the bundled ``models.json`` static asset."""
        self._models, self._aliases = self._load(raw_models if raw_models is not None else self._read_static_models())

    @staticmethod
    def _read_static_models() -> list[dict[str, object]]:
        """Read and deserialize the bundled static model registry."""
        static_pkg = files("perplexity_webui_scraper._static")
        models_file = static_pkg.joinpath("models.json")

        raw: bytes = models_file.read_bytes()  # type: ignore[arg-type]
        data = loads(raw)

        if not isinstance(data, list):
            raise TypeError("models.json must contain a list of model definitions")

        if not all(isinstance(item, dict) for item in data):
            raise TypeError("models.json must contain only object entries")

        return data

    @staticmethod
    def _load(raw_models: list[dict[str, object]]) -> tuple[dict[str, Model], dict[str, Model]]:
        """Validate raw model data and return model and alias mappings."""
        models: dict[str, Model] = {}
        aliases: dict[str, Model] = {}
        tool_names: set[str] = set()

        for item in raw_models:
            model = Model.model_validate(item)

            if model.id in models:
                raise ValueError(f"Duplicate model id in models.json: {model.id!r}")

            if model.tool_name in tool_names:
                raise ValueError(f"Duplicate MCP tool name in models.json: {model.tool_name!r}")

            models[model.id] = model
            tool_names.add(model.tool_name)

            # Auto-register bare model id without provider prefix as alias (e.g. 'best', 'claude-sonnet-5')
            if "/" in model.id:
                bare_id = model.id.split("/", 1)[1]
                aliases.setdefault(bare_id, model)

            for alias in model.aliases:
                aliases[alias] = model
                if "/" in alias:
                    aliases.setdefault(alias.split("/", 1)[1], model)

        return models, aliases

    def is_registered(self, model_id: str) -> bool:
        """Return whether model_id is present in registered models or aliases."""
        clean_id = sub(r"\[.*?\]", "", model_id).strip()
        return (
            model_id in self._models
            or model_id in self._aliases
            or clean_id in self._models
            or clean_id in self._aliases
        )

    def resolve(self, model_id: str, *, fallback_dynamic: bool = False) -> Model:
        """Look up a model by its canonical string ID or alias.

        Args:
            model_id: The model identifier, e.g. ``"perplexity/best"``.
            fallback_dynamic: Whether to create a dynamic model if not found.

        Returns:
            The matching :class:`Model` instance.

        Raises:
            ValueError: If ``model_id`` is not registered and fallback_dynamic is False.
        """
        clean_id = sub(r"\[.*?\]", "", model_id).strip()

        for candidate in (model_id, clean_id):
            if candidate in self._models:
                return self._models[candidate]
            if candidate in self._aliases:
                return self._aliases[candidate]

        if fallback_dynamic:
            provider = model_id.split("/", 1)[0] if "/" in model_id else "custom"
            identifier = model_id.split("/", 1)[1] if "/" in model_id else model_id
            return Model(
                id=model_id,
                name=f"Dynamic model ({model_id})",
                description="Dynamically resolved unlisted model identifier.",
                identifier=identifier,
                tool_name=f"pplx_dynamic_{identifier.replace('-', '_').replace('.', '_')}"[:64],
                provider=provider,
                min_tier=None,
                mode="copilot",
                status="unknown",
            )

        available = ", ".join(f'"{m}"' for m in self._models)
        raise ValueError(f"Unknown model {model_id!r}. Available models: {available}")

    def list_all(self, *, only_available: bool = False) -> list[Model]:
        """Return all registered :class:`Model` instances in definition order.

        Args:
            only_available: If True, returns only models with status == "available".

        Returns:
            List of models loaded from ``models.json``.
        """
        if only_available:
            return [m for m in self._models.values() if m.status == "available"]
        return list(self._models.values())

    def resolve_for_use(
        self,
        model_id: str,
        *,
        allow_risky_model: bool = False,
        custom_model_mode: ModelMode = "copilot",
        allow_unregistered: bool = False,
        reasoning_effort: str | None = None,
        thinking: bool | None = None,
    ) -> Model:
        """Resolve a model and enforce explicit acknowledgement of risky states."""
        if model_id.startswith(_CUSTOM_PREFIX):
            identifier = model_id.removeprefix(_CUSTOM_PREFIX)
            if not fullmatch(_CUSTOM_IDENTIFIER_PATTERN, identifier):
                raise ValueError(
                    "Custom model identifiers must contain 1-128 letters, digits, dots, colons, underscores, or hyphens"
                )

            model = Model(
                id=model_id,
                name=f"Custom model ({identifier})",
                description="User-supplied Perplexity internal model identifier.",
                identifier=identifier,
                tool_name="pplx_custom",
                provider="custom",
                min_tier=None,
                mode=custom_model_mode,
                status="unknown",
            )
        else:
            model = self.resolve(model_id, fallback_dynamic=allow_unregistered)

        # Handle thinking / reasoning_effort resolution
        wants_thinking = (
            thinking is True
            or (reasoning_effort is not None and reasoning_effort.lower() in {"low", "medium", "high"})
            or (model_id.endswith("-thinking") and model.thinking_identifier is not None)
        )
        if wants_thinking and model.thinking_identifier:
            model = model.model_copy(update={"identifier": model.thinking_identifier})

        if model.status != "available" and not allow_risky_model:
            raise ModelStatusError(model.id, model.status, MODEL_STATUS_DESCRIPTIONS[model.status])

        if model.status != "available":
            warn(MODEL_STATUS_DESCRIPTIONS[model.status], ModelRiskWarning, stacklevel=2)

        return model


#: Singleton registry.  Import and use this directly.
#: ``MODELS.resolve("model-id")`` or ``MODELS.list_all()``.
MODELS: ModelRegistry = ModelRegistry()
