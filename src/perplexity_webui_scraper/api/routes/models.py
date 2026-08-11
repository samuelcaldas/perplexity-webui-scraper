"""GET /v1/models route."""

from __future__ import annotations

from json import JSONDecodeError
from typing import TYPE_CHECKING, Annotated

from anyio import to_thread
from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from perplexity_webui_scraper._internal.exceptions import AuthenticationError, ModelAccessError, PerplexityError
from perplexity_webui_scraper.api.auth import client_pool, extract_token
from perplexity_webui_scraper.api.schemas.response import ModelCatalogMetadata, ModelList, ModelObject
from perplexity_webui_scraper.core.account import AccountSession, ensure_model_access
from perplexity_webui_scraper.models.registry import MODELS


if TYPE_CHECKING:
    from perplexity_webui_scraper.models.types import Model


router = APIRouter()


@router.get("/v1/models", response_model=None)
async def list_models(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> JSONResponse:
    """List static models, filtering authenticated results by account entitlement."""
    models = MODELS.list_all()

    if authorization is not None:
        token = extract_token(authorization)
        request_lock = client_pool.get_request_lock(token)
        lock_acquired = False
        try:
            await request_lock.acquire()
            lock_acquired = True
            models = await to_thread.run_sync(_authenticated_models, models, token)
        finally:
            if lock_acquired:
                request_lock.release()
            client_pool.release_request_lock(token, request_lock)

    data = ModelList(data=[_model_object(model) for model in models])
    return JSONResponse(content=data.model_dump(mode="json"))


def _authenticated_models(models: list[Model], token: str) -> list[Model]:
    """Return available models allowed by authenticated account tier."""
    client = client_pool.get_or_create(token)

    try:
        profile = client.get_account_profile()
        session = AccountSession.model_validate({"user": {"subscription_tier": profile.account_tier}})
    except AuthenticationError:
        client_pool.discard(token, client)
        raise
    except (PerplexityError, ValidationError, JSONDecodeError):
        client_pool.discard(token, client)
        return models

    return [model for model in models if _is_available_to_account(model, session)]


def _is_available_to_account(model: Model, session: AccountSession) -> bool:
    """Return whether model is static-available and account-entitled."""
    if model.status != "available":
        return False

    try:
        ensure_model_access(session, model)
    except ModelAccessError:
        return False

    return True


def _model_object(model: Model) -> ModelObject:
    """Convert static model metadata into OpenAI catalog shape."""
    return ModelObject(
        id=model.id,
        owned_by=model.provider,
        perplexity=ModelCatalogMetadata(
            min_tier=model.min_tier,
            is_official=model.is_official,
            status=model.status,
            last_tested_at=model.last_tested_at,
        ),
    )
