from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient
from openai import OpenAI
from pytest import fixture

from perplexity_webui_scraper.api.app import app


if TYPE_CHECKING:
    from collections.abc import Iterator


@fixture
def openai_client() -> Iterator[OpenAI]:
    """Yield an OpenAI SDK client backed by the in-process FastAPI app."""
    with TestClient(app, raise_server_exceptions=False) as test_client:
        sdk_client = OpenAI(
            api_key="sentinel",
            base_url="http://testserver/v1",
            http_client=test_client,
            max_retries=0,
        )
        try:
            yield sdk_client
        finally:
            sdk_client.close()
