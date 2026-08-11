"""OpenAI-compatible exception mapping for API responses."""

from __future__ import annotations

from fastapi import status

from perplexity_webui_scraper._internal.exceptions import (
    AuthenticationError,
    FileAccessError,
    FileUploadError,
    FileValidationError,
    HTTPError,
    ModelAccessError,
    ModelStatusError,
    PerplexityError,
    RateLimitError,
    ResearchClarifyingQuestionsError,
    ResponseParsingError,
)
from perplexity_webui_scraper.api.schemas.errors import ErrorDetail, ErrorResponse


_ERROR_CODES: tuple[tuple[type[PerplexityError], str], ...] = (
    (AuthenticationError, "authentication_error"),
    (RateLimitError, "rate_limit_error"),
    (ModelAccessError, "model_access_denied"),
    (FileAccessError, "file_access_denied"),
    (ModelStatusError, "model_status_confirmation_required"),
    (FileValidationError, "invalid_file"),
    (FileUploadError, "file_upload_error"),
    (ResearchClarifyingQuestionsError, "clarification_required"),
    (ResponseParsingError, "upstream_response_error"),
    (HTTPError, "upstream_http_error"),
)


def error_response_for(exc: PerplexityError) -> tuple[int, ErrorResponse]:
    """Map library exception to safe OpenAI-compatible status and envelope."""
    return _status_for(exc), ErrorResponse(
        error=ErrorDetail(message=str(exc), type=_type_for(exc), code=_code_for(exc))
    )


def _status_for(exc: PerplexityError) -> int:
    status_by_type: tuple[tuple[type[PerplexityError], int], ...] = (
        (AuthenticationError, status.HTTP_401_UNAUTHORIZED),
        (RateLimitError, status.HTTP_429_TOO_MANY_REQUESTS),
        (ModelAccessError, status.HTTP_403_FORBIDDEN),
        (FileAccessError, status.HTTP_403_FORBIDDEN),
        (ModelStatusError, status.HTTP_400_BAD_REQUEST),
        (ResearchClarifyingQuestionsError, status.HTTP_422_UNPROCESSABLE_CONTENT),
        (FileValidationError, status.HTTP_400_BAD_REQUEST),
    )
    for exception_type, response_status in status_by_type:
        if isinstance(exc, exception_type):
            return response_status
    if isinstance(exc, HTTPError) and exc.status_code is not None and 400 <= exc.status_code < 500:
        return exc.status_code
    return status.HTTP_502_BAD_GATEWAY


def _type_for(exc: PerplexityError) -> str:
    if isinstance(exc, (ModelAccessError, FileAccessError, ModelStatusError, FileValidationError)):
        return "invalid_request_error"
    return "server_error"


def _code_for(exc: PerplexityError) -> str:
    for exception_type, code in _ERROR_CODES:
        if isinstance(exc, exception_type):
            return code
    return "perplexity_error"
