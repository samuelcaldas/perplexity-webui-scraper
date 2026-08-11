# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development & Dependencies

- Install dependencies: `uv sync --all-extras --all-groups`
- Update dependencies: `uv sync --upgrade --all-extras --all-groups && pnpm update`

### Testing

- Run all tests: `uv run --all-extras pytest`
- Run single test file: `uv run --all-extras pytest tests/test_api.py`
- Run single test function: `uv run --all-extras pytest tests/test_api.py -k test_name`

### Linting & Formatting

- Lint all: `uv run ruff check && uv run ty check && pnpm prettier --check . && pnpm taplo lint *.toml && uv run scripts/render_model_docs.py --check`
- Format all: `uv run ruff check --fix && uv run ruff format && pnpm prettier --write . && pnpm taplo format *.toml`
- Update model docs: `uv run scripts/render_model_docs.py`

### Container Operations

- Build container: `podman build -t perplexity-webui-scraper .`
- Run container: `podman run --rm -p 8000:8000 --name perplexity-api perplexity-webui-scraper`

## Architecture & Structure

Package entrypoint: `src/perplexity_webui_scraper`

### Core Modules

- `core/`: Main library operations (`client.py`, `conversation.py`, `parser.py`, `account.py`, `files.py`, `payload.py`). Reverse-engineers Perplexity AI WebUI web sockets / REST requests.
- `http/`: Impersonated HTTP transport using `curl-cffi` (`client.py`, `fingerprint.py`, `resilience.py`) to handle Cloudflare bypass, TLS fingerprinting, and rate limiting.
- `api/`: FastAPI web server exposing OpenAI-compatible endpoints (`app.py`, `launcher.py`, `routes/`, `auth.py`, `conversation_cache.py`).
- `mcp/`: FastMCP integration exposing scraper tools for MCP clients (`server.py`, `tools/`).
- `cli/`: Typer-based terminal CLI (`commands/`, `__main__.py`).
- `models/`: Model registry (`registry.py`, `types.py`) fed by static config in `_static/models.json`.
- `_internal/`: Base exceptions, constants, types, and logging wrappers (`loguru`).
