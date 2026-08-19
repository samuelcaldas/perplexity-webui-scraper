# MCP Server (Model Context Protocol)

The library includes an MCP server that exposes every model as a separate tool for AI assistants like Claude Desktop and Antigravity. Enable only the models you need to keep agent context size small.

## Configuration

Add to your MCP config file (no installation required via npm, handled by python `uvx` native tools):

**Claude Desktop** (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "perplexity-webui-scraper": {
      "command": "uvx",
      "args": [
        "--from",
        "perplexity-webui-scraper[mcp]@latest",
        "perplexity-webui-scraper",
        "mcp"
      ],
      "env": {
        "PERPLEXITY_SESSION_TOKEN": "your_token_here"
      }
    }
  }
}
```

**From GitHub prod branch:**

```json
{
  "mcpServers": {
    "perplexity-webui-scraper": {
      "command": "uvx",
      "args": [
        "--from",
        "perplexity-webui-scraper[mcp]@git+https://github.com/henrique-coder/perplexity-webui-scraper.git@prod",
        "perplexity-webui-scraper",
        "mcp"
      ],
      "env": {
        "PERPLEXITY_SESSION_TOKEN": "your_token_here"
      }
    }
  }
}
```

**From local directory (for development):**

```json
{
  "mcpServers": {
    "perplexity-webui-scraper": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/perplexity-webui-scraper",
        "run",
        "perplexity-webui-scraper",
        "mcp"
      ],
      "env": {
        "PERPLEXITY_SESSION_TOKEN": "your_token_here"
      }
    }
  }
}
```

## Optional Podman Image

For containerized stdio setups only:

```bash
# Pull published MCP image
podman pull ghcr.io/henrique-coder/perplexity-webui-scraper:mcp

# Run MCP server (requires token)
podman run --rm -it -e PERPLEXITY_SESSION_TOKEN=your_token ghcr.io/henrique-coder/perplexity-webui-scraper:mcp
```

This is niche. Prefer `uvx` for normal MCP client setups.

## Available Tools

Each tool uses a specific AI model. Enable only the ones you need:

Tools marked `[AVAILABLE]` can be called normally. `[UNKNOWN]` and `[UNAVAILABLE]` tools require `allow_risky_model=true`. Official listing is exposed separately as `is_official`; it does not imply that a tool has been tested. The generic `pplx_custom` tool accepts an internal identifier, which always starts with `unknown` status and `is_official=false`.

<!-- BEGIN GENERATED MODEL CATALOG -->
### Status reference

| Status | Meaning | Runtime behavior |
| --- | --- | --- |
| `available` | Confirmed to work normally. | Normal use; the local minimum-tier check applies. |
| `unknown` | Current availability has not been confirmed. | Requires `allow_risky_model`; this is the default for unverified entries. |
| `unavailable` | Confirmed not to work with the current backend. | Requires `allow_risky_model`; retained for history and expected to fail. |

### Model tools

| Tool | Model ID | Name | Official | Min. tier | Status | Last tested (UTC) |
| --- | --- | --- | --- | --- | --- | --- |
| `pplx_best` | `perplexity/best` | Best | `true` | free | `available` | 2026-08-05T23:31:27.726694Z |
| `pplx_deep_research` | `perplexity/deep-research` | Deep research | `true` | pro | `available` | 2026-08-05T23:31:30.488422Z |
| `pplx_sonar` | `perplexity/sonar-2` | Sonar 2 | `true` | pro | `available` | 2026-08-05T23:31:35.277279Z |
| `pplx_gpt56_terra` | `openai/gpt-5.6-terra` | GPT-5.6 Terra | `true` | pro | `available` | 2026-08-05T23:31:39.397301Z |
| `pplx_gpt56_sol` | `openai/gpt-5.6-sol` | GPT-5.6 Sol | `true` | max | `available` | 2026-08-05T23:31:48.536501Z |
| `pplx_claude_s50` | `anthropic/claude-sonnet-5` | Claude Sonnet 5 | `true` | pro | `available` | 2026-08-05T23:31:57.917346Z |
| `pplx_glm52` | `z-ai/glm-5.2` | GLM-5.2 Thinking | `true` | pro | `available` | 2026-08-05T23:32:05.681327Z |
| `pplx_gemini31_pro` | `google/gemini-3.1-pro` | Gemini 3.1 Pro | `true` | pro | `available` | 2026-08-05T23:32:09.529962Z |
| `pplx_kimi_k3` | `moonshot/kimi-k3` | Kimi K3 Thinking | `true` | pro | `available` | 2026-08-05T23:32:13.388185Z |
| `pplx_grok45` | `x-ai/grok-4.5` | Grok 4.5 | `true` | pro | `available` | 2026-08-05T23:32:17.793450Z |
| `pplx_nemotron3_ultra` | `nvidia/nemotron-3-ultra` | Nemotron 3 Ultra | `true` | pro | `available` | 2026-08-05T23:32:26.248167Z |
| `pplx_claude_o50` | `anthropic/claude-opus-5` | Claude Opus 5 | `true` | max | `available` | 2026-08-05T23:32:30.076411Z |
| `pplx_claude_o48` | `anthropic/claude-opus-4.8` | Claude Opus 4.8 | `false` | max | `available` | 2026-08-05T23:32:40.127829Z |
| `pplx_kimi_k26` | `moonshot/kimi-k2.6` | Kimi K2.6 | `false` | pro | `available` | 2026-08-05T23:32:49.769820Z |
| `pplx_nemotron3_super` | `nvidia/nemotron-3-super` | Nemotron 3 Super | `false` | pro | `available` | 2026-08-05T23:32:55.399808Z |
| `pplx_gpt54` | `openai/gpt-5.4` | GPT-5.4 | `false` | pro | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_claude_o47` | `anthropic/claude-opus-4.7` | Claude Opus 4.7 | `false` | max | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_claude_s46` | `anthropic/claude-sonnet-4.6` | Claude Sonnet 4.6 | `false` | pro | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_gpt4o` | `openai/gpt4o` | GPT-4o | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_gpt41` | `openai/gpt41` | GPT-4.1 | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_gpt5` | `openai/gpt5` | GPT-5 | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_gpt51` | `openai/gpt51` | GPT-5.1 | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_gpt5_mini` | `openai/gpt5-mini` | GPT-5 Mini | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_gpt5_nano` | `openai/gpt5-nano` | GPT-5 Nano | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_gpt5_pro` | `openai/gpt5-pro` | GPT-5 Pro | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_gpt52` | `openai/gpt52` | GPT-5.2 | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_gpt52_pro` | `openai/gpt52-pro` | GPT-5.2 Pro | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_gpt55` | `openai/gpt55` | GPT-5.5 | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_claude2` | `anthropic/claude2` | Claude Sonnet 4.0 | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_gemini25pro` | `google/gemini25pro` | Gemini 2.5 Pro | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_gemini30pro` | `google/gemini30pro` | Gemini 3 Pro | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_gemini30flash` | `google/gemini30flash` | Gemini 3 Flash | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_gemini35flash` | `google/gemini35flash` | Gemini 3.5 Flash | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_grok` | `x-ai/grok` | Grok 3 Beta | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_claude40opus` | `anthropic/claude40opus` | Claude Opus 4.0 | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_claude41opus` | `anthropic/claude41opus` | Claude Opus 4.1 | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_claude45opus` | `anthropic/claude45opus` | Claude Opus 4.5 | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_claude46opus` | `anthropic/claude46opus` | Claude Opus 4.6 | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_claude45sonnet` | `anthropic/claude45sonnet` | Claude Sonnet 4.5 | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_claude45haiku` | `anthropic/claude45haiku` | Claude Haiku 4.5 | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_kimik2` | `moonshot/kimik2` | Kimi K2 | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_kimik25` | `moonshot/kimik25` | Kimi K2.5 | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_grok4` | `x-ai/grok4` | Grok 4 | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_grok41` | `x-ai/grok41` | Grok 4.1 | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_o4mini` | `openai/o4mini` | o4-mini | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |
| `pplx_o3pro` | `openai/o3pro` | o3-pro | `false` | unknown | `unavailable` | 2026-08-16T00:00:00Z |

### Custom tool

`pplx_custom` accepts an arbitrary `custom:<identifier>` model and requires explicit risky-model acknowledgement.

<!-- END GENERATED MODEL CATALOG -->

**All tools support `source_focus`:** `web`, `academic`, `social`, `finance`, `all`
