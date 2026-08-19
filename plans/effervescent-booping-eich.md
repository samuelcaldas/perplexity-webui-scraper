# Plan: Thinking-Only & Thinking Suffix/Mid-Name Models Consolidation and Dynamic Resolution

## Context & Motivation

O catálogo de modelos estáticos (`models.json`) ainda continha modelos duplicados para variantes _thinking_ e variantes históricas com o termo `thinking` sufixado diretamente sem hífen (ex.: `claude45haikuthinking`, `kimik2thinking`), sufixado com hífen (ex.: `gpt-5.4-thinking`), no meio do nome (ex.: `gpt51-low-thinking`, `gemini-3.1-pro-thinking-high`, `claude-3.7-sonnet-thinking-20250219`), ou modelos que são exclusivamente de raciocínio (_thinking-only_, ex.: `deep-research`, `kimi-k3`, `o3pro`, `o4mini`).

Isso poluía a listagem de `/v1/models` no sidecar e na documentação gerada, além de limitar a flexibilidade de resolução dinâmica quando clientes (Claude CLI, OpenAI SDK, LiteLLM) solicitam modelos com modificadores de raciocínio no meio ou no fim da string do modelo.

## Objetivos da Solução

1. **Deduplicação do Catálogo (`_static/models.json`)**:
   - Manter apenas modelos base canônicos em `models.json`.
   - Mapear todas as variantes de pensamento como aliases do modelo base e configurar `supports_thinking: true`, `thinking_identifier` e `thinking_only: true` onde apropriado.
2. **Suporte a Modelos Thinking-Only (`Model.thinking_only`)**:
   - Adicionar campo booleano `thinking_only: bool = False` ao schema `Model` em `models/types.py`.
   - Se `thinking_only == True`, o modelo sempre opera em modo de pensamento (utilizando seu `thinking_identifier` ou `identifier` de raciocínio) sem exigir explicitamente flags adicionais.
3. **Resolução Dinâmica Abrangente (`ModelRegistry`)**:
   - Suporte a variantes com termo `thinking` / `reasoning` em qualquer posição:
     - Sufixo separado (`-thinking`, `_thinking`, `:thinking`, `/thinking`).
     - Sufixo concatenado sem hífen (`...thinking`, `...reasoning`, ex: `claude45haikuthinking` -> base `claude45haiku`).
     - No meio do nome com modificadores de esforço (ex: `gpt51-low-thinking`, `gemini-3.1-pro-thinking-high`, `claude-3.7-sonnet-thinking-20250219`).
     - Termos negativos explícitos (`nonthinking`, `nonreasoning`, `non-thinking`, `non-reasoning`, `instant`, `direct`).
4. **Sincronização de Docs e Testes**:
   - Atualizar docs gerados via `uv run scripts/render_model_docs.py`.
   - Expandir a suíte de testes cobrindo todos os cenários de resolução (unseparated suffix, middle-name thinking, thinking-only, non-thinking explicit, deduplicated `/v1/models`).

---

## Arquivos Críticos e Modificações

### 1. `src/perplexity_webui_scraper/models/types.py`

- Adicionar o campo `thinking_only: bool = False` no modelo Pydantic `Model`.

### 2. `src/perplexity_webui_scraper/models/registry.py`

- Adicionar parser de tokens de thinking/reasoning e esforço:
  - `_extract_thinking_and_effort(identifier: str) -> tuple[str, bool | None, str | None]`
  - Reconhecer:
    - Sufixo de esforço no meio ou fim: `-(low|medium|high)-thinking`, `-thinking-(low|medium|high)`, `-reasoning-(low|medium|high)`.
    - Sufixos concatenados: regex para `([a-z0-9]+?)(?:[-_:]?)(thinking|reasoning)$`.
    - Termos não-thinking: `([a-z0-9]+?)(?:[-_:]?)(nonthinking|nonreasoning|non-thinking|non-reasoning|instant|direct)$`.
    - Termos no meio do nome: remoção de `-thinking-` ou `-reasoning-` preservando o resto da versão/data.
- Atualizar `resolve` e `resolve_for_use`:
  - Se modelo resolvido tem `thinking_only=True`, sempre utilizar o identificador de pensamento e marcar como reasoning.
  - Se extraído `wants_thinking` (ou passado via parâmetro `thinking` / `reasoning_effort`), aplicar `thinking_identifier`.
  - Se extraído `wants_non_thinking`, garantir que usa o `identifier` base não-thinking.

### 3. `src/perplexity_webui_scraper/_static/models.json`

- Consolidar todas as 21 variantes thinking duplicadas para seus respectivos modelos base:
  - **OpenAI**: `gpt-5.4` (+ `gpt-5.4-thinking`), `gpt55` (+ `gpt-5.5-thinking`), `gpt5` (+ `gpt5-thinking`), `gpt51` (+ `gpt51-thinking`, `gpt51-low-thinking`), `gpt52` (+ `gpt52-thinking`). Marcar `o4mini` e `o3pro` com `thinking_only: true`.
  - **Anthropic**: `claude-opus-4.7` (+ `claude-opus-4.7-thinking`), `claude-sonnet-4.6` (+ `claude-sonnet-4.6-thinking`), `claude37sonnet` (+ `claude37sonnetthinking`), `claude40sonnet` (+ `claude40sonnetthinking`), `claude40opus` (+ `claude40opusthinking`), `claude41opus` (+ `claude41opusthinking`), `claude45opus` (+ `claude45opusthinking`), `claude46opus` (+ `claude46opusthinking`), `claude45sonnet` (+ `claude45sonnetthinking`), `claude45haiku` (+ `claude45haikuthinking`).
  - **Moonshot**: `kimi-k3` (`thinking_only: true`), `kimik2` (+ `kimik2thinking`, `thinking_only: true`), `kimik25` (+ `kimik25thinking`, `thinking_only: true`).
  - **Google**: `gemini30flash` (+ `gemini30flash-high`), `gemini35flash` (+ `gemini35flash-medium`, `gemini35flash-high`).
  - **xAI**: `grok4` (+ `grok4nonthinking`), `grok41` (+ `grok41reasoning`, `grok41nonreasoning`).
  - **Perplexity**: `deep-research` (`thinking_only: true`).

### 4. `docs/api-reference.md` e `docs/mcp-server.md`

- Atualizar via `uv run scripts/render_model_docs.py`.

### 5. `tests/test_models.py`, `tests/test_openai_api_compatibility.py`, `tests/test_claude_cli_integration.py`

- Adicionar casos de teste para:
  - Resolução de `claude45haikuthinking` -> base `claude45haiku` com `identifier="claude45haikuthinking"`.
  - Resolução de `kimik2thinking` -> base `kimik2` com `thinking_only=True`.
  - Resolução de `gpt51-low-thinking` -> base `gpt51` com reasoning e effort="low".
  - Resolução de `grok4nonthinking` e `grok41nonreasoning`.
  - Validação de que `/v1/models` não lista modelos duplicados `*thinking`.

---

## Verificação e Testes

1. `uv run --all-extras pytest` - Todos os 197+ testes unitários e de integração passando.
2. `uv run ruff check` e `uv run ty check` - Tipagem estática e linting sem erros.
3. `pnpm prettier --check .` e `pnpm taplo lint *.toml` - Formatação perfeita.
4. `uv run scripts/render_model_docs.py --check` - Documentação sincronizada com o catálogo.
