# Development Guide

## Setup

```bash
git clone <repo>
cd langchain-chat
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Configure

```bash
cp config/.env.example .env
# Set your API keys:
#   OPENAI_API_KEY=sk-...
#   DEEPSEEK_API_KEY=sk-...
#   OPENROUTER_API_KEY=sk-...
```

## Running

```bash
# TUI chat
python -m langchain_chat.main

# or explicitly:
langchain-chat-tui
```

## Testing

```bash
pytest                          # All 313 tests
pytest tests/test_storage.py    # Single file
pytest -v                       # Verbose
```

Tests use:
- `FakeModel` — no real API calls
- `SQLiteBackend(:memory:)` — no disk I/O for most tests
- `FileBackend(tmp_path)` — temp directories
- No MySQL server required

## Code Quality

```bash
ruff check .       # Lint
ruff format .      # Format
ruff check --fix . # Auto-fix
```

## Project Conventions

### Commit Format
```
feat: stepN <description>
```

### Tag Format
```
v0.1.0-stepN-<slug>
```

### Type Hints
Required on all public APIs. Use `from __future__ import annotations`.

### Architecture Rules
- Never import concrete storage backends in Core layer
- Never import storage in UI layer
- ChatEngine receives model via injection, never creates one
- All managers depend on StorageBackend ABC

## Adding a New Provider

```python
# 1. Create provider class (no existing file changes needed)
class GroqProvider(BaseProvider):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"
    api_key_env = "GROQ_API_KEY"
    default_model = "llama-3.1-8b-instant"

# 2. Add model list to config.yaml
# llm:
#   models:
#     groq:
#       - llama-3.1-8b-instant

# 3. Register at runtime
model_manager.register_provider(GroqProvider())
```

## Adding a New Storage Backend

```python
# Implement all 23 StorageBackend abstract methods
class PostgresBackend(StorageBackend):
    ...

# Register in factory
# storage/factory.py: +1 line
if backend_type == "postgres":
    return PostgresBackend(storage_config)
```

## Adding a TUI Command

```python
# ui/commands.py
async def _cmd_mycommand(ctx, args):
    ...
    return CONTINUE

_BUILTIN_COMMANDS["mycommand"] = _cmd_mycommand
```
