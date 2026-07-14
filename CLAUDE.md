# langchain-chat

Enterprise-grade AI ChatBot framework based on LangChain.

## Architecture

Strict layered architecture:

```
UI Layer → Core Business Layer → Storage Layer → Database/File
```

- UI never touches database directly
- Storage is accessed through abstract `StorageBackend` interface
- Dependencies flow one direction: upper depends on lower, never reverse

## 15-Step Roadmap

| Step | Scope | Status |
|------|-------|--------|
| 1 | Project initialization | ✅ Done |
| 2 | Configuration management | ✅ Done |
| 3 | Storage abstraction + SQLite | 🔜 Next |
| 4 | User system | ⬜ |
| 5 | Prompt system | ⬜ |
| 6 | Chat Engine (LangChain) | ⬜ |
| 7 | TUI chat | ⬜ |
| 8 | Session enhancements | ⬜ |
| 9 | UX optimization | ⬜ |
| 10 | Model management | ⬜ |
| 11 | MySQL Backend | ⬜ |
| 12 | File Backend + Logging | ⬜ |
| 13 | Test system | ⬜ |
| 14 | Documentation + API stubs | ⬜ |
| 15 | dev/test/prod multi-env | ⬜ |

## Key Rules

1. Each Step is self-contained — never implement features from future Steps
2. Every Step ends with: working code + pytest pass + ruff pass + git commit + git tag
3. Commit format: `feat: stepN <description>` (Step1 used `chore:`)
4. Tag format: `v0.1.0-stepN-<slug>`
5. MVP and incremental — no premature design for unused features
6. Type hints on all public APIs

## Completed Steps

### Step1 — Project Init
- Tag: `v0.1.0-step1-init`
- Dependencies: pydantic, pyyaml, python-dotenv (runtime); pytest, pytest-asyncio, ruff (dev)
- No LangChain deps yet (deferred to Step6)

### Step2 — Config Management
- Tag: `v0.1.0-step2-config`
- ConfigManager loads YAML + .env → Pydantic-validated ProjectConfig
- Config models: AppConfig, StorageConfig, LLMConfig, LoggingConfig
- Config files: `config/config.yaml`, `config/logging.yaml`, `config/.env.example`
- Module-level convenience: `get_config_manager()`, `get_config()`

## Testing & Linting

```bash
pytest          # Run all tests
ruff check .    # Lint
ruff format .   # Format
```
