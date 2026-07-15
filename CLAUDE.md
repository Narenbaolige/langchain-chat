# langchain-chat

Enterprise-grade AI ChatBot framework based on LangChain.

## Architecture

```
UI Layer → Core Business Layer → Storage Layer → Database/File
```

- UI never touches database directly
- Storage is accessed through abstract `StorageBackend` interface (ABC, 23 methods)
- Three backends: SQLiteBackend, MySQLBackend, FileBackend
- Multi-provider: OpenAIProvider, DeepSeekProvider, OpenRouterProvider
- Model injection: ChatEngine receives model via `set_model()`, does not own lifecycle
- ModelManager owns provider registry + model switching

## Project Status — All 18 Steps Complete ✅

| Step | Scope | Status |
|------|-------|--------|
| 1 | Project initialization | ✅ v0.1.0-step1-init |
| 2 | Configuration management | ✅ v0.1.0-step2-config |
| 3 | Storage abstraction + SQLite | ✅ v0.1.0-step3-storage |
| 4 | User system | ✅ v0.1.0-step4-user |
| 5 | Prompt system | ✅ v0.1.0-step5-prompt |
| 6 | Chat Engine (LangChain) | ✅ v0.1.0-step6-chat-engine |
| 7 | TUI chat | ✅ v0.1.0-step7-tui |
| 8 | Session enhancements | ✅ v0.1.0-step8-session-enhance |
| 9 | UX optimization | ✅ v0.1.0-step9-tui-polish |
| 10 | Model management | ✅ v0.1.0-step10-model-manager |
| 11 | MySQL Backend | ✅ v0.1.0-step11-mysql |
| 12 | File Backend + Logging | ✅ v0.1.0-step12-file-logging |
| 13 | Test system | ✅ v0.1.0-step13-testing |
| 14 | Architecture audit + docs | ✅ v0.1.0-step14-audit-docs |
| 15 | Multi-env configuration | ✅ v0.1.0-step15-multi-env |
| 16a | Security enhancement | ✅ v0.1.0-step16a-security |
| 16b | Code quality | ✅ v0.1.0-step16b-quality |
| 17 | Git engineering | ✅ v0.1.0-step17-git-engineering |
| 18 | ChatAnywhere integration | ✅ v0.1.0-step18-chatanywhere |

## Engineering Rules

1. Each Step is self-contained — never implement features from future Steps
2. Every Step ends with: pytest pass + ruff pass + git commit + git tag
3. Commit format: `feat: stepN <description>`
4. Tag format: `v0.1.0-stepN-<slug>`
5. Minimum change principle — don't refactor for its own sake
6. Type hints on all public APIs
7. Backward compatible within the same major version

## Key Design Rules (Do Not Break)

- UI → Core → Storage ABC → DB (strict one-way)
- ChatEngine: chat/stream/memory/token only — no model creation or DB access
- ModelManager: provider/model lifecycle — no chat logic
- ChatEngine receives model via `set_model()` — never creates ChatOpenAI
- All Core managers depend on `StorageBackend` (ABC), never concrete backends
- `get_current_model()` returns fresh instance every call (no caching)
- Messages persisted immediately after each chat turn

## Testing

```bash
pytest                    # 313 tests
ruff check .              # Lint
ruff format .             # Format
```

Tests never require: real API keys, MySQL server, network access.

## Documentation

- [Architecture Freeze](docs/architecture-freeze-v0.1.0-step13.md) — baseline for Step14-17
- [Architecture](docs/architecture.md)
- [Development Guide](docs/development.md)
- [Roadmap](docs/roadmap.md)
