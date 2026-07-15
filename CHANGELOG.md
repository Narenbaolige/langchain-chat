# Changelog

All notable changes to langchain-chat.

## [v0.1.0-step18-chatanywhere] — 2026-07-15

### Added
- ChatAnywhereProvider: 4th built-in provider (free API via chatanywhere proxy)
- Default provider switched to `chatanywhere` in config.yaml
- `.env` with free API key (gitignored), `.env.example` template
- Lazy MySQL import: `aiomysql` only loaded when MySQL backend is used

### Changed
- Default `__main__` now launches TUI (`main_tui()`) instead of bootstrap demo
- `.env` loading uses `override=True` for project-root variables

## [v0.1.0-step17-git-engineering] — 2026-07-15

### Added
- Git tag conventions (`v0.1.0-stepN-<slug>`)
- Commit format enforcement (`feat:` / `fix:` / `docs:` / `chore:`)
- `.gitignore` coverage: `.env`, `__pycache__`, `.pytest_cache`, `*.db`, `dist/`
- Git hooks placeholders for pre-commit lint + pre-push test

## [v0.1.0-step16b-quality] — 2026-07-15

### Added
- Session pagination: `list_sessions(limit, offset)` across StorageBackend + 3 backends
- Integration tests: User→Session→Message full-chain (5 tests)
- pytest-cov dependency for coverage reporting

## [v0.1.0-step16a-security] — 2026-07-15

### Added
- Input length limit (`max_input_length=5000`) in TUI
- Context window trimming (`trim_messages`) with sliding window + token estimation
- JSON log formatter (`JsonFormatter`)
- API key masking in logs (`mask_api_key`)
- `SecurityConfig` Pydantic model
- 15 security tests

## [v0.1.0-step15-multi-env] — 2026-07-15

### Added
- Multi-environment config: `config.dev.yaml`, `config.test.yaml`, `config.prod.yaml`
- `APP_ENV` environment variable selection
- `LANGCHAIN_*` env var overrides with `__` nesting
- Deep merge for config files
- 23 ConfigManager tests

## [v0.1.0-step14-audit-docs] — 2026-07-15

### Added
- Architecture compliance audit (zero violations)
- Storage backend interface consistency verification (23/23 methods × 3 backends)
- Core responsibility boundary audit
- README.md rewritten with quick start, commands table, architecture
- CLAUDE.md updated with full step status
- docs/: architecture.md, development.md, roadmap.md, architecture-freeze

## [v0.1.0-step13-testing] — 2026-07-15

### Added
- Edge case tests: ChatEngine errors, UserManager boundaries, SessionManager edges
- Storage lifecycle error tests, Factory edge tests
- CommandHandler edge cases, ChatView smoke tests
- Test total: 269 (up from 230)

## [v0.1.0-step12-file-logging] — 2026-07-15

### Added
- FileBackend: JSON-file-based storage (23 methods, zero extra deps)
- Unified logging: `setup_logging()` with console + file handlers
- Idempotent logging via `_configured` guard
- 22 FileBackend tests

## [v0.1.0-step11-mysql] — 2026-07-15

### Added
- MySQLBackend: full StorageBackend via aiomysql connection pool (23 methods)
- MySQL DDL with InnoDB, utf8mb4, FOREIGN KEY CASCADE
- StorageFactory: `type=mysql` case
- MySQLConfig Pydantic model
- 19 MySQL backend tests

## [v0.1.0-step10-model-manager] — 2026-07-15

### Added
- ModelManager: provider registry, model switching, model factory
- BaseProvider ABC + OpenAIProvider, DeepSeekProvider, OpenRouterProvider
- Model lists from config.yaml (no code change needed to add models)
- TUI commands: `/providers`, `/models`, `/provider`, `/model`
- Dynamic prompt showing model: `[user:#session:openai/gpt-4o-mini] >`

### Changed
- ChatEngine: constructor takes `model=` (injected), `set_model()` for runtime switching
- ChatEngine no longer imports ChatOpenAI or LLMConfig

## [v0.1.0-step9-tui-polish] — 2026-07-15

### Added
- Panel-wrapped user messages, green-tinted streaming tokens
- Grouped `/help` with 4 categories
- Dynamic prompt: `[user:#session⚡] >`
- Friendly API error messages (missing key, timeout, network)
- `build_prompt()`, `show_warning()`, `show_sessions` with message counts

## [v0.1.0-step8-session-enhance] — 2026-07-15

### Added
- Session history list, title search (`/search`), rename (`/rename`)
- Session reopen (`/open`) with LangChain message restoration
- Session delete (`/delete-session`)
- `reopen_session()`, `update_session()`, `search_sessions()`, `search_messages()`
- ChatEngine: `load_messages()` for history restoration
- `updated_at` bump on `add_message()`

## [v0.1.0-step7-tui] — 2026-07-15

### Added
- TUI chat application with Rich (Panel, Table, Rule, Text)
- CommandHandler: 14 commands (/help, /quit, /clear, /user, /preset, etc.)
- ChatView: pure display layer
- Message persistence: user + assistant saved to DB after each turn
- Session and Message Pydantic models
- SessionManager: CRUD + message management
- Migrations framework in database.py

## [v0.1.0-step6-chat-engine] — 2026-07-15

### Added
- ChatEngine: `chat()`, `stream_chat()`, `clear_memory()`
- In-memory conversation history (LangChain BaseMessage)
- Token counting via `usage_metadata`
- FakeModel for testing (no real API key needed)

## [v0.1.0-step5-prompt] — 2026-07-15

### Added
- PromptManager: CRUD for prompt presets
- Preset Pydantic model (system/user prompt_type)
- 15 PromptManager tests

## [v0.1.0-step4-user] — 2026-07-15

### Added
- UserManager: CRUD for users
- User Pydantic model
- 15 UserManager tests

## [v0.1.0-step3-storage] — 2026-07-15

### Added
- StorageBackend ABC (14 → 23 abstract methods)
- SQLiteBackend with aiosqlite
- StorageFactory
- 5 tables: users, sessions, messages, presets, configs
- 30 storage tests

## [v0.1.0-step2-config] — 2026-07-15

### Added
- ConfigManager: YAML + .env → Pydantic ProjectConfig
- AppConfig, StorageConfig, LLMConfig, LoggingConfig models

## [v0.1.0-step1-init] — 2026-07-15

### Added
- Project skeleton: pyproject.toml, src/, tests/
- Dependencies: pydantic, pyyaml, python-dotenv, pytest, ruff
