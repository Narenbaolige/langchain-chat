# langchain-chat

Enterprise-grade AI ChatBot framework based on LangChain.

## Architecture

```
UI Layer → Core Business Layer → Storage Layer → Database/File
```

- Strict layered isolation — UI never touches database directly
- Storage access through abstract `StorageBackend` interface
- Three backend implementations: SQLite, MySQL, File (JSON)
- Multi-provider LLM support: OpenAI, DeepSeek, OpenRouter
- Runtime model switching without restart

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Configure

```bash
cp config/.env.example .env
# Edit .env with your API key:
#   OPENAI_API_KEY=sk-...
```

Edit `config/config.yaml` to select storage backend and LLM provider.

### Launch TUI Chat

```bash
python -m langchain_chat.main
# or:
langchain-chat-tui
```

### Commands

| Group | Command | Description |
|-------|---------|-------------|
| **Chat** | `/clear` | Clear conversation memory |
| | `/stats` | Show conversation statistics |
| **Session** | `/sessions` | List recent sessions |
| | `/search <q>` | Search sessions by title |
| | `/rename <title>` | Rename current session |
| | `/open <id>` | Reopen historical session |
| | `/delete-session <id>` | Delete a session |
| **User & Preset** | `/users` | List users |
| | `/user <name>` | Switch user |
| | `/presets` | List presets |
| | `/preset <name>` | Load preset as system prompt |
| | `/system <text>` | Set raw system prompt |
| **Model** | `/providers` | List available providers |
| | `/models [provider]` | List models |
| | `/provider <name>` | Switch provider |
| | `/model <name>` | Switch model |
| **System** | `/help` | Show help |
| | `/quit` | Exit |

## Storage Backends

| Backend | Type | Best For |
|---------|------|----------|
| SQLite | `sqlite` | Development, single-user |
| MySQL | `mysql` | Production, multi-user |
| File | `file` | Lightweight, zero-dependency |

Configure in `config/config.yaml`:

```yaml
storage:
  type: sqlite          # sqlite | mysql | file
  database: data/chat.db
  mysql:                # only for type=mysql
    host: localhost
    port: 3306
    database: langchain_chat
    user: root
    password: ""
```

## Supported Models

Configured in `config/config.yaml:llm.models`:

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-4.1, o4-mini |
| DeepSeek | deepseek-chat, deepseek-reasoner |
| OpenRouter | openai/gpt-4o-mini, openai/gpt-4o, anthropic/claude-sonnet-4 |

Add models via config — no code change required.

## Development

```bash
pip install -e ".[dev]"
pytest                    # 269 tests
ruff check .              # Lint
ruff format .             # Format
```

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Development Guide](docs/development.md)
- [Roadmap](docs/roadmap.md)

## License

MIT
