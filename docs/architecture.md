# Architecture Overview

## Layers

```
┌─────────────────────────────────────────────┐
│                 UI Layer                     │
│  TuiChatApp, ChatView, CommandHandler        │
│  Pure presentation, no business logic        │
└──────────────────┬──────────────────────────┘
                   │ depends on
                   ▼
┌─────────────────────────────────────────────┐
│            Core Business Layer               │
│  ChatEngine, ModelManager, SessionManager,   │
│  UserManager, PromptManager, ConfigManager   │
│  All depend on StorageBackend (ABC only)     │
└──────────────────┬──────────────────────────┘
                   │ depends on
                   ▼
┌─────────────────────────────────────────────┐
│              Storage Layer                   │
│  StorageBackend (ABC, 23 methods)            │
│  SQLiteBackend | MySQLBackend | FileBackend  │
│  StorageFactory: config → backend            │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│              Data / File                     │
│  SQLite DB | MySQL DB | JSON files           │
└─────────────────────────────────────────────┘
```

## Dependency Rules

| From → To | Allowed | Notes |
|-----------|---------|-------|
| UI → Core | ✅ | Through Manager classes |
| UI → Storage | ❌ | Must go through Core |
| Core → Storage ABC | ✅ | Through StorageBackend interface |
| Core → Concrete Backend | ❌ | Only StorageBackend ABC |
| Storage → Core | ❌ | Lower layer doesn't depend on upper |
| ChatEngine → ModelManager | ❌ | Model injected via set_model() |

## Key Classes

### ChatEngine
- **In**: Model instance (via `set_model()`)
- **Out**: ChatResponse, stream tokens
- **Owns**: In-memory conversation history
- **Does NOT**: Create models, manage providers, access storage

### ModelManager
- **In**: LLMConfig
- **Out**: ChatOpenAI instances (fresh each call)
- **Owns**: Provider registry, current model selection
- **Does NOT**: Chat, manage memory

### SessionManager
- **In**: StorageBackend
- **Out**: Session, Message Pydantic models
- **Owns**: Session + message CRUD, search, history reopen
- **Does NOT**: Chat, manage models

### StorageBackend
- **Type**: ABC, 23 abstract methods
- **Implementations**: SQLiteBackend (aiosqlite), MySQLBackend (aiomysql), FileBackend (JSON)
- **Methods**: Users(5), Sessions(6), Messages(3), Presets(5), Config(2), Lifecycle(2)

## Provider Extension

New providers implement `BaseProvider`:

```python
class MyProvider(BaseProvider):
    name = "my-provider"
    base_url = "https://api.example.com/v1"
    api_key_env = "MY_API_KEY"
    default_model = "my-model"
```

Registry: `model_manager.register_provider(MyProvider())`
Model list: added via `config.yaml:llm.models`
No code changes to ChatEngine or ModelManager required.

## Data Flow

```
User Input → TuiChatApp
  → /command → CommandHandler → Manager CRUD
  → message → ChatEngine.stream_chat() → tokens → ChatView
            → SessionManager.add_message() × 2 → Storage → DB/File
```
