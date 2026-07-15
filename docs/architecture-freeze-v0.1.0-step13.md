# Architecture Freeze Document

**Baseline**: `v0.1.0-step13-testing`  
**Date**: 2026-07-15  
**Purpose**: Step14–17 开发基线，冻结当前架构契约。

---

## 1. 当前架构

### 1.1 分层架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                        UI Layer                                   │
│  ui/app.py          TuiChatApp (启动 + REPL + 持久化 + 切换)      │
│  ui/chat_view.py    ChatView  (纯 Rich 渲染，零业务逻辑)           │
│  ui/commands.py     CommandHandler (20 命令注册/解析/分发)        │
└────────────────────────────┬─────────────────────────────────────┘
                             │ 依赖 (import)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Core Business Layer                            │
│  chat_engine.py     对话引擎 (memory / stream / token)             │
│  model_manager.py   模型管理 (provider 注册 / 切换 / 工厂)          │
│  provider.py        BaseProvider ABC + 3 内置 Provider             │
│  user_manager.py    用户 CRUD                                     │
│  prompt_manager.py  Prompt 预设 CRUD                               │
│  session_manager.py 会话 + 消息 CRUD + 搜索 + 历史恢复              │
│  config_manager.py  YAML + .env → ProjectConfig                   │
│  config_models.py   Pydantic 配置模型                              │
└────────────────────────────┬─────────────────────────────────────┘
                             │ 依赖 StorageBackend (ABC)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Storage Layer                                │
│  base.py            StorageBackend ABC (23 抽象方法)              │
│  factory.py         StorageFactory.create(type)                   │
│  sqlite_backend.py  SQLiteBackend  (aiosqlite)                    │
│  mysql_backend.py   MySQLBackend   (aiomysql, 连接池)              │
│  file_backend.py    FileBackend    (JSON, asyncio.Lock)           │
│  database.py        SQLite DDL + _run_migrations()                │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Data Layer                                   │
│  models/user.py      User    (id, username, created_at)           │
│  models/session.py   Session (id, user_id, preset_id, title,      │
│                               created_at, updated_at)             │
│  models/message.py   Message (id, session_id, role, content,      │
│                               created_at)                         │
│  models/preset.py    Preset  (id, name, content, prompt_type,     │
│                               created_at)                         │
└──────────────────────────────────────────────────────────────────┘

                         ┌─────────────────┐
                         │  Infrastructure │
                         │  logging_config │
                         │  main.py        │
                         └─────────────────┘
```

### 1.2 依赖规则（绝对不可违反）

```
✅ UI       → Core        (允许)
✅ Core     → Storage ABC (允许)
✅ Storage  → DB/File     (允许)
✅ All      → Models      (允许)

❌ UI       → Storage     (禁止 — 必须通过 Core Manager)
❌ Core     → SQLiteBackend/MySQLBackend (禁止 — 只依赖 StorageBackend ABC)
❌ Storage  → Core        (禁止 — 下层不依赖上层)
❌ ChatEngine → ModelManager (禁止 — ChatEngine 不拥有模型，通过 set_model() 注入)
```

### 1.3 数据流

```
[用户输入]
    │
    ▼
TuiChatApp._input_loop()
    │
    ├── /command ──→ CommandHandler.handle() ──→ Manager CRUD
    │
    └── 普通文本 ──→ _handle_chat(message)
                      │
                      ├── engine.stream_chat(message, system_prompt)
                      │       │
                      │       └── ChatOpenAI.astream() ──→ yield tokens
                      │
                      ├── session_manager.add_message(sid, "user", message)
                      └── session_manager.add_message(sid, "assistant", response)
```

---

## 2. 模块职责

### 2.1 UI Layer

| 文件 | 类 | 职责 | 不可做 |
|------|----|------|--------|
| `app.py` | `TuiChatApp` | 组装依赖、启动流程、REPL 循环、消息持久化、模型/用户/session 切换回调 | 不直接访问 Storage |
| `chat_view.py` | `ChatView` | Rich 渲染：Panel/Table/Rule/Text、分组帮助、动态 prompt | 不含业务逻辑 |
| `commands.py` | `CommandHandler` | 命令注册/解析/分发、CommandContext 传输 | 不含业务逻辑（通过回调委托） |

### 2.2 Core Business Layer

| 文件 | 类 | 职责 | 依赖 |
|------|----|------|------|
| `chat_engine.py` | `ChatEngine` | 对话、流式输出、内存历史、token 计数 | 注入的 model 实例 |
| `model_manager.py` | `ModelManager` | Provider 注册、模型切换、模型工厂 | LLMConfig + BaseProvider |
| `provider.py` | `BaseProvider` | 定义 Provider 契约 (name/base_url/api_key_env/default_model) | ChatOpenAI |
| `user_manager.py` | `UserManager` | 用户 CRUD + 校验 | StorageBackend |
| `prompt_manager.py` | `PromptManager` | 预设 CRUD + 校验 | StorageBackend |
| `session_manager.py` | `SessionManager` | 会话+消息 CRUD、搜索、历史恢复 | StorageBackend |

### 2.3 Storage Layer

| 文件 | 类 | 实现方式 | 方法数 |
|------|----|---------|--------|
| `base.py` | `StorageBackend` (ABC) | — | 23 abstract |
| `sqlite_backend.py` | `SQLiteBackend` | aiosqlite, `?` placeholder | 23 |
| `mysql_backend.py` | `MySQLBackend` | aiomysql, pool, `%s` placeholder | 23 |
| `file_backend.py` | `FileBackend` | JSON + asyncio.Lock | 23 |
| `factory.py` | `StorageFactory` | `create(type)` → backend | — |

### 2.4 Data Models

| 模型 | 字段 |
|------|------|
| `User` | id, username, created_at |
| `Session` | id, user_id, preset_id, title, created_at, updated_at |
| `Message` | id, session_id, role, content, created_at |
| `Preset` | id, name, content, prompt_type, created_at |

---

## 3. 数据流

### 3.1 模型切换流

```
TUI: /model gpt-4o
  → commands._cmd_model("gpt-4o")
  → ctx.on_model_change("openai", "gpt-4o")
  → app._on_model_change(provider, model)
      ├── model_manager.switch_model(provider, model)
      ├── new_model = model_manager.get_current_model()    # 每次新鲜，无缓存
      └── engine.set_model(new_model)                      # ChatEngine 被动接收
```

### 3.2 消息持久化流（每轮对话）

```
TUI._handle_chat(message)
  ├── engine.stream_chat(message, system_prompt)    # 流式输出
  │     └── yield tokens → ChatView.stream_token()
  ├── session_manager.add_message(sid, "user", message)       # 用户消息入库
  └── session_manager.add_message(sid, "assistant", response) # 助手回复入库
```

### 3.3 Session 恢复流

```
TUI: /open 5
  → commands._cmd_open("5")
  → ctx.on_session_open(5)
  → app._on_session_open(5)
      ├── session_manager.reopen_session(5)
      │     └── (session, [Message, Message, ...])
      ├── 转换 Message → LangChain HumanMessage / AIMessage
      └── engine.load_messages(langchain_messages)    # 恢复内存历史
```

### 3.4 存储后端切换流

```
main.py
  config = get_config()
  storage = StorageFactory.create(config.storage)
      ├── type="sqlite" → SQLiteBackend(config)
      ├── type="mysql"  → MySQLBackend(config)
      └── type="file"   → FileBackend(config)

  UserManager(storage)     ─┐
  PromptManager(storage)    ├── 全部依赖 StorageBackend ABC
  SessionManager(storage)  ─┘   切换后端对 Core 层完全透明
```

---

## 4. 扩展接口

### 4.1 新增 Provider

```python
# 10 行代码，ModelManager 和 ChatEngine 零改动
class GroqProvider(BaseProvider):
    @property
    def name(self) -> str: return "groq"
    @property
    def base_url(self) -> str | None: return "https://api.groq.com/openai/v1"
    @property
    def api_key_env(self) -> str: return "GROQ_API_KEY"
    @property
    def default_model(self) -> str: return "llama-3.1-8b-instant"

# 注册
mgr = ModelManager(config)
mgr.register_provider(GroqProvider())
```

### 4.2 新增 Storage Backend

```python
# 实现 StorageBackend 23 个抽象方法
class PostgresBackend(StorageBackend):
    async def initialize(self) -> None: ...
    async def create_user(self, username: str) -> dict: ...
    # ... 全部 23 个方法

# 注册
# storage/factory.py: +1 行
if backend_type == "postgres":
    return PostgresBackend(storage_config)
```

### 4.3 新增 TUI 命令

```python
# ui/commands.py
async def _cmd_mycommand(ctx: CommandContext, args: str) -> ActionResult:
    # 通过 ctx 访问所有 Manager
    ...
    return CONTINUE

_BUILTIN_COMMANDS["mycommand"] = _cmd_mycommand
```

### 4.4 配置扩展

```yaml
# config.yaml — 新增任意 section
my_feature:
  enabled: true
  option: value
```

```python
# config_models.py — 对应 Pydantic model
class MyFeatureConfig(BaseModel):
    enabled: bool = True
    option: str = "value"
```

---

## 5. 不允许破坏的设计原则

| # | 原则 | 说明 |
|---|------|------|
| 1 | **分层隔离** | UI → Core → Storage ABC → DB。禁止跨层访问。 |
| 2 | **依赖倒置** | Core 只依赖 `StorageBackend` (ABC)，不依赖 `SQLiteBackend` 等具体类。 |
| 3 | **ChatEngine 边界** | ChatEngine 不拥有模型、不管理 Provider、不访问数据库。只负责对话+内存+流式+token。 |
| 4 | **ModelManager 单一职责** | ModelManager 只管理 Provider/模型生命周期，不涉及聊天逻辑。 |
| 5 | **模型注入** | 模型通过 `engine.set_model()` 注入，ChatEngine 无 `ChatOpenAI` import。 |
| 6 | **每次新鲜** | `get_current_model()` 不缓存，每次返回新实例，保证参数变更即时生效。 |
| 7 | **消息持久化** | 每轮对话后立即 `add_message()` 保存 user + assistant 消息。 |
| 8 | **向后兼容** | StorageBackend 接口变更必须兼容已有 3 个实现。 |
| 9 | **测试隔离** | 测试不依赖真实 API Key / MySQL / 网络。 |
| 10 | **配置中心化** | 所有配置通过 `config.yaml` + `.env` 管理，不硬编码。 |

### 5.1 禁止事项速查

| 行为 | 判定 |
|------|------|
| UI 直接 `import SQLiteBackend` | ❌ |
| `ChatEngine.__init__` 接受 `LLMConfig` | ❌ (Step10 已移除) |
| Core Manager 直接 `import MySQLBackend` | ❌ |
| `model_manager.py` 调用 `engine.chat()` | ❌ |
| 新 Provider 修改 `ChatEngine` | ❌ |
| API Key 写死在代码中 | ❌ |
| 在 `config.yaml` 之外散落配置 | ❌ |
| 测试需要 MySQL 服务器才能运行 | ❌ |

---

## 6. Step14–17 演进路线

### Step14: 架构审计 + 文档

| 目标 | 具体内容 |
|------|---------|
| 架构合规检查 | 验证所有 import 不跨层 |
| 接口一致性 | 验证 3 个 Backend 统一实现 23 方法 |
| 职责边界 | 检查 ChatEngine 不 import ModelManager |
| 文档 | 更新 README, CLAUDE.md, API 文档 |
| 不改代码 | 纯审计，发现问题记录不修 |

### Step15: 多环境工程化

| 目标 | 具体内容 |
|------|---------|
| dev/test/prod 配置分离 | `config/config.dev.yaml`, `config.test.yaml`, `config.prod.yaml` |
| 环境变量覆盖 | `.env` 覆盖 config.yaml 值 |
| 日志级别按环境 | dev=DEBUG, test=WARNING, prod=INFO |
| 不改 Core | 仅 config + main.py |

### Step16a: 安全增强

| 目标 | 具体内容 |
|------|---------|
| 输入限制 | max_input_length 配置，超长截断/拒绝 |
| 上下文管理 | 滑动窗口 token 限制，避免内存/API 爆炸 |
| 模型 Token 控制 | 按模型动态设置 max_tokens |
| API Key 保护 | 日志脱敏，不允许 key 明文出现在日志中 |

### Step16b: 代码质量提升

| 目标 | 具体内容 |
|------|---------|
| Session 分页 | `list_sessions(limit, offset)` |
| 类型标注 | mypy / pyright 兼容性修正 |
| 集成测试 | User→Session→Message 全链路测试 |
| 补充测试 | 针对 Step16a 安全功能 |

### Step17: Git 工程化管理

| 目标 | 具体内容 |
|------|---------|
| Tag 规范 | 统一 tag message 格式 |
| CHANGELOG | 基于 git log 生成 |
| .gitignore 完善 | 排除 logs/, *.db, .env |
| Git Hooks | pre-commit: ruff + pytest |

---

**冻结版本**: `v0.1.0-step13-testing`  
**冻结时间**: 2026-07-15  
**下次复审**: Step14 完成后
