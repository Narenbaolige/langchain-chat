"""Tests for ChatEngine — the LangChain-powered chat engine.

All tests use a FakeModel so no real API key or network is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from langchain_chat.core.chat_engine import ChatEngine, ChatResponse
from langchain_chat.core.config_models import LLMConfig

# ------------------------------------------------------------------
# Fake model for testing (injected via constructor)
# ------------------------------------------------------------------


class FakeModel:
    """A fake ChatModel that returns predetermined responses.

    Implements the subset of the LangChain ChatModel interface that
    ChatEngine actually calls: ``ainvoke`` and ``astream``.
    """

    def __init__(
        self,
        response: str = "Hello, world!",
        input_tokens: int = 10,
        output_tokens: int = 5,
        total_tokens: int = 15,
    ) -> None:
        self._response = response
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._total_tokens = total_tokens
        self.invoke_count = 0
        self.stream_count = 0
        self.last_messages: list | None = None

    async def ainvoke(self, messages: list) -> MagicMock:
        """Simulate a non-streaming LLM call."""
        self.invoke_count += 1
        self.last_messages = messages
        resp = MagicMock()
        resp.content = self._response
        resp.usage_metadata = {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens": self._total_tokens,
        }
        return resp

    async def astream(self, messages: list):
        """Simulate a streaming LLM call — yields one token per char."""
        self.stream_count += 1
        self.last_messages = messages
        for char in self._response:
            chunk = MagicMock()
            chunk.content = char
            yield chunk


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------


@pytest.fixture
def config() -> LLMConfig:
    """A minimal LLMConfig for testing."""
    return LLMConfig(
        model="gpt-4o-mini", temperature=0.7, max_tokens=100, timeout=30, max_retries=2
    )


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def engine(config: LLMConfig, fake_model: FakeModel) -> ChatEngine:
    return ChatEngine(config, model=fake_model)


# ------------------------------------------------------------------
# Initialisation tests
# ------------------------------------------------------------------


class TestInit:
    """Tests for ChatEngine initialisation."""

    def test_init_accepts_config_and_model(self, config: LLMConfig, fake_model: FakeModel) -> None:
        engine = ChatEngine(config, model=fake_model)
        assert engine.model is fake_model

    def test_message_count_starts_at_zero(self, engine: ChatEngine) -> None:
        assert engine.message_count == 0


# ------------------------------------------------------------------
# Chat tests
# ------------------------------------------------------------------


class TestChat:
    """Tests for ChatEngine.chat()."""

    async def test_chat_returns_response(self, engine: ChatEngine) -> None:
        resp = await engine.chat("Hi!")
        assert isinstance(resp, ChatResponse)
        assert resp.content == "Hello, world!"

    async def test_chat_returns_token_counts(self, engine: ChatEngine) -> None:
        resp = await engine.chat("Hi!")
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.total_tokens == 15

    async def test_chat_builds_memory(self, engine: ChatEngine) -> None:
        assert engine.message_count == 0
        await engine.chat("First message")
        assert engine.message_count == 2  # 1 Human + 1 AI

    async def test_chat_includes_system_prompt(
        self, engine: ChatEngine, fake_model: FakeModel
    ) -> None:
        await engine.chat("Hello", system_prompt="You are helpful.")
        assert fake_model.last_messages is not None
        # First message should be a system message
        assert fake_model.last_messages[0].type == "system"
        assert fake_model.last_messages[0].content == "You are helpful."

    async def test_chat_accumulates_history(self, engine: ChatEngine) -> None:
        await engine.chat("Q1")
        await engine.chat("Q2")
        # 4 messages: Q1, A1, Q2, A2
        assert engine.message_count == 4


# ------------------------------------------------------------------
# Streaming tests
# ------------------------------------------------------------------


class TestStreamChat:
    """Tests for ChatEngine.stream_chat()."""

    async def test_stream_yields_tokens(self, engine: ChatEngine, fake_model: FakeModel) -> None:
        fake_model._response = "ABC"
        tokens = []
        async for token in engine.stream_chat("Hi"):
            tokens.append(token)
        assert tokens == ["A", "B", "C"]

    async def test_stream_builds_memory(self, engine: ChatEngine) -> None:
        async for _ in engine.stream_chat("Stream me"):
            pass
        assert engine.message_count == 2

    async def test_stream_multiple_turns(self, engine: ChatEngine) -> None:
        async for _ in engine.stream_chat("T1"):
            pass
        async for _ in engine.stream_chat("T2"):
            pass
        assert engine.message_count == 4


# ------------------------------------------------------------------
# Memory tests
# ------------------------------------------------------------------


class TestMemory:
    """Tests for ChatEngine memory management."""

    async def test_clear_memory(self, engine: ChatEngine) -> None:
        await engine.chat("Hello")
        assert engine.message_count == 2
        engine.clear_memory()
        assert engine.message_count == 0

    async def test_clear_memory_then_chat(self, engine: ChatEngine) -> None:
        await engine.chat("Old")
        engine.clear_memory()
        await engine.chat("New")
        assert engine.message_count == 2


# ------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------


class TestIntegration:
    """End-to-end ChatEngine scenarios."""

    async def test_full_conversation_flow(self, config: LLMConfig) -> None:
        """Simulate a real conversation with FakeModel."""
        model = FakeModel(response="Sure, I can help with that!")
        engine = ChatEngine(config, model=model)

        resp1 = await engine.chat("Can you help me?")
        assert resp1.content == "Sure, I can help with that!"
        assert resp1.total_tokens == 15
        assert engine.message_count == 2

        await engine.chat("What about this?")
        assert engine.message_count == 4

        engine.clear_memory()
        assert engine.message_count == 0

    async def test_streaming_full_conversation(self, config: LLMConfig) -> None:
        model = FakeModel(response="Hi!")
        engine = ChatEngine(config, model=model)

        collected: list[str] = []
        async for token in engine.stream_chat("Hello", system_prompt="Be friendly."):
            collected.append(token)

        assert "".join(collected) == "Hi!"
        assert engine.message_count == 2
