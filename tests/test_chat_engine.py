"""Tests for ChatEngine — the LangChain-powered chat engine.

All tests use a FakeModel so no real API key or network is required.
FakeModel is defined in conftest.py and shared across test modules.
"""

from __future__ import annotations

import pytest
from fakes import FakeModel  # noqa: F401 — used in type annotations
from langchain_core.messages import AIMessage, HumanMessage

from langchain_chat.core.chat_engine import ChatEngine, ChatResponse

# ------------------------------------------------------------------
# Local fixtures (FakeModel and llm_config come from conftest.py)
# ------------------------------------------------------------------


@pytest.fixture
def engine(fake_model: FakeModel) -> ChatEngine:
    """Return a ChatEngine wired to FakeModel."""
    return ChatEngine(model=fake_model)


# ------------------------------------------------------------------
# Initialisation tests
# ------------------------------------------------------------------


class TestInit:
    """Tests for ChatEngine initialisation."""

    def test_init_accepts_model_parameter(self, fake_model: FakeModel) -> None:
        engine = ChatEngine(model=fake_model)
        assert engine.model is fake_model

    def test_set_model_replaces_model(self, fake_model: FakeModel) -> None:
        engine = ChatEngine(model=fake_model)
        new_model = FakeModel(response="different")
        engine.set_model(new_model)
        assert engine.model is new_model

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

    async def test_full_conversation_flow(self) -> None:
        """Simulate a real conversation with FakeModel."""
        model = FakeModel(response="Sure, I can help with that!")
        engine = ChatEngine(model=model)

        resp1 = await engine.chat("Can you help me?")
        assert resp1.content == "Sure, I can help with that!"
        assert resp1.total_tokens == 15
        assert engine.message_count == 2

        await engine.chat("What about this?")
        assert engine.message_count == 4

        engine.clear_memory()
        assert engine.message_count == 0

    async def test_streaming_full_conversation(self) -> None:
        model = FakeModel(response="Hi!")
        engine = ChatEngine(model=model)

        collected: list[str] = []
        async for token in engine.stream_chat("Hello", system_prompt="Be friendly."):
            collected.append(token)

        assert "".join(collected) == "Hi!"
        assert engine.message_count == 2


# ------------------------------------------------------------------
# Step 8 — load_messages tests
# ------------------------------------------------------------------


class TestLoadMessages:
    """Tests for ChatEngine.load_messages()."""

    async def test_load_messages_replaces_memory(self) -> None:
        engine = ChatEngine(model=FakeModel(response="OK"))

        history = [
            HumanMessage(content="Old Q1"),
            AIMessage(content="Old A1"),
            HumanMessage(content="Old Q2"),
            AIMessage(content="Old A2"),
        ]
        engine.load_messages(history)
        assert engine.message_count == 4

    async def test_load_messages_overwrites_previous(self) -> None:
        engine = ChatEngine(model=FakeModel(response="OK"))
        await engine.chat("Hello")
        assert engine.message_count == 2

        # Overwrite with history
        history = [HumanMessage(content="History Q"), AIMessage(content="History A")]
        engine.load_messages(history)
        assert engine.message_count == 2

    async def test_load_messages_then_chat(self) -> None:
        engine = ChatEngine(model=FakeModel(response="New"))
        engine.load_messages([HumanMessage(content="Prior"), AIMessage(content="Prior A")])
        assert engine.message_count == 2

        # New message appended to loaded history
        resp = await engine.chat("New Q")
        assert engine.message_count == 4  # Prior Q, Prior A, New user, New AI
        assert resp.content == "New"

    async def test_load_empty_messages(self) -> None:
        engine = ChatEngine(model=FakeModel())
        await engine.chat("Hello")
        assert engine.message_count == 2
        engine.load_messages([])
        assert engine.message_count == 0


# ------------------------------------------------------------------
# Edge case & error tests
# ------------------------------------------------------------------


class TestChatEngineErrors:
    """Edge cases and error conditions."""

    def test_no_model_raises(self) -> None:
        engine = ChatEngine()
        with pytest.raises(RuntimeError, match="No model configured"):
            _ = engine.model

    async def test_no_model_raises_async(self) -> None:
        engine = ChatEngine()
        with pytest.raises(RuntimeError, match="No model configured"):
            await engine.chat("hello")

    async def test_chat_empty_message(self, engine: ChatEngine) -> None:
        """Empty user message should still work (model decides handling)."""
        resp = await engine.chat("")
        assert isinstance(resp, ChatResponse)

    async def test_stream_empty_message(self, engine: ChatEngine) -> None:
        tokens: list[str] = []
        async for t in engine.stream_chat(""):
            tokens.append(t)
        # Should complete without error; FakeModel returns its default response
        assert engine.message_count == 2

    async def test_chat_with_only_whitespace(self, engine: ChatEngine) -> None:
        resp = await engine.chat("   ")
        assert isinstance(resp, ChatResponse)

    async def test_long_message(self, engine: ChatEngine) -> None:
        long_msg = "Hello " * 500
        resp = await engine.chat(long_msg)
        assert isinstance(resp, ChatResponse)

    async def test_system_prompt_preserved_in_memory(self, engine: ChatEngine) -> None:
        """System prompt goes to the model but not into conversation memory."""
        await engine.chat("Q", system_prompt="You are helpful.")
        # Memory has 2 messages: Human(Q) + AI(response), NOT the system prompt
        assert engine.message_count == 2

    def test_set_model_clears_previous(self, fake_model: FakeModel) -> None:
        engine = ChatEngine(model=fake_model)
        new_model = FakeModel(response="New")
        engine.set_model(new_model)
        assert engine.model is new_model
