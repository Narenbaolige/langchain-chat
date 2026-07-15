"""Tests for Step 16a security features — input limits, context trimming, API key masking."""

from __future__ import annotations

import json
import logging

from fakes import FakeModel

from langchain_chat.core.chat_engine import ChatEngine
from langchain_chat.core.config_models import SecurityConfig
from langchain_chat.core.json_formatter import JsonFormatter, mask_api_key

# ------------------------------------------------------------------
# API key masking tests
# ------------------------------------------------------------------


class TestMaskApiKey:
    def test_masks_standard_key(self) -> None:
        result = mask_api_key("Using key sk-abcdef1234567890 for auth")
        assert "sk-abcde..." in result
        assert "sk-abcdef1234567890" not in result

    def test_masks_multiple_keys(self) -> None:
        result = mask_api_key("Keys: sk-aaaaaaaaaa and sk-bbbbbbbbbb")
        assert result.count("sk-") == 2

    def test_no_key_unchanged(self) -> None:
        text = "Hello world, no secrets here"
        assert mask_api_key(text) == text

    def test_short_key_not_masked(self) -> None:
        """sk- followed by fewer than 10 chars is not a real key."""
        text = "prefix sk-short only"
        assert mask_api_key(text) == text


# ------------------------------------------------------------------
# JsonFormatter tests
# ------------------------------------------------------------------


class TestJsonFormatter:
    def test_formats_valid_json(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=1,
            msg="Hello world", args=(), exc_info=None,
        )
        record.created = 1750000000.0
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["module"] == "test"
        assert parsed["message"] == "Hello world"
        assert "time" in parsed

    def test_masks_key_in_message(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=1,
            msg="Key: sk-abcdef1234567890 used", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "sk-abcde..." in parsed["message"]
        assert "sk-abcdef1234567890" not in parsed["message"]


# ------------------------------------------------------------------
# trim_messages tests
# ------------------------------------------------------------------


class TestTrimMessages:
    """Tests for ChatEngine.trim_messages()."""

    def test_empty_list(self) -> None:
        engine = ChatEngine()
        result = engine.trim_messages([], 100)
        assert result == []

    def test_system_messages_preserved(self) -> None:
        from langchain_core.messages import HumanMessage, SystemMessage

        engine = ChatEngine()
        msgs = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello"),
            HumanMessage(content="World"),
        ]
        # max_tokens=1 is tiny — only system should remain
        result = engine.trim_messages(msgs, 1)
        assert len(result) >= 1
        assert result[0].type == "system"

    def test_keeps_recent_messages(self) -> None:
        from langchain_core.messages import HumanMessage

        engine = ChatEngine()
        msgs = [HumanMessage(content=f"Message number {i}") for i in range(100)]
        # Tight limit should keep only the tail
        result = engine.trim_messages(msgs, 10)
        assert len(result) < 100
        # Most recent messages are at the end
        assert "Message number 99" in result[-1].content

    def test_zero_max_tokens_keeps_one(self) -> None:
        from langchain_core.messages import HumanMessage

        engine = ChatEngine()
        msgs = [HumanMessage(content="First"), HumanMessage(content="Last")]
        result = engine.trim_messages(msgs, 0)
        assert len(result) >= 1

    def test_large_limit_keeps_all(self) -> None:
        from langchain_core.messages import HumanMessage

        engine = ChatEngine()
        msgs = [HumanMessage(content=f"Msg {i}") for i in range(10)]
        result = engine.trim_messages(msgs, 99999)
        assert len(result) == 10


# ------------------------------------------------------------------
# SecurityConfig tests
# ------------------------------------------------------------------


class TestSecurityConfig:
    def test_defaults(self) -> None:
        cfg = SecurityConfig()
        assert cfg.max_input_length == 5000
        assert cfg.context_max_tokens == 4000

    def test_custom_values(self) -> None:
        cfg = SecurityConfig(max_input_length=1000, context_max_tokens=2000)
        assert cfg.max_input_length == 1000
        assert cfg.context_max_tokens == 2000


# ------------------------------------------------------------------
# ChatEngine context trimming integration tests
# ------------------------------------------------------------------


class TestChatEngineWithTrimming:
    """ChatEngine integration: chat() and stream_chat() with max_tokens."""

    async def test_chat_with_trimming(self) -> None:
        """chat() with max_tokens should complete without error."""
        engine = ChatEngine(model=FakeModel(response="OK"))
        # Build up some history
        for i in range(20):
            await engine.chat(f"Message {i}")
        # Now chat with tight context limit
        resp = await engine.chat("Final message", max_tokens=50)
        assert resp.content == "OK"  # FakeModel(response="OK")

    async def test_stream_with_trimming(self) -> None:
        """stream_chat() with max_tokens should complete without error."""
        engine = ChatEngine(model=FakeModel(response="OK"))
        for i in range(20):
            await engine.chat(f"Message {i}")
        tokens: list[str] = []
        async for t in engine.stream_chat("Final stream", max_tokens=50):
            tokens.append(t)
        assert len(tokens) > 0
