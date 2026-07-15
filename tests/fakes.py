"""Shared fake/mock objects for testing.

These are in a separate module (not conftest.py) so they can be imported
by both conftest fixtures and individual test files.
"""

from __future__ import annotations

from unittest.mock import MagicMock


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
