"""Chat Engine — Core Business Layer module.

ChatEngine wraps a ChatModel with conversation memory, streaming, and
token counting.  It does NOT own or create models — the model instance
is injected via :meth:`set_model` (or the constructor) by the caller
(typically a :class:`ModelManager` in production, or ``FakeModel`` in
tests).

Model lifecycle management belongs to :class:`ModelManager`; ChatEngine
focuses exclusively on chat, memory, streaming, and token counting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


@dataclass
class ChatResponse:
    """Result of a single chat turn."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ChatEngine:
    """Async-first chat engine backed by an injected ChatModel.

    Maintains an in-memory conversation history (not persisted to DB).
    Supports both one-shot ``chat()`` and streaming ``stream_chat()``.

    Usage::

        engine = ChatEngine(model=some_chat_openai_instance)
        resp = await engine.chat("Hello!")
        print(resp.content)

        # Runtime model switching:
        engine.set_model(another_model)
    """

    def __init__(self, *, model: Any = None) -> None:
        """Initialise the engine.

        Args:
            model: The ChatModel instance to use.  Can be set later via
                   :meth:`set_model`.  If omitted, calling :meth:`chat`
                   before a model is set raises ``RuntimeError``.
        """
        self._model: Any = model
        self._messages: list[BaseMessage] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_model(self, model: Any) -> None:
        """Replace the underlying ChatModel (for runtime model switching).

        Conversation memory is preserved across model changes.
        """
        self._model = model

    async def chat(
        self, message: str, system_prompt: str | None = None, *, max_tokens: int = 0
    ) -> ChatResponse:
        """Send a message and receive a complete response.

        Args:
            message: The user message.
            system_prompt: Optional system-level instruction.
            max_tokens: If > 0, trim conversation context to this token limit.

        Returns:
            ChatResponse with content, token counts.
        """
        llm_messages = self._build_messages(message, system_prompt)
        if max_tokens > 0:
            llm_messages = self.trim_messages(llm_messages, max_tokens)
        response = await self.model.ainvoke(llm_messages)

        self._messages.append(HumanMessage(content=message))
        self._messages.append(AIMessage(content=response.content))

        usage = _extract_usage(response)
        return ChatResponse(
            content=response.content,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    async def stream_chat(
        self, message: str, system_prompt: str | None = None, *, max_tokens: int = 0
    ) -> AsyncIterator[str]:
        """Send a message and stream the response token by token.

        Args:
            message: The user message.
            system_prompt: Optional system-level instruction.
            max_tokens: If > 0, trim conversation context to this token limit.

        Yields:
            Content chunks as they arrive from the model.
        """
        llm_messages = self._build_messages(message, system_prompt)
        if max_tokens > 0:
            llm_messages = self.trim_messages(llm_messages, max_tokens)
        full_content = ""

        async for chunk in self.model.astream(llm_messages):
            chunk_content: str = chunk.content
            if chunk_content:
                full_content += chunk_content
                yield chunk_content

        self._messages.append(HumanMessage(content=message))
        self._messages.append(AIMessage(content=full_content))

    def clear_memory(self) -> None:
        """Reset the conversation history."""
        self._messages.clear()

    def load_messages(self, messages: list[BaseMessage]) -> None:
        """Replace the in-memory conversation with *messages*.

        Used by the TUI to restore history when reopening a saved session.
        No database access — the caller is responsible for fetching and
        converting stored messages to LangChain objects.
        """
        self._messages = list(messages)

    def trim_messages(
        self, messages: list[BaseMessage], max_tokens: int
    ) -> list[BaseMessage]:
        """Trim *messages* to fit within *max_tokens* using a sliding window.

        Rules:
        1. ``SystemMessage`` instances at the front are always kept.
        2. From the end, accumulate estimated tokens until *max_tokens*.
        3. Earlier non-system messages beyond the window are dropped.

        Token estimation uses ``len(content) / 2`` as a rough heuristic
        (reasonable for mixed Chinese / English text).
        """
        if not messages:
            return []

        # Separate system messages at the front.
        system_msgs: list[BaseMessage] = []
        body_start = 0
        for m in messages:
            if m.type == "system":
                system_msgs.append(m)
                body_start += 1
            else:
                break

        body = messages[body_start:]
        if not body:
            return system_msgs

        # Walk backwards, accumulating estimated tokens.
        kept: list[BaseMessage] = []
        total = 0
        for m in reversed(body):
            total += self._estimate_tokens(m)
            if total > max_tokens and kept:  # keep at least one message
                break
            kept.insert(0, m)

        return system_msgs + kept

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> Any:
        """Return the current ChatModel instance.

        Raises:
            RuntimeError: If no model has been set.
        """
        if self._model is None:
            raise RuntimeError(
                "No model configured. Call set_model() or provide model= in constructor."
            )
        return self._model

    @property
    def message_count(self) -> int:
        """Number of messages in the conversation memory."""
        return len(self._messages)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_messages(self, message: str, system_prompt: str | None) -> list[BaseMessage]:
        """Assemble the full message list for a model call."""
        result: list[BaseMessage] = []

        if system_prompt:
            result.append(SystemMessage(content=system_prompt))

        result.extend(self._messages)
        result.append(HumanMessage(content=message))
        return result

    @staticmethod
    def _estimate_tokens(msg: BaseMessage) -> int:
        """Rough token count: ``len(content) / 2``.

        Works reasonably for mixed Chinese / English text where ~2 chars ≈ 1 token.
        """
        content = getattr(msg, "content", "") or ""
        return max(1, len(content) // 2)


def _extract_usage(response: Any) -> dict[str, int]:
    """Extract token usage from a LangChain response in a robust way.

    Different LangChain versions expose usage in slightly different
    locations, so we try multiple paths.
    """
    # Preferred: usage_metadata (newer langchain-core)
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        return {
            "input_tokens": um.get("input_tokens", 0),
            "output_tokens": um.get("output_tokens", 0),
            "total_tokens": um.get("total_tokens", 0),
        }

    # Fallback: response_metadata.token_usage (older versions)
    rm = getattr(response, "response_metadata", {}) or {}
    tu = rm.get("token_usage", {}) or {}
    return {
        "input_tokens": tu.get("prompt_tokens", 0),
        "output_tokens": tu.get("completion_tokens", 0),
        "total_tokens": tu.get("total_tokens", 0),
    }
