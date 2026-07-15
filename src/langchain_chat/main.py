"""langchain-chat — Enterprise-grade AI ChatBot framework."""

from __future__ import annotations

import asyncio
import logging

from langchain_chat.core.chat_engine import ChatEngine
from langchain_chat.core.config_manager import get_config
from langchain_chat.core.model_manager import ModelManager
from langchain_chat.core.prompt_manager import PromptManager
from langchain_chat.core.session_manager import SessionManager
from langchain_chat.core.user_manager import DuplicateUserError, UserManager
from langchain_chat.logging_config import setup_logging
from langchain_chat.storage.factory import StorageFactory
from langchain_chat.ui.app import TuiChatApp

logger = logging.getLogger(__name__)


async def _async_main() -> None:
    """Async entry point: initialise storage + UserManager, create demo user."""
    config = get_config()
    setup_logging(level=config.logging.level, log_file="logs/app.log")

    storage = StorageFactory.create(config.storage)
    await storage.initialize()

    try:
        manager = UserManager(storage)

        # Try to create the default user; skip if it already exists
        try:
            user = await manager.create_user("default")
            print(f"[UserManager] Created user: id={user.id} username={user.username!r}")
        except DuplicateUserError:
            user = await manager.get_user_by_name("default")
            print(f"[UserManager] Found existing user: id={user.id} username={user.username!r}")

        all_users = await manager.list_users()
        print(f"[UserManager] Total users: {len(all_users)}")

    finally:
        await storage.close()

    print(f"{config.app.name} initialized")
    print(f"Environment: {config.app.env}")


def main() -> None:
    """Synchronous entry point (bootstrap-only demo)."""
    asyncio.run(_async_main())


# ------------------------------------------------------------------
# TUI entry point
# ------------------------------------------------------------------


async def _async_tui() -> None:
    """Initialise all managers and launch the TUI chat application."""
    config = get_config()
    setup_logging(level=config.logging.level, log_file="logs/app.log")
    logger.info("Starting TUI chat...")

    storage = StorageFactory.create(config.storage)
    await storage.initialize()

    try:
        user_manager = UserManager(storage)
        prompt_manager = PromptManager(storage)
        session_manager = SessionManager(storage)

        model_manager = ModelManager(config.llm)
        engine = ChatEngine(model=model_manager.get_current_model())

        app = TuiChatApp(user_manager, prompt_manager, session_manager, model_manager, engine)
        await app.run()

    finally:
        await storage.close()


def main_tui() -> None:
    """Synchronous entry point for the TUI chat."""
    asyncio.run(_async_tui())


if __name__ == "__main__":
    main()
