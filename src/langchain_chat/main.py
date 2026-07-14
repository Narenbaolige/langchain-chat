"""langchain-chat — Enterprise-grade AI ChatBot framework."""

from __future__ import annotations

import asyncio

from langchain_chat.core.config_manager import get_config
from langchain_chat.core.user_manager import DuplicateUserError, UserManager
from langchain_chat.storage.factory import StorageFactory


async def _async_main() -> None:
    """Async entry point: initialise storage + UserManager, create demo user."""
    config = get_config()

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
    """Synchronous entry point."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
