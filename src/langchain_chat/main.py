"""langchain-chat — Enterprise-grade AI ChatBot framework."""

import sys

from langchain_chat.core.config_manager import ConfigManager


def main() -> None:
    """Entry point for langchain-chat."""
    cm = ConfigManager()
    config = cm.load()

    print(f"{config.app.name} initialized")
    print(f"Environment: {config.app.env}")
    print(f"Python {sys.version}")


if __name__ == "__main__":
    main()
