"""Tests for main entry point."""

from langchain_chat.main import main


def test_main_runs(capsys) -> None:
    """Verify main() exits cleanly and prints version."""
    main()
    captured = capsys.readouterr()
    assert "langchain-chat initialized" in captured.out
    assert "Environment: dev" in captured.out
