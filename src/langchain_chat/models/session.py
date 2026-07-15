"""Session data model."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Session(BaseModel):
    """Represents a chat session belonging to a user.

    ``preset_id`` and ``updated_at`` are reserved for future steps
    (Step 8 session enhancements, Step 9 UX optimisation).
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="Unique session identifier")
    user_id: int = Field(description="Owner user ID")
    preset_id: int | None = Field(default=None, description="Bound preset (reserved)")
    title: str = Field(default="", description="Human-readable session title")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last-modified timestamp (reserved)")
