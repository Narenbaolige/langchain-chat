"""Message data model."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """A single chat message within a session.

    Used by SessionManager for business-layer data transfer.
    Never constructed directly from SQL — always goes through StorageBackend.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="Unique message identifier")
    session_id: int = Field(description="Owning session ID")
    role: str = Field(description="Message role: user, assistant, or system")
    content: str = Field(description="Message body")
    created_at: datetime = Field(description="Creation timestamp")
