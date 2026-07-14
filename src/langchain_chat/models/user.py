"""User data model."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class User(BaseModel):
    """Represents a single user in the system.

    Used by UserManager for business-layer data transfer.
    Never constructed directly from SQL — always goes through StorageBackend.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="Unique user identifier")
    username: str = Field(description="Unique username", min_length=1, max_length=128)
    created_at: datetime = Field(description="Account creation timestamp")
