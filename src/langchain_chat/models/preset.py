"""Preset (prompt template) data model."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PromptType = Literal["system", "user"]


class Preset(BaseModel):
    """A reusable prompt preset / template.

    Used by PromptManager for business-layer data transfer.
    Never constructed directly from SQL — always goes through StorageBackend.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="Unique preset identifier")
    name: str = Field(description="Unique preset name", min_length=1, max_length=128)
    content: str = Field(description="Prompt template text", min_length=1)
    prompt_type: PromptType = Field(default="user", description="system or user prompt")
    created_at: datetime = Field(description="Creation timestamp")
