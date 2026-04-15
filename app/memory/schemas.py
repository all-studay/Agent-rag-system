from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


MemoryType = Literal["project_fact", "user_preference", "progress_update", "other"]


class MemoryItem(BaseModel):
    """
    一条长期记忆（L2）。
    """
    memory_id: str = Field(..., description="记忆唯一ID")
    memory_type: MemoryType = Field(..., description="记忆类型")
    content: str = Field(..., description="记忆内容")
    importance: Literal["high", "medium", "low"] = Field(..., description="重要度")
    source_session_id: Optional[str] = Field(default=None, description="来源会话ID")
    source_query: Optional[str] = Field(default=None, description="来源问题")
    created_at: str = Field(..., description="创建时间")