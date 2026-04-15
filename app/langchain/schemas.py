from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class LangChainQAOutput(BaseModel):
    """
    LangChain 链输出的结构化结果
    """
    answer: str = Field(..., description="最终回答内容")
    is_answerable: bool = Field(..., description="是否可回答")
    confidence: Literal["high", "medium", "low"] = Field(..., description="置信度")
    sources: List[str] = Field(default_factory=list, description="实际使用到的来源文件名")
    refusal_reason: Optional[str] = Field(default=None, description="拒答原因")