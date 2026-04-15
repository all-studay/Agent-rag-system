from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

class RetrievalSource(BaseModel):
    """
        单条检索来源信息。
    """
    source: str = Field(...,description="来源文件名")
    chunk_id: str = Field(...,description="chunk 唯一标识")
    chunk_index: int = Field(...,description="chunk 序号")
    distance: float = Field(...,description="向量距离")
    preview: str = Field(...,description="证据预览文本")


class QAResponse(BaseModel):
    """
        结构化问答结果。
    """
    query: str = Field(..., description="用户问题")
    answer: str = Field(..., description="最终回答")
    is_answerable: bool = Field(..., description="是否根据当前知识库回答")
    confidence: Literal["high", "medium", "low"] = Field(..., description="回答置信度")
    sources: List[str] = Field(default_factory=list, description="去重后的来源文件名列表")
    retrieval_results: List[RetrievalSource] = Field(default_factory=list, description="检索结果列表")
    refusal_reason: Optional[str] = Field(default=None, description="当无法回答时的原因说明")


