from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class GraphRAGState(TypedDict, total=False):
    """
    LangGraph 中流转的共享状态
    """

    # 输入
    query: str
    top_k: int
    session_id: Optional[str]

    # 中间结果
    conversation_context: Dict[str, str]
    relevant_memories: List[Dict[str, Any]]
    retrieval_results: List[Dict[str, Any]]
    should_refuse_early: bool

    # LangChain / LLM 输出
    parsed_output: Dict[str, Any]

    # 最终结果
    final_response: Dict[str, Any]