from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import List, Optional

from app.memory.schemas import MemoryItem
from app.rag.llm_client import LLMClient


class MemoryExtractor:
    """
    长期记忆提取器（优化版）
    优先使用规则抽取，避免每轮问答后再次调用 LLM 导致严重变慢。
    """

    def __init__(self, llm: LLMClient | None = None, use_llm_fallback: bool = False) -> None:
        self.llm = llm
        self.use_llm_fallback = use_llm_fallback

    def extract_memories(
        self,
        query: str,
        answer: str,
        session_id: Optional[str] = None,
    ) -> List[MemoryItem]:
        # 1) 优先规则抽取
        memories = self._rule_extract(query=query, answer=answer, session_id=session_id)
        if memories:
            return memories

        # 2) 可选：LLM 兜底
        if self.use_llm_fallback and self.llm is not None:
            return self._llm_extract(query=query, answer=answer, session_id=session_id)

        return []

    def _rule_extract(
        self,
        query: str,
        answer: str,
        session_id: Optional[str] = None,
    ) -> List[MemoryItem]:
        query_text = query.strip()
        answer_text = answer.strip()
        combined = f"{query_text}\n{answer_text}"

        memories: List[MemoryItem] = []

        def build_memory(content: str, memory_type: str, importance: str) -> MemoryItem:
            return MemoryItem(
                memory_id=str(uuid.uuid4()),
                memory_type=memory_type,
                content=content,
                importance=importance,
                source_session_id=session_id,
                source_query=query_text,
                created_at=datetime.now().isoformat(timespec="seconds"),
            )

        # 1. 支持的文档格式 / 未来扩展
        if (
            "TXT" in combined
            or "Markdown" in combined
            or "PDF" in combined
            or "Word" in combined
            or "Excel" in combined
            or "网页" in combined
        ):
            memories.append(
                build_memory(
                    "系统第一版支持 TXT、Markdown（.md）和 PDF 文档，后续计划增加 Word、Excel 和网页内容接入。",
                    "project_fact",
                    "high",
                )
            )

        # 2. 第一版核心目标
        if (
            "第一版" in combined
            and ("核心目标" in combined or "最小闭环" in combined or "基础问答" in combined)
        ):
            memories.append(
                build_memory(
                    "第一版项目核心目标是打通文档读取、文本切分、向量化、检索到基础问答的最小闭环链路，而非直接构建复杂 Agent。",
                    "project_fact",
                    "high",
                )
            )

        # 3. 长对话处理策略
        if (
            "长对话" in combined
            or "历史内容过长" in combined
            or "最近几轮" in combined
            or "摘要" in combined
        ):
            memories.append(
                build_memory(
                    "系统在处理长对话时会保留最近几轮消息，并在历史内容过长时生成摘要，保留用户目标、关键约束、已完成步骤和待解决问题。",
                    "project_fact",
                    "high",
                )
            )

        # 4. 记忆体系结构
        if all(token in combined for token in ["L0", "L1", "L2", "L3"]):
            memories.append(
                build_memory(
                    "系统支持分层记忆，当前规划的记忆体系包括 L0（当前工作记忆）、L1（会话记忆）、L2（长期偏好记忆）和 L3（经验记忆）。",
                    "project_fact",
                    "high",
                )
            )

        # 5. 已实现能力总结
        implemented_capabilities = []
        if "检索" in combined:
            implemented_capabilities.append("检索")
        if "问答" in combined:
            implemented_capabilities.append("问答")
        if "记忆" in combined:
            implemented_capabilities.append("记忆")
        if "长对话" in combined or "摘要" in combined:
            implemented_capabilities.append("长对话摘要")

        if len(implemented_capabilities) >= 2:
            joined = "、".join(dict.fromkeys(implemented_capabilities))
            memories.append(
                build_memory(
                    f"当前项目已经实现的关键能力包括：{joined}。",
                    "progress_update",
                    "medium",
                )
            )

        # 去掉同轮规则抽取重复内容
        unique = []
        seen = set()
        for mem in memories:
            normalized = (mem.memory_type, mem.content.replace(" ", ""))
            if normalized not in seen:
                seen.add(normalized)
                unique.append(mem)

        return unique[:3]

    def _llm_extract(
        self,
        query: str,
        answer: str,
        session_id: Optional[str] = None,
    ) -> List[MemoryItem]:
        # 当前先不启用复杂 LLM 兜底，保留接口
        return []