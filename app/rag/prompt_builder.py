from __future__ import annotations

from typing import List, Dict


class PromptBuilder:
    """
    用于构造带长期记忆、摘要和最近历史的结构化 RAG Prompt。
    """

    @staticmethod
    def build_context(results: List[Dict]) -> str:
        if not results:
            return "无"

        context_parts = []

        for i, item in enumerate(results, start=1):
            source = item.get("source", "unknown")
            chunk_index = item.get("chunk_index", "unknown")
            text = item.get("text", "").strip()

            context_parts.append(
                f"[证据 {i}]\n"
                f"来源文件: {source}\n"
                f"Chunk编号: {chunk_index}\n"
                f"内容:\n{text}\n"
            )

        return "\n".join(context_parts)

    @staticmethod
    def build_memory_context(memories: List[Dict]) -> str:
        if not memories:
            return "无"

        memory_parts = []
        for i, item in enumerate(memories, start=1):
            memory_parts.append(
                f"[长期记忆 {i}]\n"
                f"类型: {item.get('memory_type', 'other')}\n"
                f"重要度: {item.get('importance', 'low')}\n"
                f"内容: {item.get('content', '')}\n"
            )

        return "\n".join(memory_parts)

    @classmethod
    def build_qa_prompt(
        cls,
        query: str,
        results: List[Dict],
        conversation_summary: str = "无",
        recent_history: str = "无",
        long_term_memories: List[Dict] | None = None,
    ) -> str:
        context = cls.build_context(results)
        memory_context = cls.build_memory_context(long_term_memories or [])

        prompt = f"""你是一个企业知识库问答助手。
请严格根据“长期记忆”“历史摘要”“最近对话”和“已检索到的证据”回答用户问题，不要凭空编造。

信息使用优先级：
1. 已检索到的证据
2. 长期记忆
3. 历史摘要和最近对话
如果它们之间有冲突，优先相信“已检索到的证据”。

你的任务：
1. 优先结合长期记忆、历史摘要和最近对话理解当前问题中的省略、代词和追问关系。
2. 再根据已检索到的证据回答问题。
3. 如果证据不足，请明确表示无法根据现有知识库确定答案。
4. 不允许根据常识补全未出现的信息。
5. 你必须只输出合法 JSON，不能输出额外说明，不能输出 Markdown 代码块。

JSON 输出格式必须严格如下：
{{
  "answer": "最终回答内容",
  "is_answerable": true,
  "confidence": "high",
  "sources": ["文件1", "文件2"],
  "refusal_reason": null
}}

字段要求：
- answer: 字符串，最终回答
- is_answerable: 布尔值，true 或 false
- confidence: 只能是 "high" / "medium" / "low"
- sources: 字符串列表，填写实际使用到的来源文件名，去重
- refusal_reason:
  - 如果 is_answerable=true，填 null
  - 如果 is_answerable=false，填写不能回答的原因，例如"知识库未提供明确答案"

长期记忆：
{memory_context}

历史摘要：
{conversation_summary}

最近对话：
{recent_history}

当前用户问题：
{query}

已检索到的证据：
{context}

现在请只输出 JSON：
"""
        return prompt