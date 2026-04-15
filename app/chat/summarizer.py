from __future__ import annotations

import json
from typing import List

from app.chat.memory import ChatTurn
from app.rag.llm_client import LLMClient


class ConversationSummarizer:
    """
    使用大模型对较早历史进行压缩总结。
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def summarize(self, history: List[ChatTurn]) -> str:
        """
        将历史问答压缩为简洁摘要。
        """
        if not history:
            return "无"

        history_text = self._format_history_for_summary(history)
        prompt = self._build_summary_prompt(history_text)

        raw_output = self.llm.generate(prompt)
        return self._parse_summary_output(raw_output)

    @staticmethod
    def _format_history_for_summary(history: List[ChatTurn]) -> str:
        parts = []
        for i, turn in enumerate(history, start=1):
            parts.append(
                f"[历史轮次 {i}]\n"
                f"用户：{turn.user_query}\n"
                f"助手：{turn.assistant_answer}\n"
            )
        return "\n".join(parts)

    @staticmethod
    def _build_summary_prompt(history_text: str) -> str:
        return f"""你是一个对话摘要助手。
请根据给定的历史问答，提炼出后续继续对话最有价值的信息。

摘要要求：
1. 只保留关键事实、用户目标、已确认结论、待继续追问点。
2. 删除重复、寒暄、无关细节。
3. 用简洁中文输出。
4. 只输出 JSON，不要输出额外说明，不要输出 Markdown 代码块。

JSON 格式必须严格如下：
{{
  "summary": "这里填写压缩后的历史摘要"
}}

历史问答如下：
{history_text}

现在请只输出 JSON：
"""

    @staticmethod
    def _parse_summary_output(raw_output: str) -> str:
        text = raw_output.strip()

        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").strip()
            if text.endswith("```"):
                text = text[:-3].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 如果模型偶尔没严格输出 JSON，退化成原始文本
            return raw_output.strip()

        summary = data.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return raw_output.strip()

        return summary.strip()