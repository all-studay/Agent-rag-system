from __future__ import annotations

from typing import List

from app.chat.memory import ChatTurn


class HistoryFormatter:
    """
    将历史对话格式化为可注入 Prompt 的文本。
    """

    @staticmethod
    def format_history(history: List[ChatTurn]) -> str:
        if not history:
            return "无"

        parts = []
        for i, turn in enumerate(history, start=1):
            parts.append(
                f"[历史轮次 {i}]\n"
                f"用户：{turn.user_query}\n"
                f"助手：{turn.assistant_answer}\n"
            )

        return "\n".join(parts)