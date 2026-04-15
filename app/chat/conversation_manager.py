from __future__ import annotations

from typing import Optional

from app.chat.history_formatter import HistoryFormatter
from app.chat.memory import InMemoryChatStore
from app.chat.summarizer import ConversationSummarizer


class ConversationManager:
    """
    负责管理会话历史、摘要生成和上下文压缩策略。
    """

    def __init__(
        self,
        chat_store: InMemoryChatStore,
        summarizer: ConversationSummarizer,
        recent_turns_to_keep: int = 2,
        summary_trigger_turns: int = 5,
    ) -> None:
        if recent_turns_to_keep <= 0:
            raise ValueError("recent_turns_to_keep 必须大于 0")
        if summary_trigger_turns <= recent_turns_to_keep:
            raise ValueError("summary_trigger_turns 必须大于 recent_turns_to_keep")

        self.chat_store = chat_store
        self.summarizer = summarizer
        self.recent_turns_to_keep = recent_turns_to_keep
        self.summary_trigger_turns = summary_trigger_turns

    def build_context(self, session_id: Optional[str]) -> dict:
        """
        返回当前会话需要注入 Prompt 的上下文：
        - summary
        - recent_history
        """
        if not session_id or not session_id.strip():
            return {
                "conversation_summary": "无",
                "recent_history": "无",
            }

        session_id = session_id.strip()
        history = self.chat_store.get_history(session_id)

        if not history:
            return {
                "conversation_summary": "无",
                "recent_history": "无",
            }

        # 历史较短：不做摘要，直接全部作为 recent history
        if len(history) <= self.summary_trigger_turns:
            return {
                "conversation_summary": "无",
                "recent_history": HistoryFormatter.format_history(history),
            }

        # 历史较长：前面做摘要，后面保留 recent turns
        old_history = history[:-self.recent_turns_to_keep]
        recent_history = history[-self.recent_turns_to_keep:]

        summary = self.summarizer.summarize(old_history)
        recent_history_text = HistoryFormatter.format_history(recent_history)

        return {
            "conversation_summary": summary,
            "recent_history": recent_history_text,
        }