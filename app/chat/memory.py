from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class ChatTurn:
    """
    单轮对话记录。
    """
    user_query: str
    assistant_answer: str

    def to_dict(self) -> dict:
        return asdict(self)


class InMemoryChatStore:
    """
    基础版会话存储。
    使用内存字典按 session_id 保存最近若干轮问答。

    当前版本：
    - 适合本地开发和单进程测试
    - 服务重启后历史会丢失
    - 后续可替换成 Redis / 数据库
    """

    def __init__(self, max_turns: int = 12) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns 必须大于 0")

        self.max_turns = max_turns
        self.store: Dict[str, List[ChatTurn]] = {}

    def get_history(self, session_id: str) -> List[ChatTurn]:
        return list(self.store.get(session_id, []))

    def append_turn(self, session_id: str, user_query: str, assistant_answer: str) -> None:
        if not session_id.strip():
            raise ValueError("session_id 不能为空")

        session_id = session_id.strip()
        history = self.store.get(session_id, [])

        history.append(
            ChatTurn(
                user_query=user_query,
                assistant_answer=assistant_answer,
            )
        )

        if len(history) > self.max_turns:
            history = history[-self.max_turns:]

        self.store[session_id] = history

    def clear_history(self, session_id: str) -> None:
        session_id = session_id.strip()
        if session_id in self.store:
            del self.store[session_id]

    def list_sessions(self) -> List[str]:
        return list(self.store.keys())