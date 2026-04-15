from __future__ import annotations

import json
from pathlib import Path
from typing import List

from app.memory.schemas import MemoryItem


class JsonMemoryStore:
    """
    使用本地 JSON 文件保存长期记忆（L2）。
    第一版特点：
    - 简单
    - 可调试
    - 适合本地开发
    """

    def __init__(self, file_path: str = "data/memory/l2_memory.json") -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.file_path.exists():
            self._write_all([])

    def load_all(self) -> List[MemoryItem]:
        with self.file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [MemoryItem(**item) for item in data]

    def save(self, memory: MemoryItem) -> None:
        memories = self.load_all()
        memories.append(memory)
        self._write_all(memories)

    def list_memories(self) -> List[MemoryItem]:
        return self.load_all()

    def clear(self) -> None:
        self._write_all([])

    def _write_all(self, memories: List[MemoryItem | dict]) -> None:
        serialized = []
        for item in memories:
            if isinstance(item, MemoryItem):
                serialized.append(item.model_dump())
            else:
                serialized.append(item)

        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)