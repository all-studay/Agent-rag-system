from __future__ import annotations

import re
from typing import List, Optional

from app.memory.extractor import MemoryExtractor
from app.memory.schemas import MemoryItem
from app.memory.store import JsonMemoryStore


class MemoryManager:
    def __init__(
        self,
        store: JsonMemoryStore,
        extractor: MemoryExtractor,
    ) -> None:
        self.store = store
        self.extractor = extractor

    def extract_and_save(
        self,
        query: str,
        answer: str,
        session_id: Optional[str] = None,
    ) -> List[MemoryItem]:
        new_memories = self.extractor.extract_memories(
            query=query,
            answer=answer,
            session_id=session_id,
        )

        print(f"[memory] extracted: {len(new_memories)}")

        saved: List[MemoryItem] = []
        existing = self.store.list_memories()

        for mem in new_memories:
            if not self._is_duplicate(mem, existing):
                self.store.save(mem)
                saved.append(mem)
                existing.append(mem)

        print(f"[memory] saved: {len(saved)}")
        return saved

    def list_memories(self) -> List[MemoryItem]:
        return self.store.list_memories()

    def clear_memories(self) -> None:
        self.store.clear()

    def retrieve_relevant_memories(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.5,
    ) -> List[MemoryItem]:
        cleaned_query = query.strip().lower()
        if not cleaned_query:
            return []

        memories = self.store.list_memories()
        if not memories:
            return []

        query_tokens = self._simple_tokenize(cleaned_query)

        scored: List[tuple[MemoryItem, float]] = []
        for memory in memories:
            score = self._score_memory(query_tokens, cleaned_query, memory)
            if score >= min_score:
                scored.append((memory, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored[:top_k]]

    def _score_memory(
        self,
        query_tokens: set[str],
        cleaned_query: str,
        memory: MemoryItem,
    ) -> float:
        content_text = memory.content.lower()
        content_tokens = self._simple_tokenize(content_text)

        overlap = len(query_tokens & content_tokens) * 1.0

        substring_bonus = 0.0
        for token in query_tokens:
            if len(token) >= 2 and token in content_text:
                substring_bonus += 0.3

        source_query_bonus = 0.0
        if memory.source_query:
            sq = memory.source_query.lower()
            sq_tokens = self._simple_tokenize(sq)
            source_query_bonus += len(query_tokens & sq_tokens) * 0.3
            for token in query_tokens:
                if len(token) >= 2 and token in sq:
                    source_query_bonus += 0.2

        importance_bonus = {
            "high": 0.8,
            "medium": 0.4,
            "low": 0.1,
        }.get(memory.importance, 0.0)

        type_bonus = {
            "project_fact": 0.4,
            "progress_update": 0.3,
            "user_preference": 0.2,
            "other": 0.0,
        }.get(memory.memory_type, 0.0)

        return overlap + substring_bonus + source_query_bonus + importance_bonus + type_bonus

    @staticmethod
    def _simple_tokenize(text: str) -> set[str]:
        word_tokens = re.findall(r"[a-zA-Z0-9_]+", text)
        char_tokens = [ch for ch in text if not ch.isspace()]
        return set(word_tokens + char_tokens)

    @staticmethod
    def _is_duplicate(new_memory: MemoryItem, existing: List[MemoryItem]) -> bool:
        normalized_new = new_memory.content.replace(" ", "")
        for item in existing:
            normalized_old = item.content.replace(" ", "")
            if (
                item.memory_type == new_memory.memory_type
                and (
                    item.content == new_memory.content
                    or normalized_old == normalized_new
                )
            ):
                return True
        return False