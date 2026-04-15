from __future__ import annotations

import time
import json
from typing import Dict, Any, List, Optional

from app.chat.conversation_manager import ConversationManager
from app.chat.history_formatter import HistoryFormatter
from app.chat.memory import InMemoryChatStore
from app.chat.summarizer import ConversationSummarizer
from app.memory.extractor import MemoryExtractor
from app.memory.manager import MemoryManager
from app.memory.store import JsonMemoryStore
from app.rag.document_loader import DocumentLoader
from app.rag.embedder import TextEmbedder
from app.rag.llm_client import LLMClient
from app.rag.prompt_builder import PromptBuilder
from app.rag.schemas import QAResponse, RetrievalSource
from app.rag.text_splitter import TextSplitter, Chunk
from app.rag.vector_store import VectorStore
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker


class RAGQAService:
    """
    增强版 RAG 问答服务：
    - 结构化输出
    - 程序侧兜底
    - 混合检索 + reranker
    - 多轮历史
    - 上下文压缩 / 历史摘要
    - L2 长期记忆存储与注入
    - 记忆质量优化第一版
    """

    def __init__(
        self,
        persist_dir: str = "data/vectorstore",
        collection_name: str = "rag_chunks",
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
        reranker_model_name: str = "BAAI/bge-reranker-base",
        llm_model_name: str = "qwen3.5-flash",
        max_history_turns: int = 12,
        raw_data_dir: str = "data/raw",
        chunk_size: int = 300,
        chunk_overlap: int = 80,
    ) -> None:
        self.store = VectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )
        self.embedder = TextEmbedder(
            model_name=embedding_model_name,
            local_files_only=True,
        )
        self.reranker = Reranker(
            model_name=reranker_model_name,
            local_files_only=True,
        )
        self.llm = LLMClient(model_name=llm_model_name)

        # L1：会话记忆
        self.chat_store = InMemoryChatStore(max_turns=max_history_turns)
        self.summarizer = ConversationSummarizer(self.llm)
        self.conversation_manager = ConversationManager(
            chat_store=self.chat_store,
            summarizer=self.summarizer,
            recent_turns_to_keep=2,
            summary_trigger_turns=5,
        )

        # L2：长期记忆
        self.memory_store = JsonMemoryStore()
        self.memory_extractor = MemoryExtractor(llm=None, use_llm_fallback=False)
        self.memory_manager = MemoryManager(
            store=self.memory_store,
            extractor=self.memory_extractor,
        )

        self.chunks = self._load_chunks_for_keyword_retrieval(
            raw_data_dir=raw_data_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.retriever = HybridRetriever(
            vector_store=self.store,
            embedder=self.embedder,
            chunks=self.chunks,
            reranker=self.reranker,
        )

    def ask(
        self,
        query: str,
        top_k: int = 3,
        session_id: Optional[str] = None,
    ) -> QAResponse:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query 不能为空")

        if self.store.count() == 0:
            raise ValueError("向量库为空，请先构建索引。")

        start = time.time()

        conversation_context = self.conversation_manager.build_context(session_id)
        print(f"[timing] build_context: {time.time() - start:.2f}s")

        # 召回相关长期记忆
        t1 = time.time()
        relevant_memories = self.memory_manager.retrieve_relevant_memories(
            query=cleaned_query,
            top_k=3,
            min_score=0.5,
        )
        print(f"[timing] retrieve_memories: {time.time() - t1:.2f}s")
        relevant_memory_dicts = [item.model_dump() for item in relevant_memories]

        t2 = time.time()
        retrieval_results = self.retriever.retrieve(
            query=cleaned_query,
            top_k=top_k,
            vector_top_k=max(top_k, 5),
            keyword_top_k=max(top_k, 5),
            enable_rerank=True,
            rerank_top_k=max(top_k, 5),
            min_rerank_score=0.30,
        )
        print(f"[timing] retrieval: {time.time() - t2:.2f}s")

        # ===== 关键修复：检索为空但长期记忆存在时，仍然继续走问答 =====
        if not retrieval_results and not relevant_memory_dicts:
            response = QAResponse(
                query=cleaned_query,
                answer="我无法根据现有知识库内容确定答案。",
                is_answerable=False,
                confidence="low",
                sources=[],
                refusal_reason="未检索到任何相关证据，且无可用长期记忆",
                retrieval_results=[],
            )

            if session_id and session_id.strip():
                self.chat_store.append_turn(
                    session_id=session_id.strip(),
                    user_query=cleaned_query,
                    assistant_answer=response.answer,
                )

            return response

        prompt = PromptBuilder.build_qa_prompt(
            query=cleaned_query,
            results=retrieval_results,
            conversation_summary=conversation_context["conversation_summary"],
            recent_history=conversation_context["recent_history"],
            long_term_memories=relevant_memory_dicts,
        )

        t3 = time.time()
        raw_output = self.llm.generate(prompt)
        print(f"[timing] llm_generate: {time.time() - t3:.2f}s")
        parsed_output = self._parse_llm_json(raw_output)

        retrieval_sources = self._build_retrieval_sources(retrieval_results)

        program_sources = self._extract_sources_from_results(retrieval_results)
        program_confidence = self._estimate_confidence(
            retrieval_results=retrieval_results,
            relevant_memories=relevant_memory_dicts,
        )
        retrieval_is_answerable = self._judge_answerable(
            retrieval_results=retrieval_results,
            relevant_memories=relevant_memory_dicts,
        )

        model_is_answerable = parsed_output["is_answerable"]
        final_is_answerable = bool(model_is_answerable and retrieval_is_answerable)

        final_answer = parsed_output["answer"]
        final_refusal_reason = parsed_output.get("refusal_reason")

        if not final_is_answerable:
            if retrieval_is_answerable is False and model_is_answerable is True:
                final_answer = "我无法根据现有知识库内容确定答案。"
                final_refusal_reason = "检索证据和长期记忆均不足以支撑明确结论"
            elif not final_refusal_reason:
                final_refusal_reason = "知识库与长期记忆均未提供明确答案"

        response = QAResponse(
            query=cleaned_query,
            answer=final_answer,
            is_answerable=final_is_answerable,
            confidence=program_confidence,
            sources=program_sources,
            refusal_reason=final_refusal_reason if not final_is_answerable else None,
            retrieval_results=retrieval_sources,
        )

        if session_id and session_id.strip():
            self.chat_store.append_turn(
                session_id=session_id.strip(),
                user_query=cleaned_query,
                assistant_answer=response.answer,
            )

        # 当前轮问答抽取为长期记忆
        t4 = time.time()
        self.memory_manager.extract_and_save(
            query=cleaned_query,
            answer=response.answer,
            session_id=session_id,
        )
        print(f"[timing] extract_and_save_memory: {time.time() - t4:.2f}s")

        print(f"[timing] total ask: {time.time() - start:.2f}s")
        return response

    def clear_session_history(self, session_id: str) -> None:
        self.chat_store.clear_history(session_id)

    def get_session_debug_info(self, session_id: str) -> dict:
        cleaned_session_id = session_id.strip()
        if not cleaned_session_id:
            raise ValueError("session_id 不能为空")

        history = self.chat_store.get_history(cleaned_session_id)
        conversation_context = self.conversation_manager.build_context(cleaned_session_id)

        return {
            "session_id": cleaned_session_id,
            "history_count": len(history),
            "raw_history": [turn.to_dict() for turn in history],
            "raw_history_text": HistoryFormatter.format_history(history),
            "conversation_summary": conversation_context["conversation_summary"],
            "recent_history": conversation_context["recent_history"],
        }

    def list_long_term_memories(self) -> list[dict]:
        return [item.model_dump() for item in self.memory_manager.list_memories()]

    def clear_long_term_memories(self) -> None:
        self.memory_manager.clear_memories()

    def get_memory_debug_info(self, query: str) -> dict:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query 不能为空")

        relevant_memories = self.memory_manager.retrieve_relevant_memories(
            query=cleaned_query,
            top_k=5,
            min_score=0.5,
        )

        return {
            "query": cleaned_query,
            "retrieved_memory_count": len(relevant_memories),
            "retrieved_memories": [item.model_dump() for item in relevant_memories],
        }

    @staticmethod
    def _build_retrieval_sources(results: List[Dict[str, Any]]) -> List[RetrievalSource]:
        sources: List[RetrievalSource] = []

        for item in results:
            preview = item["text"][:200].replace("\n", " ")
            distance = item["distance"] if item.get("distance") is not None else -1.0

            sources.append(
                RetrievalSource(
                    source=item["source"],
                    chunk_id=item["chunk_id"],
                    chunk_index=item["chunk_index"],
                    distance=float(distance),
                    preview=preview,
                )
            )

        return sources

    @staticmethod
    def _extract_sources_from_results(results: List[Dict[str, Any]]) -> List[str]:
        sources: List[str] = []
        for item in results:
            source = item.get("source")
            if source and source not in sources:
                sources.append(source)
        return sources

    @staticmethod
    def _estimate_confidence(
        retrieval_results: List[Dict[str, Any]],
        relevant_memories: List[Dict[str, Any]],
    ) -> str:
        """
        优先依据检索结果；
        若检索为空但有长期记忆，则给 memory-based 置信度。
        """
        if retrieval_results:
            top1 = retrieval_results[0]

            if "rerank_score" in top1:
                rerank_score = float(top1["rerank_score"])
                if rerank_score >= 0.90:
                    return "high"
                if rerank_score >= 0.50:
                    return "medium"
                return "low"

            distance = top1.get("distance")
            if distance is None:
                return "medium"

            distance = float(distance)
            if distance <= 0.60:
                return "high"
            if distance <= 0.80:
                return "medium"
            return "low"

        if relevant_memories:
            if len(relevant_memories) >= 2:
                return "medium"
            return "low"

        return "low"

    @staticmethod
    def _judge_answerable(
        retrieval_results: List[Dict[str, Any]],
        relevant_memories: List[Dict[str, Any]],
    ) -> bool:
        """
        只要检索结果可回答，或长期记忆存在，就允许继续回答。
        最终仍由模型判断是否真的 answerable。
        """
        if retrieval_results:
            top1 = retrieval_results[0]

            if "rerank_score" in top1:
                return float(top1["rerank_score"]) >= 0.30

            distance = top1.get("distance")
            if distance is None:
                return True

            return float(distance) <= 0.85

        return len(relevant_memories) > 0

    @staticmethod
    def _parse_llm_json(raw_output: str) -> Dict[str, Any]:
        text = raw_output.strip()

        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").strip()
            if text.endswith("```"):
                text = text[:-3].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"模型输出不是合法 JSON。\n原始输出:\n{raw_output}"
            ) from exc

        required_fields = ["answer", "is_answerable", "confidence", "sources", "refusal_reason"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"模型输出缺少必要字段: {field}")

        if data["confidence"] not in {"high", "medium", "low"}:
            raise ValueError("confidence 字段非法，必须是 high / medium / low")

        if not isinstance(data["sources"], list):
            raise ValueError("sources 字段必须是列表")

        if not isinstance(data["is_answerable"], bool):
            raise ValueError("is_answerable 字段必须是布尔值")

        if not isinstance(data["answer"], str):
            raise ValueError("answer 字段必须是字符串")

        return data

    @staticmethod
    def _load_chunks_for_keyword_retrieval(
        raw_data_dir: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[Chunk]:
        loader = DocumentLoader(raw_data_dir)
        documents = loader.load_documents()

        splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_documents(documents)