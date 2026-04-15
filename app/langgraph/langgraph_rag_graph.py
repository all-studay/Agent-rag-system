from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.chat.conversation_manager import ConversationManager
from app.chat.history_formatter import HistoryFormatter
from app.chat.memory import InMemoryChatStore
from app.chat.summarizer import ConversationSummarizer
from app.langchain.schemas import LangChainQAOutput
from app.langgraph.state import GraphRAGState
from app.memory.extractor import MemoryExtractor
from app.memory.manager import MemoryManager
from app.memory.store import JsonMemoryStore
from app.rag.document_loader import DocumentLoader
from app.rag.embedder import TextEmbedder
from app.rag.llm_client import LLMClient
from app.rag.schemas import QAResponse, RetrievalSource
from app.rag.text_splitter import Chunk, TextSplitter
from app.rag.vector_store import VectorStore
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker


class LangGraphRAGService:
    """
    基于 LangGraph 的 RAG 问答服务第一版

    编排流程：
    1. build_context
    2. retrieve_memory
    3. retrieve_docs
    4. route_after_retrieval
    5. generate_answer
    6. finalize_response
    7. persist_memory_and_history
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
        self.raw_llm_client = LLMClient(model_name=llm_model_name)

        self.chat_model = ChatOpenAI(
            model=llm_model_name,
            api_key=self.raw_llm_client.api_key,
            base_url=self.raw_llm_client.base_url,
            temperature=0,
        )

        # L1：会话记忆
        self.chat_store = InMemoryChatStore(max_turns=max_history_turns)
        self.summarizer = ConversationSummarizer(self.raw_llm_client)
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

        self.output_parser = PydanticOutputParser(pydantic_object=LangChainQAOutput)
        self.prompt = self._build_prompt()
        self.graph = self._build_graph()

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

        init_state: GraphRAGState = {
            "query": cleaned_query,
            "top_k": top_k,
            "session_id": session_id,
        }

        final_state = self.graph.invoke(init_state)
        final_response = final_state["final_response"]

        retrieval_results = final_response["retrieval_results"]
        return QAResponse(
            query=final_response["query"],
            answer=final_response["answer"],
            is_answerable=final_response["is_answerable"],
            confidence=final_response["confidence"],
            sources=final_response["sources"],
            refusal_reason=final_response["refusal_reason"],
            retrieval_results=retrieval_results,
        )

    def clear_session_history(self, session_id: str) -> None:
        self.chat_store.clear_history(session_id)

    def list_long_term_memories(self) -> list[dict]:
        return [item.model_dump() for item in self.memory_manager.list_memories()]

    def clear_long_term_memories(self) -> None:
        self.memory_manager.clear_memories()

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

    def _build_graph(self):
        graph_builder = StateGraph(GraphRAGState)

        graph_builder.add_node("build_context", self._node_build_context)
        graph_builder.add_node("retrieve_memory", self._node_retrieve_memory)
        graph_builder.add_node("retrieve_docs", self._node_retrieve_docs)
        graph_builder.add_node("generate_answer", self._node_generate_answer)
        graph_builder.add_node("finalize_response", self._node_finalize_response)
        graph_builder.add_node("persist_memory_and_history", self._node_persist_memory_and_history)
        graph_builder.add_node("early_refuse", self._node_early_refuse)

        graph_builder.set_entry_point("build_context")
        graph_builder.add_edge("build_context", "retrieve_memory")
        graph_builder.add_edge("retrieve_memory", "retrieve_docs")

        graph_builder.add_conditional_edges(
            "retrieve_docs",
            self._route_after_retrieval,
            {
                "early_refuse": "early_refuse",
                "generate_answer": "generate_answer",
            },
        )

        graph_builder.add_edge("early_refuse", END)
        graph_builder.add_edge("generate_answer", "finalize_response")
        graph_builder.add_edge("finalize_response", "persist_memory_and_history")
        graph_builder.add_edge("persist_memory_and_history", END)

        return graph_builder.compile()

    def _node_build_context(self, state: GraphRAGState) -> GraphRAGState:
        session_id = state.get("session_id")
        conversation_context = self.conversation_manager.build_context(session_id)
        return {
            **state,
            "conversation_context": conversation_context,
        }

    def _node_retrieve_memory(self, state: GraphRAGState) -> GraphRAGState:
        query = state["query"]
        relevant_memories = self.memory_manager.retrieve_relevant_memories(
            query=query,
            top_k=3,
            min_score=0.5,
        )
        return {
            **state,
            "relevant_memories": [item.model_dump() for item in relevant_memories],
        }

    def _node_retrieve_docs(self, state: GraphRAGState) -> GraphRAGState:
        query = state["query"]
        top_k = state["top_k"]

        retrieval_results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            vector_top_k=max(top_k, 5),
            keyword_top_k=max(top_k, 5),
            enable_rerank=True,
            rerank_top_k=max(top_k, 5),
            min_rerank_score=0.30,
        )

        should_refuse_early = not retrieval_results and not state.get("relevant_memories")

        return {
            **state,
            "retrieval_results": retrieval_results,
            "should_refuse_early": should_refuse_early,
        }

    def _route_after_retrieval(self, state: GraphRAGState) -> str:
        if state.get("should_refuse_early"):
            return "early_refuse"
        return "generate_answer"

    def _node_early_refuse(self, state: GraphRAGState) -> GraphRAGState:
        final_response = {
            "query": state["query"],
            "answer": "我无法根据现有知识库内容确定答案。",
            "is_answerable": False,
            "confidence": "low",
            "sources": [],
            "refusal_reason": "未检索到任何相关证据，且无可用长期记忆",
            "retrieval_results": [],
        }
        return {
            **state,
            "final_response": final_response,
        }

    def _node_generate_answer(self, state: GraphRAGState) -> GraphRAGState:
        conversation_context = state["conversation_context"]
        relevant_memories = state.get("relevant_memories", [])
        retrieval_results = state.get("retrieval_results", [])

        chain = self.prompt | self.chat_model | self.output_parser

        parsed_output: LangChainQAOutput = chain.invoke(
            {
                "format_instructions": self.output_parser.get_format_instructions(),
                "long_term_memories": self._build_memory_context(relevant_memories),
                "conversation_summary": conversation_context["conversation_summary"],
                "recent_history": conversation_context["recent_history"],
                "query": state["query"],
                "evidence_context": self._build_evidence_context(retrieval_results),
            }
        )

        return {
            **state,
            "parsed_output": parsed_output.model_dump(),
        }

    def _node_finalize_response(self, state: GraphRAGState) -> GraphRAGState:
        retrieval_results = state.get("retrieval_results", [])
        relevant_memories = state.get("relevant_memories", [])
        parsed_output = state["parsed_output"]

        retrieval_sources = self._build_retrieval_sources(retrieval_results)
        program_sources = self._extract_sources_from_results(retrieval_results)
        program_confidence = self._estimate_confidence(
            retrieval_results=retrieval_results,
            relevant_memories=relevant_memories,
        )
        retrieval_is_answerable = self._judge_answerable(
            retrieval_results=retrieval_results,
            relevant_memories=relevant_memories,
        )

        final_is_answerable = bool(parsed_output["is_answerable"] and retrieval_is_answerable)
        final_answer = parsed_output["answer"]
        final_refusal_reason = parsed_output.get("refusal_reason")

        if not final_is_answerable:
            if retrieval_is_answerable is False and parsed_output["is_answerable"] is True:
                final_answer = "我无法根据现有知识库内容确定答案。"
                final_refusal_reason = "检索证据和长期记忆均不足以支撑明确结论"
            elif not final_refusal_reason:
                final_refusal_reason = "知识库与长期记忆均未提供明确答案"

        final_response = {
            "query": state["query"],
            "answer": final_answer,
            "is_answerable": final_is_answerable,
            "confidence": program_confidence,
            "sources": program_sources if program_sources else parsed_output["sources"],
            "refusal_reason": final_refusal_reason if not final_is_answerable else None,
            "retrieval_results": retrieval_sources,
        }

        return {
            **state,
            "final_response": final_response,
        }

    def _node_persist_memory_and_history(self, state: GraphRAGState) -> GraphRAGState:
        session_id = state.get("session_id")
        final_response = state["final_response"]

        if session_id and session_id.strip():
            self.chat_store.append_turn(
                session_id=session_id.strip(),
                user_query=state["query"],
                assistant_answer=final_response["answer"],
            )

        self.memory_manager.extract_and_save(
            query=state["query"],
            answer=final_response["answer"],
            session_id=session_id,
        )

        return state

    def _build_prompt(self) -> ChatPromptTemplate:
        template = """
你是一个企业知识库问答助手。
请严格根据“长期记忆”“历史摘要”“最近对话”和“已检索到的证据”回答用户问题，不要凭空编造。

信息使用优先级：
1. 已检索到的证据
2. 长期记忆
3. 历史摘要和最近对话

如果它们之间有冲突，优先相信“已检索到的证据”。

如果证据不足，请明确表示无法根据现有知识库确定答案。
你必须只输出结构化结果。

{format_instructions}

长期记忆：
{long_term_memories}

历史摘要：
{conversation_summary}

最近对话：
{recent_history}

当前用户问题：
{query}

已检索到的证据：
{evidence_context}
"""
        return ChatPromptTemplate.from_template(template)

    @staticmethod
    def _build_evidence_context(results: List[Dict[str, Any]]) -> str:
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
    def _build_memory_context(memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return "无"

        parts = []
        for i, item in enumerate(memories, start=1):
            parts.append(
                f"[长期记忆 {i}]\n"
                f"类型: {item.get('memory_type', 'other')}\n"
                f"重要度: {item.get('importance', 'low')}\n"
                f"内容: {item.get('content', '')}\n"
            )
        return "\n".join(parts)

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
            return "medium" if len(relevant_memories) >= 2 else "low"

        return "low"

    @staticmethod
    def _judge_answerable(
        retrieval_results: List[Dict[str, Any]],
        relevant_memories: List[Dict[str, Any]],
    ) -> bool:
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