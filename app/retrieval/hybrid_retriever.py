from __future__ import annotations

from typing import List, Dict, Any

from app.rag.embedder import TextEmbedder
from app.rag.text_splitter import Chunk
from app.rag.vector_store import VectorStore
from app.retrieval.keyword_retriever import KeywordRetriever
from app.retrieval.reranker import Reranker

class HybridRetriever:
    """
        混合检索器：
        - 向量检索
        - BM25 关键词检索
        - 结果融合、去重、排序
        - Reranker 二次精排
        - 检索层 no-answer 检测
    """
    def __init__(
            self,
            vector_store: VectorStore,
            embedder: TextEmbedder,
            chunks: List[Chunk],
            reranker: Reranker | None = None
    ) -> None:
        if not chunks:
            raise ValueError("初始化 HybridRetrieber 时， chunks 不能为空")

        self.vector_store = vector_store
        self.embedder = embedder
        self.keyword_retriever = KeywordRetriever(chunks)
        self.reranker = reranker


    def retrieve(
            self,
            query: str,
            top_k: int = 3,
            vector_top_k: int = 5,
            keyword_top_k: int = 5,
            enable_rerank: bool = True,
            rerank_top_k: int | None = None,
            min_rerank_score: float | None = 0.30,
    ) -> List[Dict[str, Any]]:
        """
            执行混合检索并融合结果。
            如果启用 rerank，则对融合结果再做一次精排。
            如果 top1 rerank_score 低于阈值，则判定为“无足够证据”，返回空列表。
        """

        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query 不能为空")

        vector_results = self.vector_store.similarity_search(
            query=cleaned_query,
            embedder=self.embedder,
            top_k=vector_top_k,
        )

        keyword_results = self.keyword_retriever.retrieve(
            query=cleaned_query,
            top_k=keyword_top_k,
        )

        fused_results = self._fused_results(vector_results, keyword_results)

        if enable_rerank and self.reranker is not None:
            reranked = self.reranker.rerank(
                query=cleaned_query,
                candidates=fused_results,
                top_k=rerank_top_k or max(top_k, 5),
            )

            for item in reranked:
                item["final_retrieval_type"] = "hybrid_rerank"

            # ========= no-answer 检测 =================
            if min_rerank_score is not None and reranked:
                top1_score = float(reranked[0]["rerank_score"])
                if top1_score < min_rerank_score:
                    return []
            return reranked[:top_k]

        return fused_results[: top_k]

    @staticmethod
    def _fused_results(
            vectors_results: List[Dict[str, Any]],
            keyword_results: List[Dict[str, Any]],

    ) -> List[Dict[str, Any]]:
        """
            简单融合策略：
            - 按 chunk_id 去重
            - 双命中优先
            - 使用更敏感的 rank 融合
        """

        merged: Dict[str, Dict[str, Any]] = {}

        #向量结果先写入
        for rank, item in enumerate(vectors_results, start=1):
            chunk_id = item["chunk_id"]
            merged[chunk_id] = {
                **item,
                "vector_rank": rank,
                "key_word_rank": None,
                "retrieval_type": "vector",
            }

        #再写入关键词结果
        for rank, item in enumerate(keyword_results, start=1):
            chunk_id = item["chunk_id"]

            if chunk_id in merged:
                merged[chunk_id]["keyword_rank"] = rank
                merged[chunk_id]["keyword_score"] = item.get("keyword_score")
                merged[chunk_id]["retrieval_type"] = "hybrid"
            else:
                merged[chunk_id] = {
                    **item,
                    "vector_rank": None,
                    "keyword_rank": rank,
                    "retrieval_type": "keyword",
                }

        fused_list = []
        for item in merged.values():
            vector_rank = item.get("vector_rank")
            keyword_rank = item.get("key_word_rank")

            score = 0.0

            # Reciprocal Rank 风格融合

            if vector_rank is not None:
                score += 1.0/(10 + vector_rank)
            if keyword_rank is not None:
                score += 1.0/(10 + keyword_rank)

            if vector_rank is not None and keyword_rank is not None:
                score += 0.02

            item["hybrid_score"] = score
            fused_list.append(item)

        fused_list.sort(
            key=lambda x: (
                x["hybrid_score"],
                x["keyword_score"]  if x.get("keyword_score") is not None else 0.0,
            ),
            reverse=True,
        )

        return fused_list