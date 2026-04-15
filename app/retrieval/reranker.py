from __future__ import annotations

from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

class Reranker:
    """
        基于 CrossEncoder 的重排序器。

        第一版目标：
        - 接收 query 和候选 chunks
        - 为每个候选 (query, text) 打分
        - 按 rerank_score 从高到低重排
    """
    def __init__(
            self,
            model_name: str = "BAAI/bge-reranker-base",
            local_files_only: bool = True,
    ) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only

        try:
            self.model = CrossEncoder(
                model_name,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            if local_files_only:
                raise RuntimeError(
                    f"无法从本地加载 reranker 模型：{model_name}。\n"
                    f"请先确保模型已下载到本地缓存，或将 local_files_only 改为 False。"
                ) from exc

            raise RuntimeError(
                f"加载 reranker 模型失败：{model_name}。\n"
                f"这通常是网络、代理、SSL 或 Hugging Face 访问问题。"
            ) from exc

    def rerank(
            self,
            query: str,
            candidates: List[Dict[str, Any]],
            top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
            对候选结果进行重排序。
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query 不能为空")

        if not candidates:
            return []

        pairs = [(cleaned_query, item["text"]) for item in candidates]
        scores = self.model.predict(pairs)

        reranked_results: List[Dict[str, Any]] = []
        for item, score in zip(candidates, scores):
            enriched_item = dict(item)
            enriched_item["rerank_score"] = float(score)
            reranked_results.append(enriched_item)

        reranked_results.sort(
            key=lambda x : x["rerank_score"],
            reverse=True,
        )

        if top_k is not None:
            return reranked_results[:top_k]

        return reranked_results