from __future__ import annotations

from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

from app.rag.text_splitter import Chunk

class KeywordRetriever:
    """
        基于 BM25 的关键词检索器。
        第一版设计：
        - 直接基于 chunk 文本建立索引
        - 使用简单分词策略
        - 返回与向量检索格式尽量一致的结果
    """

    def __init__(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("初始化 KeywordRetriever 时，chunks 不能为空")

        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)


    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
            对 query 做 BM25 检索，返回 top-k 结果。
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query 不能为空")

        tokenized_query = self._tokenize(cleaned_query)
        scores = self.bm25.get_scores(tokenized_query)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        results: List[Dict[str, Any]] = []
        for idx, score in indexed_scores[: top_k]:
            chunk = self.chunks[idx]
            results.append(
                {"chunk_id": chunk.chunk_id,
                 "text": chunk.text,
                 "source": chunk.source,
                 "file_path": chunk.file_path,
                 "file_type": chunk.file_type,
                 "chunk_index": chunk.chunk_index,
                 "start_char": chunk.start_char,
                 "end_char": chunk.end_char,
                 "page_count": chunk.page_count,
                 "distance": None,  # BM25 不使用向量距离
                 "keyword_score": float(score),
                 "retrieval_type": "keyword",
                 }
            )

        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
            简单分词策略：
            - 先按空白切
            - 再补充字符级切分（适配中文小规模 demo）

            说明：
            这不是工业级中文分词方案，但对当前项目演示足够。
            后续可以替换为 jieba / pkuseg 等更正式方案。
        """
        text = text.strip().lower()

        whitespace_tokens = text.split()
        char_tokens = [ch for ch in text if not ch.isspace()]

        return whitespace_tokens + char_tokens