from __future__ import annotations

from typing import List
from sentence_transformers import SentenceTransformer

from app.rag.text_splitter import Chunk

class TextEmbedder:
    """
       文本向量化器。

       第一版目标：
       - 加载本地 sentence-transformers 模型
       - 支持单条文本和批量文本向量化
       - 支持直接对 Chunk 列表生成向量
    """

    def __init__(
            self,
            model_name: str = "BAAI/bge-small-zh-v1.5",
            local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        try:
            self.model = SentenceTransformer(
                model_name,
                local_files_only = local_files_only,
            )
        except Exception as exc:
            if local_files_only:
                raise RuntimeError(
                    f"无法从本地加载 embedding 模型：{model_name}。\n"
                    f"请先确保模型已下载到本地缓存，或将 local_files_only 改为 False。"
                ) from exc
            raise RuntimeError(
                f"加载 embedding 模型失败：{model_name}。\n"
                f"这通常是网络、代理、SSL 或 Hugging Face 访问问题。"
            ) from exc

    @staticmethod
    def _add_query_instruction(query: str) -> str:
        return f"为这个句子生成表示以用于检索相关文章：{query}"

    def embed_text(self, text:str, is_query: bool = False) -> List[float]:
        """
            对单条文本生成 embedding。
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("输入文本不能为空")

        if is_query:
            cleaned_text = self._add_query_instruction(cleaned_text)

        embedding = self.model.encode(cleaned_text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
            对多条文本批量生成 embedding。
        """
        if not texts:
            return []

        cleaned_texts = [text.strip() for text in texts]
        if any(not text for text in cleaned_texts):
            raise ValueError("texts 中存在空文本，请先清理后再向量化")

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def embed_chunks(self, chunks: List[Chunk], batch_size: int = 32) -> List[List[float]]:
        """
            对 Chunk 列表生成 embedding。
            只取每个 chunk 的 text 字段进行向量化。
        """
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts, batch_size=batch_size)



if __name__ == "__main__":
    from app.rag.document_loader import DocumentLoader
    from app.rag.text_splitter import TextSplitter

    loader = DocumentLoader("data/raw")
    documents = loader.load_documents()

    splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embedder = TextEmbedder(model_name="BAAI/bge-small-zh-v1.5")
    embeddings = embedder.embed_chunks(chunks)

    print(f"文档数量: {len(documents)}")
    print(f"chunk 数量: {len(chunks)}")
    print(f"embedding 数量: {len(embeddings)}")

    if embeddings:
        print(f"单条 embedding 维度: {len(embeddings[0])}")
        print(f"第一个 chunk_id: {chunks[0].chunk_id}")
        print(f"第一个向量前 10 维: {embeddings[0][:10]}")
