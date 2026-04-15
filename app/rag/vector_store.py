from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.rag.text_splitter import Chunk
from app.rag.embedder import TextEmbedder

class VectorStore:
    """
        基于 Chroma 的本地向量库封装。

        第一版目标：
        - 支持持久化存储
        - 支持批量写入 chunk + embedding
        - 支持相似度检索
    """

    def __init__(
            self,
            persist_dir: str = "data/vectorstore",
            collection_name: str = "rag_chunks",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()


    def _get_or_create_collection(self) -> Collection:
        return self.client.get_or_create_collection(name=self.collection_name)

    def reset_collection(self) -> None:
        """
            重建 collection。
            在重复测试时很有用，避免旧数据干扰。
        """
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
            批量写入 chunks 和对应 embeddings。
        """

        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 数量与 embeddings 数量不一致")

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "source": chunk.source,
                    "file_path": chunk.file_path,
                    "file_type": chunk.file_type,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "page_count": chunk.page_count if chunk.page_count is not None else -1
                }
            )

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def similarity_search(
            self,
            query: str,
            embedder: TextEmbedder,
            top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
            对 query 做向量化，并返回最相似的 top-k chunk。
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query 不能为空")

        query_embedding = embedder.embed_text(cleaned_query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        return self._format_query_results(results)

    @staticmethod
    def _format_query_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
            将 Chroma 返回结果整理成更易用的结构。
        """
        formatted_results: List[Dict[str, Any]] = []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(ids)):
            metadata = metadatas[i] if metadatas[i] is not None else {}

            page_count = metadata.get("page_count", -1)
            if page_count == -1:
                page_count = None

            formatted_results.append(
                {
                    "chunk_id": ids[i],
                    "text": documents[i],
                    "distance": distances[i],
                    "source": metadata.get("source"),
                    "file_path": metadata.get("file_path"),
                    "file_type": metadata.get("file_type"),
                    "chunk_index": metadata.get("chunk_index"),
                    "start_char": metadata.get("start_char"),
                    "end_char": metadata.get("end_char"),
                    "page_count": page_count,
                }
            )

        return formatted_results

    def count(self) -> int:
        """
            返回当前 collection 中的向量条数。
        """

        return self.collection.count()



if __name__ == "__main__":
    from app.rag.document_loader import DocumentLoader
    from app.rag.text_splitter import TextSplitter

    # 1. 读取文档
    loader = DocumentLoader("data/raw")
    documents = loader.load_documents()

    # 2. 切分
    splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # 3. 向量化
    embedder = TextEmbedder(model_name="BAAI/bge-small-zh-v1.5")
    embeddings = embedder.embed_chunks(chunks)

    # 4. 写入向量库
    store = VectorStore(
        persist_dir="data/vectorstore",
        collection_name="rag_chunks"
    )
    store.reset_collection()
    store.add_chunks(chunks, embeddings)

    print(f"已写入向量条数: {store.count()}")

    # 5. 测试检索
    query = "年假需要提前多久申请？"
    results = store.similarity_search(query=query, embedder=embedder, top_k=3)

    print(f"\n查询问题: {query}")
    print("=" * 80)

    for i, item in enumerate(results, start=1):
        preview = item["text"][:150].replace("\n", " ")
        print(f"[Top {i}]")
        print(f"chunk_id    : {item['chunk_id']}")
        print(f"source      : {item['source']}")
        print(f"chunk_index : {item['chunk_index']}")
        print(f"distance    : {item['distance']}")
        print(f"preview     : {preview}")
        print("-" * 80)