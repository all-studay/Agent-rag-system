from __future__ import annotations

from app.rag.document_loader import DocumentLoader
from app.rag.text_splitter import TextSplitter
from app.rag.embedder import TextEmbedder
from app.rag.vector_store import VectorStore

def main() -> None:
    #1, 读取文档
    loader = DocumentLoader("data/raw")
    documents = loader.load_documents()

    if not documents:
        print("未加载到任何文档，请检查 data/raw 目录")
        return

    print(f"成功加载文档数： {len(documents)}")

    #2, 切分文档
    splitter = TextSplitter(chunk_size=300, chunk_overlap=80)
    chunks = splitter.split_documents(documents)

    if not chunks:
        print("未生成任何 chunk， 请检查文档内容。")
        return

    print(f"成功生成 chunk 数: {len(chunks)}")

    #3, 生成向量
    embedder = TextEmbedder(model_name="BAAI/bge-small-zh-v1.5", local_files_only=True)
    embeddings = embedder.embed_chunks(chunks)

    print(f"f成功生成 embedding 数：{len(embeddings)}")


    #4, 写入向量库
    store = VectorStore(
        persist_dir="data/vectorstore",
        collection_name="rag_chunks"
    )

    # 为了方便反复测试，默认先重建collection
    store.reset_collection()
    store.add_chunks(chunks, embeddings)

    print(f"向量库当前条数：{store.count()}")
    print("索引构建完成。")



if __name__ == '__main__':
    main()