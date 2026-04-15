from __future__ import annotations

from app.rag.document_loader import DocumentLoader
from app.rag.embedder import TextEmbedder
from app.rag.text_splitter import TextSplitter
from app.rag.vector_store import VectorStore
from app.retrieval.hybrid_retriever import HybridRetriever


def print_results(results: list[dict]) -> None:
    if not results:
        print("没有检索到结果。")
        return

    print("\n混合检索结果：")
    print("=" * 120)

    for i, item in enumerate(results, start=1):
        preview = item["text"][:180].replace("\n", " ")
        print(f"[Top {i}]")
        print(f"chunk_id       : {item['chunk_id']}")
        print(f"source         : {item['source']}")
        print(f"retrieval_type : {item.get('retrieval_type')}")
        print(f"vector_rank    : {item.get('vector_rank')}")
        print(f"keyword_rank   : {item.get('keyword_rank')}")
        print(f"distance       : {item.get('distance')}")
        print(f"keyword_score  : {item.get('keyword_score')}")
        print(f"hybrid_score   : {item.get('hybrid_score')}")
        print(f"preview        : {preview}")
        print("-" * 120)


def main() -> None:
    loader = DocumentLoader("data/raw")
    documents = loader.load_documents()

    splitter = TextSplitter(chunk_size=300, chunk_overlap=80)
    chunks = splitter.split_documents(documents)

    store = VectorStore(
        persist_dir="data/vectorstore",
        collection_name="rag_chunks",
    )
    embedder = TextEmbedder(model_name="BAAI/bge-small-zh-v1.5")

    retriever = HybridRetriever(
        vector_store=store,
        embedder=embedder,
        chunks=chunks,
    )

    if store.count() == 0:
        print("当前向量库为空，请先运行 scripts/build_index.py 构建索引。")
        return

    print(f"当前向量库条数: {store.count()}")
    print("输入问题开始测试混合检索，输入 q 退出。")

    while True:
        query = input("\n请输入问题: ").strip()

        if query.lower() in {"q", "quit", "exit"}:
            print("已退出。")
            break

        if not query:
            print("问题不能为空。")
            continue

        results = retriever.retrieve(
            query=query,
            top_k=5,
            vector_top_k=5,
            keyword_top_k=5,
        )
        print_results(results)


if __name__ == "__main__":
    main()