from __future__ import annotations

from app.rag.embedder import TextEmbedder
from app.rag.vector_store import VectorStore

def print_results(results: list[dict]) -> None:
    if not results:
        print("没有检索到结果。")
        return

    print("\n检索结果：")
    print("=" * 100)

    for i, item in enumerate(results, start=1):
        preview = item["text"][:200].replace("\n", " ")
        print(f"[Top {i}]")
        print(f"chunk_id    : {item['chunk_id']}")
        print(f"source      : {item['source']}")
        print(f"file_type   : {item['file_type']}")
        print(f"chunk_index : {item['chunk_index']}")
        print(f"distance    : {item['distance']}")
        print(f"char_range  : {item['start_char']} - {item['end_char']}")
        print(f"preview     : {preview}")
        print("-" * 100)


def main() -> None:
    store = VectorStore(
        persist_dir="data/vectorstore",
        collection_name="rag_chunks",
    )

    embedder = TextEmbedder(model_name="BAAI/bge-small-zh-v1.5")

    if store.count() == 0:
        print("当前向量库为空，请先运行 scripts/build_index.py 构建索引。")
        return

    print(f"当前向量库条数：{store.count()}")
    print("输入问题开始检索, 输入 q 退出。")

    while True:
        query = input("\n请输入问题：").strip()

        if query.lower() in {"q", "quit", "exit"}:
            print("已退出检索测试。")
            break

        if not query:
            print("问题不能为空，请重新输入。")
            continue

        results = store.similarity_search(
            query=query,
            embedder=embedder,
            top_k=3,
        )
        print_results(results)


if __name__ == '__main__':
    main()