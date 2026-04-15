from __future__ import annotations

from app.rag.rag_qa import RAGQAService


def print_answer(result: dict) -> None:
    print("\n最终回答：")
    print("=" * 100)
    print(result["answer"])

    print("\n检索证据：")
    print("=" * 100)
    for i, item in enumerate(result["retrieval_results"], start=1):
        preview = item["text"][:200].replace("\n", " ")
        print(f"[Top {i}]")
        print(f"source      : {item['source']}")
        print(f"chunk_id    : {item['chunk_id']}")
        print(f"chunk_index : {item['chunk_index']}")
        print(f"distance    : {item['distance']}")
        print(f"preview     : {preview}")
        print("-" * 100)


def main() -> None:
    service = RAGQAService()

    print("RAG 问答测试已启动，输入 q 退出。")

    while True:
        query = input("\n请输入问题: ").strip()

        if query.lower() in {"q", "quit", "exit"}:
            print("已退出。")
            break

        if not query:
            print("问题不能为空。")
            continue

        try:
            result = service.ask(query=query, top_k=3)
            print_answer(result)
        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    main()