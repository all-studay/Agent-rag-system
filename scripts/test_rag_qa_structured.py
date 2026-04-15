from __future__ import annotations

from app.rag.rag_qa import RAGQAService


def print_result(result) -> None:
    print("\n结构化问答结果：")
    print("=" * 100)
    print(f"query         : {result.query}")
    print(f"is_answerable : {result.is_answerable}")
    print(f"confidence    : {result.confidence}")
    print(f"sources       : {result.sources}")
    print(f"refusal_reason: {result.refusal_reason}")
    print(f"answer        : {result.answer}")

    print("\n检索证据：")
    print("=" * 100)
    for i, item in enumerate(result.retrieval_results, start=1):
        print(f"[Top {i}]")
        print(f"source      : {item.source}")
        print(f"chunk_id    : {item.chunk_id}")
        print(f"chunk_index : {item.chunk_index}")
        print(f"distance    : {item.distance}")
        print(f"preview     : {item.preview}")
        print("-" * 100)


def main() -> None:
    service = RAGQAService()

    print("结构化 RAG 问答测试已启动，输入 q 退出。")

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
            print_result(result)
        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    main()