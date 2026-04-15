from __future__ import annotations

from app.langchain.langchain_rag_chain import LangChainRAGService


def print_result(result) -> None:
    print("\n问答结果：")
    print("=" * 100)
    print(f"query         : {result.query}")
    print(f"is_answerable : {result.is_answerable}")
    print(f"confidence    : {result.confidence}")
    print(f"sources       : {result.sources}")
    print(f"refusal_reason: {result.refusal_reason}")
    print(f"answer        : {result.answer}")
    print("-" * 100)


def main() -> None:
    service = LangChainRAGService()
    session_id = "langchain-test-session"

    test_queries = [
        "系统支持上传哪些文档格式？",
        "后续还计划增加什么？",
        "第一版项目的核心目标是什么？",
        "系统支持记忆功能吗？",
    ]

    for q in test_queries:
        result = service.ask(query=q, top_k=3, session_id=session_id)
        print_result(result)


if __name__ == "__main__":
    main()