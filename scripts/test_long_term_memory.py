from __future__ import annotations

from app.rag.rag_qa import RAGQAService


def print_memories(memories: list[dict]) -> None:
    if not memories:
        print("当前没有长期记忆。")
        return

    print("\n当前长期记忆：")
    print("=" * 100)
    for i, item in enumerate(memories, start=1):
        print(f"[{i}]")
        print(f"memory_id        : {item['memory_id']}")
        print(f"memory_type      : {item['memory_type']}")
        print(f"importance       : {item['importance']}")
        print(f"source_session_id: {item['source_session_id']}")
        print(f"source_query     : {item['source_query']}")
        print(f"created_at       : {item['created_at']}")
        print(f"content          : {item['content']}")
        print("-" * 100)


def print_retrieved_memories(debug_info: dict) -> None:
    print("\n相关长期记忆召回结果：")
    print("=" * 100)
    print(f"query                : {debug_info['query']}")
    print(f"retrieved_memory_count: {debug_info['retrieved_memory_count']}")
    print("-" * 100)

    for i, item in enumerate(debug_info["retrieved_memories"], start=1):
        print(f"[{i}]")
        print(f"memory_type      : {item['memory_type']}")
        print(f"importance       : {item['importance']}")
        print(f"source_session_id: {item['source_session_id']}")
        print(f"source_query     : {item['source_query']}")
        print(f"content          : {item['content']}")
        print("-" * 100)


def print_answer(result) -> None:
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
    service = RAGQAService()
    session_id = "memory-test-session"

    print("长期记忆测试脚本已启动。")
    print("将先清空长期记忆，再写入几轮问答，然后测试记忆召回与问答。")

    # 1. 清空长期记忆
    service.clear_long_term_memories()
    print("\n已清空长期记忆。")

    # 2. 写入几轮典型问答，触发长期记忆抽取
    seed_queries = [
        "系统支持上传哪些文档格式？",
        "第一版项目的核心目标是什么？",
        "系统支持记忆功能吗？",
        "系统如何处理长对话？",
    ]

    print("\n开始写入种子问答...")
    for q in seed_queries:
        result = service.ask(query=q, top_k=3, session_id=session_id)
        print_answer(result)

    # 3. 查看当前长期记忆
    memories = service.list_long_term_memories()
    print_memories(memories)

    # 4. 测试长期记忆召回
    test_memory_queries = [
        "系统后续还计划增加什么？",
        "当前项目第一版的目标是什么？",
        "系统怎么处理长上下文？",
        "这个项目支持记忆吗？",
    ]

    for q in test_memory_queries:
        debug_info = service.get_memory_debug_info(q)
        print_retrieved_memories(debug_info)

    # 5. 测试带长期记忆注入的问答
    final_queries = [
        "这个项目现在已经实现了哪些能力？",
        "系统在处理长对话时会怎么做？",
        "这个系统未来准备扩展哪些内容？",
    ]

    for q in final_queries:
        result = service.ask(query=q, top_k=3, session_id=session_id)
        print_answer(result)

    print("\n长期记忆测试完成。")


if __name__ == "__main__":
    main()