from __future__ import annotations

from typing import Any

from app.rag.rag_qa import RAGQAService


def contains_any_keyword(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True

    normalized_text = text.replace(" ", "")
    for keyword in keywords:
        if keyword.replace(" ", "") in normalized_text:
            return True
    return False


def evaluate_long_term_memory(service: RAGQAService) -> None:
    """
    长期记忆效果第一版评估：
    1. 先清空长期记忆
    2. 写入一组种子问答
    3. 测试记忆召回
    4. 测试带记忆的问答回答是否符合预期
    """
    session_id = "memory-eval-session"
    service.clear_long_term_memories()
    service.clear_session_history(session_id)

    print("=" * 120)
    print("开始评估长期记忆效果")
    print("=" * 120)

    # 1. 种子问答：用于生成长期记忆
    seed_queries = [
        "系统支持上传哪些文档格式？",
        "第一版项目的核心目标是什么？",
        "系统支持记忆功能吗？",
        "系统如何处理长对话？",
    ]

    print("\n[阶段1] 写入种子问答以生成长期记忆")
    for q in seed_queries:
        result = service.ask(query=q, top_k=3, session_id=session_id)
        print(f"Seed Query : {q}")
        print(f"Answer     : {result.answer}")
        print("-" * 120)

    all_memories = service.list_long_term_memories()
    print(f"\n当前长期记忆条数: {len(all_memories)}")

    # 2. 记忆召回测试集
    memory_recall_cases: list[dict[str, Any]] = [
        {
            "query": "系统后续还计划增加什么？",
            "expected_keywords": ["Word", "Excel", "网页"],
        },
        {
            "query": "这个项目第一版主要目标是什么？",
            "expected_keywords": ["文档读取", "文本切分", "向量化", "检索", "基础问答"],
        },
        {
            "query": "系统怎么处理长对话？",
            "expected_keywords": ["最近几轮", "摘要"],
        },
        {
            "query": "系统支持记忆功能吗？",
            "expected_keywords": ["L0", "L1", "L2", "L3"],
        },
    ]

    recall_total = len(memory_recall_cases)
    recall_hits = 0

    print("\n[阶段2] 评估长期记忆召回效果")
    for idx, case in enumerate(memory_recall_cases, start=1):
        query = case["query"]
        expected_keywords = case["expected_keywords"]

        debug_info = service.get_memory_debug_info(query)
        retrieved_memories = debug_info["retrieved_memories"]

        merged_memory_text = "\n".join(
            item["content"] for item in retrieved_memories
        )

        hit = contains_any_keyword(merged_memory_text, expected_keywords)
        if hit:
            recall_hits += 1

        print(f"[{idx}] Query: {query}")
        print(f"    Retrieved Count  : {debug_info['retrieved_memory_count']}")
        print(f"    Expected Keywords: {expected_keywords}")
        print(f"    Recall Hit       : {hit}")
        print("-" * 120)

    recall_acc = recall_hits / recall_total if recall_total else 0.0

    # 3. 带记忆注入的问答测试
    qa_cases: list[dict[str, Any]] = [
        {
            "query": "这个项目现在已经实现了哪些能力？",
            "expected_keywords": ["检索", "问答", "记忆", "长对话", "摘要"],
        },
        {
            "query": "这个系统未来还准备扩展哪些内容？",
            "expected_keywords": ["Word", "Excel", "网页"],
        },
    ]

    qa_total = len(qa_cases)
    qa_hits = 0

    print("\n[阶段3] 评估长期记忆注入后的问答效果")
    for idx, case in enumerate(qa_cases, start=1):
        query = case["query"]
        expected_keywords = case["expected_keywords"]

        result = service.ask(query=query, top_k=3, session_id=session_id)
        hit = contains_any_keyword(result.answer, expected_keywords)

        if hit:
            qa_hits += 1

        print(f"[{idx}] Query: {query}")
        print(f"    Answer           : {result.answer}")
        print(f"    Expected Keywords: {expected_keywords}")
        print(f"    QA Hit           : {hit}")
        print("-" * 120)

    qa_acc = qa_hits / qa_total if qa_total else 0.0

    print("\n" + "=" * 120)
    print("长期记忆评估结果汇总")
    print("=" * 120)
    print(f"长期记忆条数          : {len(all_memories)}")
    print(f"记忆召回测试数        : {recall_total}")
    print(f"记忆召回命中数        : {recall_hits}")
    print(f"记忆召回命中率        : {recall_acc:.2%}")
    print(f"记忆增强问答测试数    : {qa_total}")
    print(f"记忆增强问答命中数    : {qa_hits}")
    print(f"记忆增强问答命中率    : {qa_acc:.2%}")


def main() -> None:
    service = RAGQAService()
    evaluate_long_term_memory(service)


if __name__ == "__main__":
    main()