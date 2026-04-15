from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.rag.rag_qa import RAGQAService


def load_eval_dataset(file_path: str) -> list[dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"评测文件不存在: {file_path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def keyword_match(answer: str, expected_keywords: list[str]) -> bool:
    """
    简单关键词命中规则：
    - expected_keywords 为空时直接返回 True
    - 否则命中任意一个关键词就算通过
    """
    if not expected_keywords:
        return True

    normalized_answer = answer.replace(" ", "")
    for keyword in expected_keywords:
        if keyword.replace(" ", "") in normalized_answer:
            return True
    return False


def evaluate_rag_qa(
    dataset: list[dict[str, Any]],
    service: RAGQAService,
) -> None:
    total = len(dataset)
    answerable_judgment_hits = 0
    source_hits = 0
    keyword_hits = 0

    print("=" * 120)
    print("开始评估 RAG 问答效果")
    print("=" * 120)

    for idx, item in enumerate(dataset, start=1):
        query = item["query"]
        expected_source = item["expected_source"]
        expected_keywords = item["expected_keywords"]
        expected_answerable = item["answerable"]

        result = service.ask(query=query, top_k=3, session_id=None)

        predicted_answerable = result.is_answerable
        predicted_sources = result.sources
        answer_text = result.answer

        answerable_hit = predicted_answerable == expected_answerable
        source_hit = (expected_source in predicted_sources) if expected_source else True

        if expected_answerable:
            keyword_hit = keyword_match(answer_text, expected_keywords)
        else:
            # 不可回答问题不要求关键词命中，只要求拒答正确
            keyword_hit = True

        if answerable_hit:
            answerable_judgment_hits += 1
        if source_hit:
            source_hits += 1
        if keyword_hit:
            keyword_hits += 1

        print(f"[{idx}] Query: {query}")
        print(f"    Expected Answerable : {expected_answerable}")
        print(f"    Predicted Answerable: {predicted_answerable}")
        print(f"    Expected Source     : {expected_source}")
        print(f"    Predicted Sources   : {predicted_sources}")
        print(f"    Expected Keywords   : {expected_keywords}")
        print(f"    Answer              : {answer_text}")
        print(f"    Answerable Correct  : {answerable_hit}")
        print(f"    Source Correct      : {source_hit}")
        print(f"    Keyword Match       : {keyword_hit}")
        print("-" * 120)

    answerable_acc = answerable_judgment_hits / total if total else 0.0
    source_acc = source_hits / total if total else 0.0
    keyword_acc = keyword_hits / total if total else 0.0

    print("\n" + "=" * 120)
    print("RAG 问答评估结果汇总")
    print("=" * 120)
    print(f"样本总数               : {total}")
    print(f"is_answerable 判断正确数: {answerable_judgment_hits}")
    print(f"来源命中正确数          : {source_hits}")
    print(f"关键词命中正确数        : {keyword_hits}")
    print(f"is_answerable 准确率    : {answerable_acc:.2%}")
    print(f"来源命中率              : {source_acc:.2%}")
    print(f"关键词命中率            : {keyword_acc:.2%}")


def main() -> None:
    dataset = load_eval_dataset("data/eval/eval_dataset.json")
    service = RAGQAService()
    evaluate_rag_qa(dataset, service)


if __name__ == "__main__":
    main()