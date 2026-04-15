from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.rag.document_loader import DocumentLoader
from app.rag.embedder import TextEmbedder
from app.rag.text_splitter import TextSplitter
from app.rag.vector_store import VectorStore
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker


def load_eval_dataset(file_path: str) -> list[dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"评测文件不存在: {file_path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_hybrid_retriever() -> HybridRetriever:
    loader = DocumentLoader("data/raw")
    documents = loader.load_documents()

    splitter = TextSplitter(chunk_size=300, chunk_overlap=80)
    chunks = splitter.split_documents(documents)

    store = VectorStore(
        persist_dir="data/vectorstore",
        collection_name="rag_chunks",
    )

    if store.count() == 0:
        raise ValueError("向量库为空，请先运行 scripts/build_index.py 构建索引。")

    embedder = TextEmbedder(model_name="BAAI/bge-small-zh-v1.5", local_files_only=True)
    reranker = Reranker(model_name=r"D:\souer\SoftwareData\Huggingface\cache\hub\models--BAAI--bge-reranker-base\snapshots\2cfc18c9415c912f9d8155881c133215df768a70", local_files_only=True)

    retriever = HybridRetriever(
        vector_store=store,
        embedder=embedder,
        chunks=chunks,
        reranker=reranker,
    )
    return retriever


def evaluate_retrieval(
    dataset: list[dict[str, Any]],
    retriever: HybridRetriever,
    top_k: int = 3,
    min_rerank_score: float = 0.30,
) -> None:
    """
    检索评估拆成三类：
    1. answerable：评 Top1 / Top3 命中率
    2. insufficient_evidence：评是否检到相关文档
    3. out_of_scope：评 no-answer 检出率
    """
    total = len(dataset)

    answerable_total = 0
    answerable_top1_hits = 0
    answerable_top3_hits = 0

    insufficient_total = 0
    insufficient_source_hits = 0

    out_of_scope_total = 0
    out_of_scope_detect_hits = 0

    print("=" * 120)
    print("开始评估检索效果")
    print("=" * 120)

    for idx, item in enumerate(dataset, start=1):
        query = item["query"]
        expected_source = item["expected_source"]
        case_type = item["case_type"]

        results = retriever.retrieve(
            query=query,
            top_k=top_k,
            vector_top_k=5,
            keyword_top_k=5,
            enable_rerank=True,
            rerank_top_k=5,
            min_rerank_score=min_rerank_score,
        )

        retrieved_sources = [r["source"] for r in results]
        top1_source = retrieved_sources[0] if retrieved_sources else None

        # 1) 可回答问题：继续看 Top1/Top3
        if case_type == "answerable":
            answerable_total += 1

            top1_hit = top1_source == expected_source
            top3_hit = expected_source in retrieved_sources[:3]

            if top1_hit:
                answerable_top1_hits += 1
            if top3_hit:
                answerable_top3_hits += 1

            print(f"[{idx}] Query: {query}")
            print(f"    Case Type        : answerable")
            print(f"    Expected Source  : {expected_source}")
            print(f"    Top1 Source      : {top1_source}")
            print(f"    Top3 Sources     : {retrieved_sources[:3]}")
            print(f"    Top1 Hit         : {top1_hit}")
            print(f"    Top3 Hit         : {top3_hit}")
            print("-" * 120)

        # 2) 证据不足问题：允许检到相关文档，但不要求返回空
        elif case_type == "insufficient_evidence":
            insufficient_total += 1

            source_hit = expected_source in retrieved_sources
            if source_hit:
                insufficient_source_hits += 1

            print(f"[{idx}] Query: {query}")
            print(f"    Case Type        : insufficient_evidence")
            print(f"    Expected Source  : {expected_source}")
            print(f"    Retrieved Sources: {retrieved_sources[:3]}")
            print(f"    Related Source Hit: {source_hit}")
            print("-" * 120)

        # 3) 超出知识库范围：希望检索层直接识别并返回空
        elif case_type == "out_of_scope":
            out_of_scope_total += 1

            no_answer_detected = len(results) == 0
            if no_answer_detected:
                out_of_scope_detect_hits += 1

            print(f"[{idx}] Query: {query}")
            print(f"    Case Type        : out_of_scope")
            print(f"    Retrieved Sources: {retrieved_sources[:3]}")
            print(f"    No-answer Detect : {no_answer_detected}")
            print("-" * 120)

        else:
            raise ValueError(f"未知 case_type: {case_type}")

    answerable_top1_acc = (
        answerable_top1_hits / answerable_total if answerable_total else 0.0
    )
    answerable_top3_acc = (
        answerable_top3_hits / answerable_total if answerable_total else 0.0
    )
    insufficient_source_acc = (
        insufficient_source_hits / insufficient_total if insufficient_total else 0.0
    )
    out_of_scope_detect_acc = (
        out_of_scope_detect_hits / out_of_scope_total if out_of_scope_total else 0.0
    )

    print("\n" + "=" * 120)
    print("检索评估结果汇总")
    print("=" * 120)
    print(f"样本总数                        : {total}")
    print(f"answerable 样本数              : {answerable_total}")
    print(f"insufficient_evidence 样本数   : {insufficient_total}")
    print(f"out_of_scope 样本数            : {out_of_scope_total}")
    print(f"answerable Top1 命中数         : {answerable_top1_hits}")
    print(f"answerable Top3 命中数         : {answerable_top3_hits}")
    print(f"insufficient_evidence 命中数   : {insufficient_source_hits}")
    print(f"out_of_scope 检出数            : {out_of_scope_detect_hits}")
    print(f"answerable Top1 命中率         : {answerable_top1_acc:.2%}")
    print(f"answerable Top3 命中率         : {answerable_top3_acc:.2%}")
    print(f"insufficient_evidence 命中率   : {insufficient_source_acc:.2%}")
    print(f"out_of_scope 检出率            : {out_of_scope_detect_acc:.2%}")
    print(f"当前 no-answer 阈值            : rerank_score < {min_rerank_score}")


def main() -> None:
    dataset = load_eval_dataset("data/eval/eval_dataset.json")
    retriever = build_hybrid_retriever()
    evaluate_retrieval(dataset, retriever, top_k=3, min_rerank_score=0.30)


if __name__ == "__main__":
    main()