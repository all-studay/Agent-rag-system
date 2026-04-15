from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

from app.rag.document_loader import Document

@dataclass
class Chunk:
    """
    文本切分后的最小检索单元。
    """
    chunk_id: str
    source: str
    file_path: str
    file_type: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    page_count: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class TextSplitter:
    """
    简单字符级文本切分器。

    第一版目标：
    - 按固定长度切分
    - 支持 chunk overlap
    - 保留来源文档和位置信息
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能小于0")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """
            批量切分文档。
        """

        all_chunks: List[Chunk] = []

        for document in documents:
            chunks = self.split_single_document(document)
            all_chunks.extend(chunks)

        return all_chunks


    def split_single_document(self, document: Document) -> List[Chunk]:
        """
        切分单篇文档。
        """
        cleaned_text = self._clean_text(document.text)

        if not cleaned_text:
            return []

        chunks: List[Chunk] = []
        start = 0
        text_length = len(cleaned_text)
        chunk_index = 0

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_text = cleaned_text[start:end].strip()

            if chunk_text:
                chunk = Chunk(
                    chunk_id=f"{document.source}_chunk_{chunk_index}",
                    source=document.source,
                    file_path=document.file_path,
                    file_type=document.file_type,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    page_count=document.page_count,
                )
                chunks.append(chunk)

            if end >=text_length:
                break

            start = end - self.chunk_overlap
            chunk_index += 1


        return chunks

    @staticmethod
    def _clean_text(text: str) -> str:
        """
            简单清洗文本：
            - 去掉首尾空白
            - 将连续超过 2 个的换行压缩为 2 个
            - 将连续空格压缩为单个空格
        """
        text = text.strip()

        # 先统一换行
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 压缩连续空格和制表符
        while "  " in text:
            text = text.replace("  ", " ")
        text = text.replace("\t", " ")

        # 压缩过多空行
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")

        return text





if __name__ == "__main__":
    from app.rag.document_loader import DocumentLoader

    loader = DocumentLoader("data/raw")
    documents = loader.load_documents()

    splitter = TextSplitter(chunk_size=300, chunk_overlap=80)
    chunks = splitter.split_documents(documents)

    source_count = {}
    for chunk in chunks:
        source_count[chunk.source] = source_count.get(chunk.source, 0) + 1

    print(source_count)

    print(f"文档数量: {len(documents)}")
    print(f"chunk 数量: {len(chunks)}")
    print("-" * 80)

    for i, chunk in enumerate(chunks[:10], start=1):
        preview = chunk.text[:120].replace("\n", " ")
        print(f"[{i}] {chunk.chunk_id}")
        print(f"    source      : {chunk.source}")
        print(f"    chunk_index : {chunk.chunk_index}")
        print(f"    char_range  : {chunk.start_char} - {chunk.end_char}")
        print(f"    preview     : {preview}")
        print("-" * 80)

















