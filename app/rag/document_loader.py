from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import logging

import fitz

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

@dataclass
class Document:
    """
    统一的文档数据结构，方便后续切分和入库
    """
    source: str               # 文件名，如leave_policy.md
    file_path: str            # 文件完整路径
    text: str                 # 文档全文
    file_type: str            # txt / md /pdf
    page_count: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)



class DocumentLoader:
    """
        文档加载器：
        - 支持读取 txt / md / pdf
        - 支持批量加载目录中的文件
        - 返回统一的 Document 对象列表
    """

    def __init__(self, data_dir: str | Path)->None:
        self.data_dir = Path(data_dir)


    def load_documents(self)->List[Document]:
        """
            读取目录下所有支持的文件，返回 Document 列表。
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在：{self.data_dir}")

        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"给定路径不是目录：{self.data_dir}")

        documents: List[Document] = []

        for file_path in sorted(self.data_dir.rglob("*")):
            if not file_path.is_file():
                continue

            suffix = file_path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                logger.debug("跳过不支持的文件类型： %s", file_path)
                continue

            try:
                document = self._load_single_file(file_path)
                if document and document.text.strip():
                    documents.append(document)
                    logger.info("已加载文件： %s", file_path.name)
                else:
                    logger.warning("文件内容为空，已跳过： %s", file_path.name)

            except Exception as exc:
                logger.exception("加载文件失败：%s, 错误： %s", file_path, exc)


        return documents


    def _load_single_file(self, file_path: Path) -> Document:
        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            text = self._read_text_file(file_path)
            return Document(
                source=file_path.name,
                file_path=str(file_path.resolve()),
                text=text,
                file_type="txt",
            )

        if suffix == ".md":
            text = self._read_text_file(file_path)
            return Document(
                source=file_path.name,
                file_path=str(file_path.resolve()),
                text=text,
                file_type="md",
            )

        if suffix == ".pdf":
            text, page_count = self._read_pdf_file(file_path)
            return Document(
                source=file_path.name,
                file_path=str(file_path.resolve()),
                text=text,
                file_type="pdf",
                page_count=page_count
            )


        raise ValueError(f"不支持的文件类型：{file_path.suffix}")

    @staticmethod
    def _read_text_file(file_path: Path) -> str:
        """
            读取 txt / md 文件。
            优先用 utf-8，失败后尝试 utf-8-sig 和 gbk。
        """
        encodings = ["utf-8", "utf-8-sig", "gbk"]

        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue

        raise UnicodeDecodeError(
            "unknown",
            b"",
            0,
            1,
            f"无法解码文件：{file_path}",
        )

    @staticmethod
    def _read_pdf_file(file_path: Path) -> tuple[str, int]:
        """
            读取 PDF 全文。
            使用 PyMuPDF 提取每一页文本，并拼接成一个字符串。
         """

        text_parts: List[str] = []

        with fitz.open(file_path) as pdf:
            page_count = pdf.page_count

            for page_index in range(page_count):
                page = pdf.load_page(page_index)
                page_text = page.get_text("text")
                if page_text:
                    text_parts.append(page_text.strip())


        full_text = "\n\n".join(text_parts)
        return full_text, page_count


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )



if __name__ == "__main__":
    setup_logging()

    # 这里默认读取项目下的 data/raw 目录
    loader = DocumentLoader("data/raw")
    docs = loader.load_documents()

    print(f"\n共加载文档: {len(docs)}\n")

    for i, doc in enumerate(docs, start=1):
        preview = doc.text[:200].replace("\n", " ")
        print(f"[{i}] {doc.source}")
        print(f"    类型: {doc.file_type}")
        print(f"    路径: {doc.file_path}")
        print(f"    页数: {doc.page_count}")
        print(f"    预览: {preview}")
        print("-" * 80)




























