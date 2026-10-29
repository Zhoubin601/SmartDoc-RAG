# -*- coding: utf-8 -*-
"""
多格式文档解析与切分模块
支持 PDF、DOCX、Markdown、TXT 格式文件的智能解析、文本清洗与语义分块
"""
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf
import docx


def load_single_file(file_path: Path) -> List[Document]:
    """
    解析单个文件并返回 LangChain Document 列表
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    docs: List[Document] = []

    if suffix == ".pdf":
        reader = pypdf.PdfReader(str(file_path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": file_path.name, "page": i + 1}
                ))

    elif suffix == ".docx":
        doc = docx.Document(str(file_path))
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        content = "\n".join(full_text)
        if content.strip():
            docs.append(Document(
                page_content=content,
                metadata={"source": file_path.name, "page": 1}
            ))

    elif suffix in [".txt", ".md", ".markdown"]:
        # 兼容 UTF-8 和 GBK 编码
        content = ""
        for encoding in ["utf-8", "gbk", "gb2312", "utf-16"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if content.strip():
            docs.append(Document(
                page_content=content,
                metadata={"source": file_path.name, "page": 1}
            ))
    else:
        raise ValueError(f"不支持的文件格式: {suffix}。支持格式为: .pdf, .docx, .md, .txt")

    return docs


def split_documents(
    documents: List[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    对文档集合进行智能递归字符切分
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    return splitter.split_documents(documents)


def load_and_split_files(
    file_paths: List[Path],
    chunk_size: int = 600,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    批量加载并切分所有文件
    """
    all_raw_docs: List[Document] = []
    for fp in file_paths:
        try:
            docs = load_single_file(fp)
            all_raw_docs.extend(docs)
        except Exception as e:
            print(f"解析文件 {fp.name} 失败: {e}")

    return split_documents(all_raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
