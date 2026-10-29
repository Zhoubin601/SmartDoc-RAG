# -*- coding: utf-8 -*-
"""
文档加载与切分模块单元测试
"""
import pytest
from pathlib import Path
from langchain_core.documents import Document
from src.document_loader import (
    load_single_file,
    split_documents,
    load_and_split_files,
    SUPPORTED_EXTENSIONS
)


def test_load_txt_file(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("这是开源软件构建技术的测试文本内容。\n包含第二行说明。", encoding="utf-8")

    docs = load_single_file(file_path)
    assert len(docs) == 1
    assert "开源软件构建技术" in docs[0].page_content
    assert docs[0].metadata["source"] == "sample.txt"
    assert docs[0].metadata["page"] == 1


def test_load_md_file(tmp_path: Path):
    file_path = tmp_path / "sample.md"
    file_path.write_text("# 标题一\n这是 Markdown 测试段落。\n\n## 标题二\n第二段内容。", encoding="utf-8")

    docs = load_single_file(file_path)
    assert len(docs) == 1
    assert "Markdown 测试段落" in docs[0].page_content
    assert docs[0].metadata["source"] == "sample.md"


def test_load_empty_file_raises(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="大小为 0 字节"):
        load_single_file(file_path)


def test_load_whitespace_only_file_raises(tmp_path: Path):
    file_path = tmp_path / "blank.txt"
    file_path.write_text("   \n\n\t  ", encoding="utf-8")

    with pytest.raises(ValueError, match="仅包含空白字符"):
        load_single_file(file_path)


def test_load_unsupported_format_raises(tmp_path: Path):
    file_path = tmp_path / "image.png"
    file_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValueError, match="不支持的文件格式"):
        load_single_file(file_path)


def test_split_documents():
    long_text = "段落一内容。" * 50 + "\n\n" + "段落二内容。" * 50
    doc = Document(page_content=long_text, metadata={"source": "long.txt"})

    chunks = split_documents([doc], chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 250
        assert chunk.metadata["source"] == "long.txt"


def test_load_and_split_files_batch_report(tmp_path: Path):
    # 正常文件 1
    f1 = tmp_path / "doc1.txt"
    f1.write_text("文档一的核心内容与关键参数说明。", encoding="utf-8")

    # 正常文件 2
    f2 = tmp_path / "doc2.md"
    f2.write_text("# 文档二\n详细架构与实施指南。", encoding="utf-8")

    # 异常空文件
    f3 = tmp_path / "empty.txt"
    f3.write_text("", encoding="utf-8")

    split_docs, report = load_and_split_files([f1, f2, f3], chunk_size=500, chunk_overlap=50)

    assert len(report["success"]) == 2
    assert len(report["failed"]) == 1
    assert report["failed"][0]["file"] == "empty.txt"
    assert "0 字节" in report["failed"][0]["reason"]
    assert len(split_docs) >= 2
