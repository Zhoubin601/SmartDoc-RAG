# -*- coding: utf-8 -*-
"""
RAG 基准评测集（10 题多场景评测）自动化验证测试
验证：直接事实问答、同义改写、跨段落、多轮指代追问、以及文档外无关问题的检索与拒答表现
"""
import json
from pathlib import Path
import pytest
from typing import List, Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from src.document_loader import load_and_split_files
from src.models import LocalTfidfEmbeddings
from src.vector_store import SmartVectorStore
from src.rag_engine import RAGEngine, format_docs


class MockChatModel(BaseChatModel):
    response_text: str = "测试回答"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response_text))])

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"


BENCHMARK_DIR = Path(__file__).resolve().parent / "benchmark"
MANUAL_PATH = BENCHMARK_DIR / "sample_manual.md"
QUESTIONS_PATH = BENCHMARK_DIR / "eval_questions.json"


@pytest.fixture(scope="module")
def benchmark_engine():
    # 按章节适合的粒度切分演示手册（chunk_size=260, overlap=40）
    split_docs, report = load_and_split_files([MANUAL_PATH], chunk_size=260, chunk_overlap=40)
    assert len(report["failed"]) == 0
    assert len(split_docs) >= 5

    emb = LocalTfidfEmbeddings(max_features=2048)
    store = SmartVectorStore.from_documents(split_docs, emb)
    mock_llm = MockChatModel()
    engine = RAGEngine(llm=mock_llm, vector_store=store, top_k=3, min_score_threshold=0.08)
    return engine


def test_benchmark_suite_retrieval_and_refusal(benchmark_engine):
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    assert len(questions) == 10

    in_doc_hits = 0
    in_doc_total = 0

    for item in questions:
        qid = item["id"]
        question = item["question"]
        chat_history = item.get("chat_history", None)
        expected_keywords = item["expected_keywords"]
        should_refuse = item["should_refuse"]

        retrieved = benchmark_engine.retrieve_with_scores(question, chat_history=chat_history)
        top_score = retrieved[0][1] if retrieved else 0.0
        combined_retrieved_text = "\n".join(doc.page_content for doc, _ in retrieved)

        if not should_refuse:
            in_doc_total += 1
            # 验证所有预期关键信息均在 Top-K 召回结果中
            keywords_found = all(kw in combined_retrieved_text for kw in expected_keywords)
            assert keywords_found, f"[{qid}] 未能在 Top-K 中召回全部预期关键词 {expected_keywords}，实际召回: {combined_retrieved_text}"
            assert top_score >= 0.08, f"[{qid}] 最高相关性得分 ({top_score:.4f}) 低于阈值 0.08"
            in_doc_hits += 1
        else:
            # 文档外无关问题：验证拒答保护或低相关度预警生效
            context_str = format_docs(retrieved, min_score_threshold=0.08)
            assert "提供的参考文档中未提及此信息" in context_str
            if qid == "Q10":
                assert top_score < 0.08
                assert "未检索到与提问紧密相关的有效资料片段" in context_str
            elif qid == "Q9":
                assert top_score < 0.15
                assert "【低相关度预警】" in context_str

    assert in_doc_hits == in_doc_total == 8
