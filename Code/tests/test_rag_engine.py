# -*- coding: utf-8 -*-
"""
RAG 引擎与多轮对话链路单元测试
使用 Fake/Mock ChatModel 进行稳定的自动化测试，无需网络或消耗 API Token
"""
import pytest
from typing import List, Optional, Any, Iterator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.documents import Document

from src.models import LocalTfidfEmbeddings
from src.vector_store import SmartVectorStore
from src.rag_engine import RAGEngine, format_docs, convert_chat_history


class MockChatModel(BaseChatModel):
    """
    轻量自包含 Mock Chat 模型，捕获发送给大模型的完整 Prompt 消息，
    支持同步调用与流式 Stream 生成。
    """
    response_text: str = "根据参考资料，系统默认端口为 8080。"
    captured_messages: List[BaseMessage] = []

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> ChatResult:
        self.captured_messages = list(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response_text))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        self.captured_messages = list(messages)
        for token in ["根据", "参考资料", "，", "系统默认端口为", " 8080。"]:
            yield ChatGenerationChunk(message=AIMessageChunk(content=token))

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"


@pytest.fixture
def sample_vector_store():
    emb = LocalTfidfEmbeddings()
    store = SmartVectorStore(embedding_function=emb)
    store.add_documents([
        Document(
            page_content="SmartDoc-RAG 核心系统默认监听端口为 8080，配置文件位于 config.py。",
            metadata={"source": "config.md", "page": 1}
        ),
        Document(
            page_content="系统支持多轮对话上下文指代理解，事实依据严格限定在参考片段中。",
            metadata={"source": "design.md", "page": 2}
        )
    ])
    return store


def test_rag_engine_direct_query(sample_vector_store):
    mock_llm = MockChatModel(response_text="系统默认监听端口为 8080。")
    engine = RAGEngine(llm=mock_llm, vector_store=sample_vector_store, top_k=2)

    result = engine.query("系统默认端口是多少？")

    assert result["answer"] == "系统默认监听端口为 8080。"
    assert len(result["source_documents"]) > 0
    top_doc, top_score = result["source_documents"][0]
    assert "8080" in top_doc.page_content
    assert top_score > 0.1

    # 验证传递给 Mock 大模型的 System Prompt 中包含检索到的上下文
    system_msg = mock_llm.captured_messages[0]
    assert "8080" in system_msg.content
    assert "config.md" in system_msg.content


def test_rag_engine_low_score_refusal_mechanism(sample_vector_store):
    """
    测试文档外无答案问题：
    当输入与知识库完全无关的问题时，提示词注入严谨拒答引导，模型诚实拒绝
    """
    mock_llm = MockChatModel(response_text="提供的参考文档中未提及此信息。")
    engine = RAGEngine(llm=mock_llm, vector_store=sample_vector_store, top_k=2, min_score_threshold=0.08)

    # 提问一个知识库中完全未提及的外星人话题
    result = engine.query("火星上有外星飞船吗？")

    system_msg = mock_llm.captured_messages[0]
    # 验证上下文明确指示拒答
    assert "未在当前知识库中检索到" in system_msg.content or "未检索到与提问紧密相关" in system_msg.content
    assert "提供的参考文档中未提及此信息" in system_msg.content


def test_rag_engine_multi_turn_history(sample_vector_store):
    """
    测试连续追问与指代理解（P0.2 核心要求）：
    向引擎传递多轮历史，验证历史会话被正确注入 LCEL 提示词链中
    """
    mock_llm = MockChatModel(response_text="它的配置文件位于 config.py。")
    engine = RAGEngine(llm=mock_llm, vector_store=sample_vector_store, top_k=2)

    chat_history = [
        {"role": "user", "content": "SmartDoc-RAG 系统的默认端口是多少？"},
        {"role": "assistant", "content": "默认端口为 8080。"}
    ]

    # 用户基于上一问进行追问，使用代词“它”
    follow_up_q = "它的配置文件在哪里？"
    result = engine.query(follow_up_q, chat_history=chat_history)

    assert result["answer"] == "它的配置文件位于 config.py。"
    # 检查发送给模型的完整消息列表中包含历史对话
    role_types = [type(m).__name__ for m in mock_llm.captured_messages]
    assert "SystemMessage" in role_types
    assert "HumanMessage" in role_types
    assert "AIMessage" in role_types

    # 确认最后一条消息为当前问题
    assert mock_llm.captured_messages[-1].content == follow_up_q


def test_rag_engine_stream_query(sample_vector_store):
    mock_llm = MockChatModel()
    engine = RAGEngine(llm=mock_llm, vector_store=sample_vector_store, top_k=2)

    stream_gen, sources = engine.stream_query("查询端口配置")
    tokens = list(stream_gen)

    assert len(tokens) > 0
    assert "".join(tokens) == "根据参考资料，系统默认端口为 8080。"
    assert len(sources) > 0
