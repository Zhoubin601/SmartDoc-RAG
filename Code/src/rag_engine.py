# -*- coding: utf-8 -*-
"""
LangChain LCEL 核心 RAG 检索问答引擎
构建端到端检索增强生成链，支持多轮对话、流式输出与答案知识源溯源
"""
from typing import List, Tuple, Generator, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.chat_models import BaseChatModel

from .vector_store import SmartVectorStore


def format_docs(docs: List[Document]) -> str:
    """
    格式化检索出的参考文档片段，附带来源标签
    """
    if not docs:
        return "暂无相关参考文档内容。"

    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知文件")
        page = doc.metadata.get("page", 1)
        formatted.append(f"【参考片段 {i+1}】（来源：{source}，页码/位置：{page}）\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)


class RAGEngine:
    """
    SmartDoc-RAG 核心问答引擎（基于 LangChain LCEL 链规范）
    """
    def __init__(self, llm: BaseChatModel, vector_store: SmartVectorStore, top_k: int = 4):
        self.llm = llm
        self.vector_store = vector_store
        self.top_k = top_k

        # 构建规范的提示词模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "你是一个专业、严谨的智能文档问答助手「SmartDoc-AI」。\n"
                "请严格依据以下提供的【参考资料】回答用户的提问。\n"
                "回答要求：\n"
                "1. 忠实于参考资料，条理清晰、层次分明，逻辑严密；\n"
                "2. 若参考资料中未包含与问题相关的信息，请诚实告知「提供的参考文档中未提及此信息」，切勿凭空编造；\n"
                "3. 在回答关键论点时，可自然注明信息引自哪份资料片段。\n\n"
                "【参考资料】：\n{context}"
            )),
            ("human", "{question}")
        ])

        # 构建 LangChain LCEL 核心执行链
        self.chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        检索相关文档并附带相似度评分
        """
        return self.vector_store.similarity_search_with_score(query, k=self.top_k)

    def query(self, question: str) -> Dict[str, Any]:
        """
        单次问答执行（返回答案文本与引用文档列表）
        """
        results_with_scores = self.retrieve_with_scores(question)
        relevant_docs = [doc for doc, _ in results_with_scores]
        context_str = format_docs(relevant_docs)

        # 执行 LCEL 链调用
        answer = self.chain.invoke({
            "context": context_str,
            "question": question
        })

        return {
            "answer": answer,
            "source_documents": results_with_scores,
            "raw_context": context_str
        }

    def stream_query(self, question: str) -> Tuple[Generator[str, None, None], List[Tuple[Document, float]]]:
        """
        流式生成问答结果（用于 Streamlit 打字机效果与即时交互）
        返回: (流式生成器, 命中的参考片段列表)
        """
        results_with_scores = self.retrieve_with_scores(question)
        relevant_docs = [doc for doc, _ in results_with_scores]
        context_str = format_docs(relevant_docs)

        stream = self.chain.stream({
            "context": context_str,
            "question": question
        })

        return stream, results_with_scores
