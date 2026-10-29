# -*- coding: utf-8 -*-
"""
模型适配与工厂模块
支持标准 OpenAI 兼容接口（如 DeepSeek、通义千问、OpenAI、硅基流动等）
以及智能容错的本地/在线 Embedding 向量嵌入支持
"""
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class LocalTfidfEmbeddings(Embeddings):
    """
    轻量级无依赖本地嵌入适配器（基于 TF-IDF + N-gram）
    完全兼容 LangChain Embeddings 接口。
    用于在用户使用 DeepSeek（官方暂无 Embedding 接口）或无独立 Embedding Key 时的智能保底，
    确保即使没有在线 Embedding 接口，向量知识库检索依然能够精准、秒级运行。
    """
    def __init__(self, max_features: int = 512):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=self.max_features
        )
        self._is_fitted = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # 对文本集合进行 fit_transform
        tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()
        self._is_fitted = True
        return tfidf_matrix.astype(float).tolist()

    def embed_query(self, text: str) -> List[float]:
        if not self._is_fitted:
            # 如果尚未拟合文档，临时用当前查询进行拟合（全零或者单位向量）
            return [0.0] * self.max_features
        vec = self.vectorizer.transform([text]).toarray()[0]
        return vec.astype(float).tolist()


def get_chat_model(
    api_key: str,
    base_url: str,
    model_name: str = "deepseek-chat",
    temperature: float = 0.3,
    streaming: bool = True
) -> ChatOpenAI:
    """
    构建 LangChain 兼容的 Chat 模型实例
    """
    if not api_key:
        raise ValueError("请提供有效的大模型 API Key！")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url if base_url else None,
        temperature=temperature,
        streaming=streaming
    )


def get_embedding_model(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: str = "text-embedding-3-small",
    use_local_fallback: bool = False
) -> Embeddings:
    """
    构建向量 Embedding 实例
    若指定使用本地轻量模式或 API Key 未提供，自动切换为 LocalTfidfEmbeddings
    """
    # 如果明确指定使用本地模式，或没有提供 API Key
    if use_local_fallback or not api_key:
        return LocalTfidfEmbeddings()

    # 如果是 DeepSeek 官方接口，官方目前仅开放 /chat/completions，无 /embeddings 接口
    if base_url and "deepseek.com" in base_url.lower():
        # 智能提示并使用内置轻量 LocalTfidfEmbeddings
        return LocalTfidfEmbeddings()

    try:
        return OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            base_url=base_url if base_url else None
        )
    except Exception:
        # 发生异常时优雅降级
        return LocalTfidfEmbeddings()
