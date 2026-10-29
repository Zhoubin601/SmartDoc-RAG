# -*- coding: utf-8 -*-
"""
模型适配与工厂模块
支持标准 OpenAI 兼容接口（如 DeepSeek、通义千问、OpenAI、硅基流动等）
以及智能容错的本地/在线 Embedding 向量嵌入支持
"""
import re
import pickle
from pathlib import Path
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def mask_secret(text: str) -> str:
    """
    对含有 API Key 等敏感凭据的文本进行脱敏处理，防止敏感信息写入日志或向界面泄露
    例如: sk-1234567890abcdef -> sk-***cdef
    """
    if not text:
        return ""
    # 匹配类似 sk-xxxx 的常见密钥格式
    pattern = r"(sk-[a-zA-Z0-9]{4})[a-zA-Z0-9_-]+([a-zA-Z0-9]{4})"
    return re.sub(pattern, r"\1****\2", str(text))


class EmbeddingAPIError(RuntimeError):
    """Embedding API 接口调用异常封装"""
    pass


class SafeOpenAIEmbeddings(Embeddings):
    """
    带异常捕获与凭据脱敏的安全 OpenAI 兼容 Embedding 包装类
    """
    def __init__(self, raw_embeddings: OpenAIEmbeddings):
        self.raw_embeddings = raw_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.raw_embeddings.embed_documents(texts)
        except Exception as e:
            safe_msg = mask_secret(str(e))
            raise EmbeddingAPIError(f"在线 Embedding 生成失败（文档向量化）: {safe_msg}") from None

    def embed_query(self, text: str) -> List[float]:
        try:
            return self.raw_embeddings.embed_query(text)
        except Exception as e:
            safe_msg = mask_secret(str(e))
            raise EmbeddingAPIError(f"在线 Embedding 生成失败（查询向量化）: {safe_msg}") from None


class LocalTfidfEmbeddings(Embeddings):
    """
    轻量级无依赖本地嵌入适配器（基于 TF-IDF + 字符 N-gram）
    完全兼容 LangChain Embeddings 接口。
    用于在用户使用 DeepSeek（官方暂无 Embedding 接口）或无独立 Embedding Key 时的智能保底，
    确保即使没有在线 Embedding 接口，向量知识库检索依然能够精准、秒级运行。
    
    支持状态完整持久化（词表、IDF 权重、特征配置）与热恢复，确保恢复后的查询向量与已建索引位于同一特征空间。
    """
    def __init__(self, max_features: int = 2048):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 3),
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
            # 尚未拟合文档时，安全返回全零向量，长度保持与 max_features 一致
            return [0.0] * self.max_features
        vec = self.vectorizer.transform([text]).toarray()[0]
        return vec.astype(float).tolist()

    def save_state(self, folder_path: Path):
        """
        持久化保存 TF-IDF 向量器状态（包括词表 vocabulary_、IDF 权重 idf_ 等）
        """
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        state_file = folder_path / "tfidf_state.pkl"
        state = {
            "max_features": self.max_features,
            "_is_fitted": self._is_fitted,
            "vectorizer": self.vectorizer
        }
        with open(state_file, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, folder_path: Path):
        """
        从指定目录恢复 TF-IDF 向量器内部状态
        """
        folder_path = Path(folder_path)
        state_file = folder_path / "tfidf_state.pkl"
        if not state_file.exists():
            raise FileNotFoundError(f"TF-IDF 状态文件不存在: {state_file}")
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        self.max_features = state.get("max_features", self.max_features)
        self._is_fitted = state.get("_is_fitted", False)
        self.vectorizer = state.get("vectorizer", self.vectorizer)


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
        raw_emb = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            base_url=base_url if base_url else None
        )
        return SafeOpenAIEmbeddings(raw_emb)
    except Exception as e:
        # 发生异常时优雅降级为本地 TF-IDF
        safe_msg = mask_secret(str(e))
        print(f"[警告] 初始化在线 Embedding 失败，降级为本地 TF-IDF 方案: {safe_msg}")
        return LocalTfidfEmbeddings()

