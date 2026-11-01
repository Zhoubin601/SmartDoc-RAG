# -*- coding: utf-8 -*-
"""
模型适配与工厂模块
支持标准 OpenAI 兼容接口（如 DeepSeek、通义千问、OpenAI、硅基流动等）
本地 TF-IDF 与在线 Embedding 显式选择，不静默降级
"""
import re
import json
from urllib.parse import urlsplit
from pathlib import Path
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def mask_secret(text: str, secrets=()) -> str:
    """
    对含有 API Key 等敏感凭据的文本进行脱敏处理，防止敏感信息写入日志或向界面泄露
    例如: sk-1234567890abcdef -> sk-***cdef
    """
    if not text:
        return ""
    text = str(text)
    for secret in sorted((str(s) for s in secrets if s), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"sk-[a-zA-Z0-9_-]+", lambda m: (
        m[0][:7] + "****" + m[0][-4:] if len(m[0]) > 15 else "sk-****"
    ), text)
    text = re.sub(r"(?i)(bearer\s+)[^\s\"',;]+", r"\1[REDACTED]", text)
    return re.sub(
        r'''(?i)(["']?(?:api[_-]?key|access[_-]?token|token|authorization)["']?\s*[:=]\s*["']?)[^\s"',;}]+''',
        r"\1[REDACTED]", text,
    )


def validate_api_url(base_url: str) -> str:
    """拒绝会将凭据持久化到元数据或日志的 URL。"""
    value = (base_url or "https://api.openai.com/v1").rstrip("/")
    parts = urlsplit(value)
    if (parts.scheme not in {"https", "http"} or not parts.hostname
            or parts.username or parts.password or parts.query or parts.fragment):
        raise ValueError("API URL 必须为 HTTP(S) 地址，不能包含凭据、查询参数或片段。")
    return value


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
            safe_msg = mask_secret(str(e), self._secrets())
            raise EmbeddingAPIError(f"在线 Embedding 生成失败（文档向量化）: {safe_msg}") from None

    def embed_query(self, text: str) -> List[float]:
        try:
            return self.raw_embeddings.embed_query(text)
        except Exception as e:
            safe_msg = mask_secret(str(e), self._secrets())
            raise EmbeddingAPIError(f"在线 Embedding 生成失败（查询向量化）: {safe_msg}") from None

    def _secrets(self):
        key = getattr(self.raw_embeddings, "openai_api_key", None)
        return [key.get_secret_value() if hasattr(key, "get_secret_value") else key]


class LocalTfidfEmbeddings(Embeddings):
    """
    基于 scikit-learn 的本地 TF-IDF 字符 N-gram 适配器
    完全兼容 LangChain Embeddings 接口。
    字面相似度检索，不是深度语义 Embedding。由用户显式选择。
    
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
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=self.max_features)
        tfidf_matrix = vectorizer.fit_transform(texts).toarray()
        self.vectorizer = vectorizer
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
        state = self.export_state()
        (folder_path / "tfidf_state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    def export_state(self):
        return {
            "version": 1,
            "max_features": self.max_features,
            "_is_fitted": self._is_fitted,
            "vocabulary": {term: int(index) for term, index in self.vectorizer.vocabulary_.items()} if self._is_fitted else {},
            "idf": self.vectorizer.idf_.tolist() if self._is_fitted else [],
        }

    def load_state(self, folder_path: Path):
        """
        从指定目录恢复 TF-IDF 向量器内部状态
        """
        folder_path = Path(folder_path)
        state_file = folder_path / "tfidf_state.json"
        if not state_file.exists():
            raise FileNotFoundError("缺少 tfidf_state.json；旧 pickle 状态不安全，请重新上传文档建库。")
        self.import_state(json.loads(state_file.read_text(encoding="utf-8")))

    def import_state(self, state):
        vocab, weights = state["vocabulary"], state["idf"]
        if state.get("version") != 1 or type(state["max_features"]) is not int or state["max_features"] <= 0:
            raise ValueError("TF-IDF 状态配置损坏。")
        if (not isinstance(vocab, dict) or any(not isinstance(k, str) or type(v) is not int for k, v in vocab.items())
                or sorted(vocab.values()) != list(range(len(vocab)))):
            raise ValueError("TF-IDF 词表索引损坏。")
        idf = np.asarray(weights, dtype=float)
        if idf.ndim != 1 or len(idf) != len(vocab) or not np.isfinite(idf).all() or (idf <= 0).any():
            raise ValueError("TF-IDF IDF 权重损坏。")
        fitted = state["_is_fitted"]
        if type(fitted) is not bool or fitted != bool(vocab) or len(vocab) > state["max_features"]:
            raise ValueError("TF-IDF 拟合状态与词表不一致。")
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=state["max_features"])
        if fitted:
            vectorizer.vocabulary_ = vocab
            vectorizer.fixed_vocabulary_ = True
            vectorizer.idf_ = idf
        self.max_features, self._is_fitted, self.vectorizer = state["max_features"], fitted, vectorizer


def get_chat_model(
    api_key: str,
    base_url: str,
    model_name: str = "deepseek-flash",
    temperature: float = 0.3,
    streaming: bool = True,
    timeout: float = 60,
    max_retries: int = 0,
    max_tokens: Optional[int] = None,
    extra_body: Optional[dict] = None,
    http_client=None,
) -> ChatOpenAI:
    """
    构建 LangChain 兼容的 Chat 模型实例
    """
    if not api_key:
        raise ValueError("请提供有效的大模型 API Key！")

    try:
        endpoint = validate_api_url(base_url)
        if (extra_body is None and urlsplit(endpoint).hostname == "api.deepseek.com"
                and model_name in {"deepseek-flash", "deepseek-v4-pro"}):
            extra_body = {"thinking": {"type": "disabled"}}
        return ChatOpenAI(
            model=model_name, api_key=api_key, base_url=endpoint,
            temperature=temperature, streaming=streaming, timeout=timeout,
            max_retries=max_retries, max_tokens=max_tokens, extra_body=extra_body,
            http_client=http_client,
        )
    except Exception as exc:
        raise ValueError(f"大模型配置无效：{mask_secret(str(exc), [api_key])}") from None


def get_embedding_model(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: str = "text-embedding-3-small",
    use_local_fallback: bool = False
) -> Embeddings:
    """
    构建向量 Embedding 实例
    仅当 use_local_fallback=True 时使用本地；在线模式错误明确抛出。
    """
    # 如果明确指定使用本地模式，或没有提供 API Key
    if use_local_fallback:
        return LocalTfidfEmbeddings()

    if not api_key:
        raise EmbeddingAPIError("在线 Embedding 模式需要 API Key；也可手动选择本地 TF-IDF。")

    # 如果是 DeepSeek 官方接口，官方目前仅开放 /chat/completions，无 /embeddings 接口
    endpoint = validate_api_url(base_url)
    if urlsplit(endpoint).hostname in {"api.deepseek.com", "deepseek.com"}:
        raise EmbeddingAPIError("DeepSeek 官方未提供独立 Embedding API，请选择本地 TF-IDF 或配置其他 Embedding 服务。")

    try:
        raw_emb = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            base_url=endpoint,
            request_timeout=60,
            max_retries=0,
        )
        return SafeOpenAIEmbeddings(raw_emb)
    except Exception as e:
        safe_msg = mask_secret(str(e), [api_key])
        raise EmbeddingAPIError(f"初始化在线 Embedding 失败（未切换模式）: {safe_msg}") from None


def embedding_identity(embedding):
    """不含凭据的索引特征空间标识。"""
    if isinstance(embedding, LocalTfidfEmbeddings):
        return {"type": "LocalTfidfEmbeddings"}
    raw = embedding.raw_embeddings if isinstance(embedding, SafeOpenAIEmbeddings) else embedding
    identity = {"type": type(embedding).__name__}
    if isinstance(raw, OpenAIEmbeddings):
        identity.update(model=raw.model, base_url=validate_api_url(raw.openai_api_base))
    return identity


def embedding_label(embedding):
    identity = embedding_identity(embedding)
    if isinstance(embedding, LocalTfidfEmbeddings):
        return "本地 TF-IDF 字符检索（字面相似度）"
    return f"在线 Embedding：{identity.get('model', identity['type'])}"

