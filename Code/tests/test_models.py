# -*- coding: utf-8 -*-
"""
模型工厂与安全脱敏模块单元测试
"""
import pytest
from src.models import (
    mask_secret,
    get_chat_model,
    get_embedding_model,
    LocalTfidfEmbeddings,
    SafeOpenAIEmbeddings,
    EmbeddingAPIError
)
from langchain_core.embeddings import Embeddings


def test_mask_secret():
    # 标准 OpenAI 格式密钥
    raw_key = "sk-1234567890abcdef1234567890abcdef"
    masked = mask_secret(raw_key)
    assert "sk-1234" in masked
    assert "cdef" in masked
    assert "sk-****" in masked or "****" in masked
    assert "567890abcdef1234" not in masked

    # 包含在错误信息中的异常文本
    err_text = "Authentication failed with key sk-abcdef0123456789xyz9876: unauthorized"
    masked_err = mask_secret(err_text)
    assert "sk-abcd****9876" in masked_err

    # 普通文本保持不变
    normal_text = "正常中文日志信息"
    assert mask_secret(normal_text) == normal_text


def test_get_chat_model_empty_key_raises():
    with pytest.raises(ValueError, match="请提供有效的大模型 API Key"):
        get_chat_model(api_key="", base_url="https://api.deepseek.com/v1")


def test_get_embedding_model_local_fallback():
    # 明确请求 local fallback
    emb = get_embedding_model(use_local_fallback=True)
    assert isinstance(emb, LocalTfidfEmbeddings)

    # 回归修复：在线模式缺少 Key 应报错，不能把自动降级当正确行为。
    with pytest.raises(EmbeddingAPIError, match="需要 API Key"):
        get_embedding_model(api_key=None)

    with pytest.raises(EmbeddingAPIError, match="未提供独立 Embedding"):
        get_embedding_model(api_key="sk-test", base_url="https://api.deepseek.com/v1")


def test_safe_openai_embeddings_masks_credentials():
    class DummyFailingEmbeddings(Embeddings):
        def embed_documents(self, texts):
            raise RuntimeError("Request failed for token sk-99998888777766665555444433332222: 401 Unauthorized")

        def embed_query(self, text):
            raise RuntimeError("Request failed for token sk-99998888777766665555444433332222: 401 Unauthorized")

    safe_emb = SafeOpenAIEmbeddings(DummyFailingEmbeddings())

    with pytest.raises(EmbeddingAPIError) as exc_info:
        safe_emb.embed_documents(["hello"])
    assert "sk-9999****2222" in str(exc_info.value)
    assert "sk-999988887777" not in str(exc_info.value)


def test_chat_factory_limits_and_provider_options():
    model = get_chat_model("dummy", "https://api.deepseek.com", streaming=True, timeout=12, max_tokens=256)
    assert model.max_retries == 0
    assert model.request_timeout == 12
    assert model.extra_body == {"thinking": {"type": "disabled"}}


def test_chat_factory_preserves_other_provider_options():
    model = get_chat_model("dummy", "https://example.org/v1", model_name="other-model")
    assert model.extra_body is None
