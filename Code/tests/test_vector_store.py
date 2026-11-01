# -*- coding: utf-8 -*-
"""
向量存储与检索模块单元测试
验证 TF-IDF 嵌入、余弦相似度计算、状态持久化与恢复、以及特征维度一致性校验
"""
import pytest
import numpy as np
from pathlib import Path
from langchain_core.documents import Document
from src.models import LocalTfidfEmbeddings
from src.vector_store import SmartVectorStore, SmartRetriever


def test_local_tfidf_embeddings():
    emb = LocalTfidfEmbeddings(max_features=128)
    texts = [
        "系统架构设计与模块划分说明",
        "数据库主从同步与网络端口配置",
        "前端页面响应式布局与样式组件"
    ]
    matrix = emb.embed_documents(texts)
    assert len(matrix) == 3
    assert len(matrix[0]) <= 128
    assert emb._is_fitted is True

    # 针对拟合后的词表进行查询嵌入
    query_vec = emb.embed_query("数据库端口")
    assert len(query_vec) == len(matrix[0])
    assert any(v > 0 for v in query_vec)


def test_smart_vector_store_add_and_search():
    emb = LocalTfidfEmbeddings(max_features=256)
    store = SmartVectorStore(embedding_function=emb)

    docs = [
        Document(page_content="PostgreSQL 默认监听端口为 5432，管理员账号为 postgres。", metadata={"source": "db.md", "page": 1}),
        Document(page_content="Redis 内存缓存系统默认监听端口为 6379，支持键值过期机制。", metadata={"source": "redis.md", "page": 1}),
        Document(page_content="Nginx 反向代理服务器负责 80 端口与 443 端口流量转发。", metadata={"source": "nginx.md", "page": 1})
    ]

    store.add_documents(docs)
    assert len(store.documents) == 3
    assert store.vectors.shape[0] == 3

    # 查询 PostgreSQL 端口
    results = store.similarity_search_with_score("PostgreSQL 端口是多少", k=2)
    assert len(results) == 2
    top_doc, top_score = results[0]
    assert "5432" in top_doc.page_content
    assert top_score > 0.1
    assert top_doc.metadata["source"] == "db.md"


def test_vector_store_save_and_load_persistence(tmp_path: Path):
    """
    回归测试（P0.1 核心要求）：
    构建索引 -> 保存 -> 销毁旧对象 -> 加载 -> 使用相同问题检索，比较恢复前后的命中文档与排序
    """
    emb = LocalTfidfEmbeddings(max_features=128)
    store = SmartVectorStore(embedding_function=emb)

    docs = [
        Document(page_content="Kubernetes 集群控制平面包含 API Server、etcd 与 Controller Manager。", metadata={"id": 1}),
        Document(page_content="Docker 容器镜像构建使用 Dockerfile，支持多阶段构建以精简体积。", metadata={"id": 2}),
        Document(page_content="Prometheus 监控服务通过 pull 模式抓取各组件指标。", metadata={"id": 3})
    ]
    store.add_documents(docs)

    test_query = "如何精简 Docker 镜像体积"
    pre_results = store.similarity_search_with_score(test_query, k=2)
    pre_top_doc, pre_top_score = pre_results[0]
    assert pre_top_doc.metadata["id"] == 2

    # 持久化到临时目录
    save_dir = tmp_path / "test_vector_db"
    store.save_local(save_dir)
    snapshot = SmartVectorStore.snapshot_path(save_dir)

    # 验证生成的文件清单
    assert (snapshot / "docs.json").exists()
    assert (snapshot / "vectors.npy").exists()
    assert (snapshot / "metadata.json").exists()
    assert (snapshot / "tfidf_state.json").exists()

    # 彻底销毁旧对象
    del store
    del emb

    # 从磁盘全新恢复
    loaded_store = SmartVectorStore.load_local(save_dir)

    # 验证数据完整性
    assert len(loaded_store.documents) == 3
    assert loaded_store.vectors.shape[0] == 3

    # 执行完全相同的检索
    post_results = loaded_store.similarity_search_with_score(test_query, k=2)
    post_top_doc, post_top_score = post_results[0]

    # 断言恢复前后的命中文档、排序及相似度评分一致
    assert post_top_doc.metadata["id"] == pre_top_doc.metadata["id"]
    assert post_top_doc.page_content == pre_top_doc.page_content
    assert pytest.approx(post_top_score, rel=1e-4) == pre_top_score


def test_vector_store_corrupted_data_raises(tmp_path: Path):
    """
    测试文档数量与向量行数不一致时，严格抛出 ValueError
    """
    emb = LocalTfidfEmbeddings(max_features=64)
    store = SmartVectorStore(embedding_function=emb)
    store.add_documents([
        Document(page_content="测试条目 A"),
        Document(page_content="测试条目 B")
    ])
    save_dir = tmp_path / "corrupted_db"
    store.save_local(save_dir)

    # 篡改 vectors.npy（改成 3 行，使之与 docs.json 的 2 篇文档不符）
    corrupted_vecs = np.zeros((3, store.vectors.shape[1]), dtype=np.float32)
    np.save(SmartVectorStore.snapshot_path(save_dir) / "vectors.npy", corrupted_vecs)

    with pytest.raises(ValueError, match="向量库完整性损坏|不匹配"):
        SmartVectorStore.load_local(save_dir)


def test_smart_retriever():
    emb = LocalTfidfEmbeddings(max_features=64)
    store = SmartVectorStore(embedding_function=emb)
    store.add_documents([
        Document(page_content="Git 代码版本控制常用分支管理规范包括 Git Flow 和 Trunk Based。")
    ])

    retriever = SmartRetriever(vectorstore=store, k=1, score_threshold=0.01)
    retrieved = retriever.invoke("Git 分支规范")
    assert len(retrieved) == 1
    assert "Git Flow" in retrieved[0].page_content
