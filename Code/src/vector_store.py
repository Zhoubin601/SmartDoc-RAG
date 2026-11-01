# -*- coding: utf-8 -*-
"""
向量知识库管理模块
实现文本分块向量化、持久化存储以及 Top-K 相似度检索
基于 NumPy 的本地余弦检索，兼容 LangChain VectorStore / Retriever 接口
"""
import json
import copy
import hashlib
import os
from uuid import uuid4
from pathlib import Path
from typing import List, Tuple, Optional, Any
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from .models import LocalTfidfEmbeddings, embedding_identity


class SmartVectorStore(VectorStore):
    """
    本地向量检索库
    继承自 LangChain 官方 VectorStore 规范，无需安装外部数据库驱动，
    支持余弦相似度（Cosine Similarity）与检索评分计算，支持全状态持久化校验与热加载。
    """
    def __init__(self, embedding_function: Embeddings):
        self.embedding_function = embedding_function
        self.documents: List[Document] = []
        self.vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> List[str]:
        texts = list(texts)
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("元数据数量必须与文本数量一致。")
        if not texts:
            return []

        docs = [
            Document(page_content=t, metadata=(metadatas[i] if metadatas else {}))
            for i, t in enumerate(texts)
        ]
        return self.add_documents(docs)

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        if not documents:
            return []

        offset = len(self.documents)
        combined_docs = self.documents + list(documents)
        local = isinstance(self.embedding_function, LocalTfidfEmbeddings)
        # 本地拟合也采用候选对象：失败不能改变旧词表。
        candidate = copy.deepcopy(self.embedding_function) if local else self.embedding_function
        to_embed = combined_docs if local else documents
        vectors = np.asarray(candidate.embed_documents([d.page_content for d in to_embed]), dtype=np.float32)
        self._validate_vectors(vectors, len(to_embed))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms > 0, norms, 1)
        if not local and offset:
            if vectors.shape[1] != self.vectors.shape[1]:
                raise ValueError("新增文档特征维度与已有向量库维度不一致。")
            vectors = np.vstack([self.vectors, vectors])
        self.documents, self.vectors, self.embedding_function = combined_docs, vectors, candidate
        return [str(i) for i in range(offset, len(combined_docs))]

    @staticmethod
    def _validate_vectors(vectors, count):
        if (vectors.ndim != 2 or vectors.shape[0] != count
                or (count > 0 and vectors.shape[1] == 0)
                or (count == 0 and vectors.shape != (0, 0))
                or not np.isfinite(vectors).all()):
            raise ValueError("向量库完整性损坏：矩阵维度、文档数量或有限数值不匹配。")

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        根据查询语句检索最相似的 Top-K 文档切片及相似度得分（余弦相似度：0~1）
        """
        if type(k) is not int or k <= 0:
            raise ValueError("Top-K 必须为正整数。")
        if self.vectors.size == 0 or not self.documents:
            return []

        query_vec = np.array(self.embedding_function.embed_query(query), dtype=np.float32)

        # 严格校验特征维度一致性
        if query_vec.ndim != 1 or not np.isfinite(query_vec).all():
            raise ValueError("查询向量必须是一维有限数值数组。")
        if query_vec.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"查询向量维度 ({query_vec.shape[0]}) 与知识库向量维度 ({self.vectors.shape[1]}) 不一致！"
                "可能原因：向量库在重新恢复后未加载匹配的词表模型，或更换了不同维度的 Embedding 模型。"
            )

        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        else:
            query_vec = np.zeros_like(query_vec)

        # 计算余弦相似度点积
        scores = np.dot(self.vectors, query_vec)
        # 截断由于浮点精度可能微超 1.0 的情况
        scores = np.clip(scores, -1.0, 1.0)
        top_k_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_indices:
            score = float(scores[idx])
            results.append((self.documents[idx], score))
        return results

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[Document]:
        results_with_score = self.similarity_search_with_score(query, k=k)
        return [doc for doc, _ in results_with_score]

    @staticmethod
    def snapshot_path(folder_path: Path) -> Path:
        folder = Path(folder_path)
        pointer = folder / "CURRENT"
        if not pointer.exists():
            return folder  # 旧目录只允许读取经过严格校验的 JSON 格式。
        generation = pointer.read_text(encoding="ascii").strip()
        if len(generation) != 32 or any(c not in "0123456789abcdef" for c in generation):
            raise ValueError("知识库快照指针损坏。")
        snapshot = folder / "snapshots" / generation
        if not snapshot.resolve().is_relative_to(folder.resolve()):
            raise ValueError("知识库快照路径无效。")
        return snapshot

    @classmethod
    def exists(cls, folder_path: Path) -> bool:
        folder = Path(folder_path)
        return (folder / "CURRENT").exists() or (folder / "docs.json").exists()

    def save_local(self, folder_path: Path):
        """新快照完整写出并验证后，原子切换指针；失败保留上一个版本。"""
        self._validate_vectors(self.vectors, len(self.documents))
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        generation = uuid4().hex
        snapshot = folder / "snapshots" / generation
        snapshot.mkdir(parents=True)
        data = {"doc_count": len(self.documents), "documents": [
            {"page_content": d.page_content, "metadata": d.metadata} for d in self.documents
        ]}
        (snapshot / "docs.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        np.save(snapshot / "vectors.npy", self.vectors, allow_pickle=False)
        files = ["docs.json", "vectors.npy"]
        if isinstance(self.embedding_function, LocalTfidfEmbeddings):
            self.embedding_function.save_state(snapshot)
            files.append("tfidf_state.json")
        meta = {
            "version": 2, "doc_count": len(self.documents),
            "vector_rows": self.vectors.shape[0], "feature_dim": self.vectors.shape[1],
            "embedding_type": type(self.embedding_function).__name__,
            "embedding_identity": embedding_identity(self.embedding_function),
            "sha256": {name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest() for name in files},
        }
        (snapshot / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        # 校验磁盘候选副本，整个过程不调用在线 Embedding API。
        candidate = copy.deepcopy(self.embedding_function) if isinstance(self.embedding_function, LocalTfidfEmbeddings) else self.embedding_function
        self.load_local(snapshot, candidate)
        pointer = folder / (".CURRENT-" + generation)
        pointer.write_text(generation, encoding="ascii")
        os.replace(pointer, folder / "CURRENT")

    @classmethod
    def load_local(cls, folder_path: Path, embedding_function: Optional[Embeddings] = None) -> "SmartVectorStore":
        folder = cls.snapshot_path(folder_path)
        if not (folder / "metadata.json").exists():
            raise ValueError("缺少知识库元数据，无法确认模型身份，请重新构建。")
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if meta.get("version") != 2:
            raise ValueError("旧版知识库无法安全恢复，请重新上传文档构建（不读取 pickle）。")
        data = json.loads((folder / "docs.json").read_text(encoding="utf-8"))
        documents = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data["documents"]]
        vectors = np.load(folder / "vectors.npy", allow_pickle=False)
        cls._validate_vectors(vectors, len(documents))
        if (data.get("doc_count") != len(documents) or meta.get("doc_count") != len(documents)
                or meta.get("vector_rows") != vectors.shape[0] or meta.get("feature_dim") != vectors.shape[1]):
            raise ValueError("知识库元数据的文档数、向量行数或特征维度不匹配。")
        identity = meta.get("embedding_identity")
        if not isinstance(identity, dict) or meta.get("embedding_type") != identity.get("type"):
            raise ValueError("Embedding 身份元数据损坏。")
        if embedding_function is None:
            if identity.get("type") != "LocalTfidfEmbeddings":
                raise ValueError("恢复在线知识库需要显式提供相同模型和端点的 Embedding 配置。")
            embedding_function = LocalTfidfEmbeddings()
        if embedding_identity(embedding_function) != identity:
            raise ValueError("Embedding 类型、模型或 API 地址与已保存知识库不一致，请选择原配置。")
        expected_files = {"docs.json", "vectors.npy"}
        if isinstance(embedding_function, LocalTfidfEmbeddings):
            expected_files.add("tfidf_state.json")
        if set(meta.get("sha256", {})) != expected_files:
            raise ValueError("知识库完整性校验清单损坏。")
        for name in expected_files:
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != meta["sha256"][name]:
                raise ValueError(f"知识库完整性校验失败：{name}")
        if isinstance(embedding_function, LocalTfidfEmbeddings):
            candidate = copy.deepcopy(embedding_function)
            candidate.load_state(folder)
            if (bool(documents) != candidate._is_fitted
                    or (documents and len(candidate.vectorizer.vocabulary_) != vectors.shape[1])):
                raise ValueError("TF-IDF 词表维度或拟合状态与向量矩阵不匹配。")
            embedding_function = candidate
        store = cls(embedding_function)
        store.documents, store.vectors = documents, vectors
        return store

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> "SmartVectorStore":
        store = cls(embedding_function=embedding)
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any
    ) -> "SmartVectorStore":
        store = cls(embedding_function=embedding)
        store.add_documents(documents)
        return store


class SmartRetriever(BaseRetriever):
    """
    LangChain 标准兼容 Retriever，支持 Top-K 与相似度过滤
    """
    vectorstore: SmartVectorStore
    k: int = 4
    score_threshold: Optional[float] = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        results = self.vectorstore.similarity_search_with_score(query, k=self.k)
        if self.score_threshold is not None:
            results = [(doc, score) for doc, score in results if score >= self.score_threshold]
        return [doc for doc, _ in results]

