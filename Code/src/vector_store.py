# -*- coding: utf-8 -*-
"""
向量知识库管理模块
实现文本分块向量化、持久化存储以及 Top-K 相似度检索
支持自适应向量检索存储，零环境依赖，100% 兼容 LangChain VectorStore / Retriever 体系
"""
import json
from pathlib import Path
from typing import List, Tuple, Optional, Any
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class SmartVectorStore(VectorStore):
    """
    高性能、跨平台、自适应本地向量检索库
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

        from .models import LocalTfidfEmbeddings

        # 如果是基于词袋/TF-IDF 的嵌入适配器，增量添加时需要统合全部语料重新拟合，保持统一特征空间与维度
        if isinstance(self.embedding_function, LocalTfidfEmbeddings) and self.documents:
            combined_docs = list(self.documents) + list(documents)
            all_texts = [d.page_content for d in combined_docs]
            all_embeddings = self.embedding_function.embed_documents(all_texts)
            all_vecs = np.array(all_embeddings, dtype=np.float32)

            # 归一化便于计算余弦相似度
            norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            all_vecs = all_vecs / norms

            self.vectors = all_vecs
            self.documents = combined_docs
            return [str(i) for i in range(len(self.documents) - len(documents), len(self.documents))]

        # 常规稠密向量（固定维度，如 OpenAI 1536 维）
        new_texts = [d.page_content for d in documents]
        new_embeddings = self.embedding_function.embed_documents(new_texts)
        new_vecs = np.array(new_embeddings, dtype=np.float32)

        # 归一化便于计算余弦相似度
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        new_vecs = new_vecs / norms

        if self.vectors.size == 0:
            self.vectors = new_vecs
            self.documents = list(documents)
        else:
            if new_vecs.shape[1] != self.vectors.shape[1]:
                raise ValueError(
                    f"新增文档特征维度 ({new_vecs.shape[1]}) 与已有向量库维度 ({self.vectors.shape[1]}) 不一致！"
                )
            self.vectors = np.vstack([self.vectors, new_vecs])
            self.documents.extend(documents)

        return [str(i) for i in range(len(documents))]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        根据查询语句检索最相似的 Top-K 文档切片及相似度得分（余弦相似度：0~1）
        """
        if self.vectors.size == 0 or not self.documents:
            return []

        query_vec = np.array(self.embedding_function.embed_query(query), dtype=np.float32)

        # 严格校验特征维度一致性
        if self.vectors.ndim == 2 and query_vec.shape[0] != self.vectors.shape[1]:
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

    def save_local(self, folder_path: Path):
        """
        持久化保存向量、文档元数据、模型内部状态与特征维度元信息
        """
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)

        # 1. 保存文档列表
        data = {
            "doc_count": len(self.documents),
            "documents": [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in self.documents
            ]
        }
        with open(folder_path / "docs.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 2. 保存向量特征矩阵
        np.save(folder_path / "vectors.npy", self.vectors)

        # 3. 保存元数据描述信息
        meta = {
            "doc_count": len(self.documents),
            "vector_rows": int(self.vectors.shape[0]) if self.vectors.ndim == 2 else 0,
            "feature_dim": int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0,
            "embedding_type": self.embedding_function.__class__.__name__
        }
        with open(folder_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 4. 如果 embedding_function 包含状态保存能力（如 TF-IDF 词表和 IDF 权重）
        if hasattr(self.embedding_function, "save_state"):
            self.embedding_function.save_state(folder_path)

    @classmethod
    def load_local(
        cls,
        folder_path: Path,
        embedding_function: Optional[Embeddings] = None
    ) -> "SmartVectorStore":
        """
        从本地目录恢复向量库，包含严格的数据与维度一致性校验
        """
        folder_path = Path(folder_path)
        docs_path = folder_path / "docs.json"
        vecs_path = folder_path / "vectors.npy"
        meta_path = folder_path / "metadata.json"

        if not docs_path.exists() or not vecs_path.exists():
            raise FileNotFoundError(f"向量库目录缺失必要文件: docs.json 或 vectors.npy (路径: {folder_path})")

        with open(docs_path, "r", encoding="utf-8") as f:
            docs_data = json.load(f)

        documents = [
            Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
            for item in docs_data.get("documents", [])
        ]
        vectors = np.load(vecs_path)

        # 核心校验 1：文档数量与向量行数是否一致
        if len(documents) != vectors.shape[0]:
            raise ValueError(
                f"向量库完整性损坏：文档总数 ({len(documents)}) 与向量行数 ({vectors.shape[0]}) 不匹配，禁止继续加载！"
            )

        # 核心校验 2：检查元数据
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("doc_count") != len(documents):
                raise ValueError(
                    f"向量库元数据校验失败：记录的 doc_count ({meta.get('doc_count')}) 与实际解析数 ({len(documents)}) 不符！"
                )
            expected_dim = meta.get("feature_dim", 0)
            actual_dim = vectors.shape[1] if vectors.ndim == 2 else 0
            if expected_dim != actual_dim:
                raise ValueError(
                    f"向量库元数据校验失败：记录特征维度 ({expected_dim}) 与矩阵特征维度 ({actual_dim}) 不符！"
                )

        # 恢复或配置 Embedding 模型
        if embedding_function is None:
            from .models import LocalTfidfEmbeddings
            embedding_function = LocalTfidfEmbeddings()

        # 核心步骤：恢复 TF-IDF 向量器内部词表与权重状态
        if hasattr(embedding_function, "load_state"):
            state_file = folder_path / "tfidf_state.pkl"
            if not state_file.exists():
                raise FileNotFoundError(
                    "缺少 tfidf_state.pkl：该向量库可能由旧版本生成，无法恢复 TF-IDF 词表，请重新上传文档构建知识库。"
                )
            embedding_function.load_state(folder_path)

        store = cls(embedding_function=embedding_function)
        store.documents = documents
        store.vectors = vectors
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

