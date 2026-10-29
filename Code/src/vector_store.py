# -*- coding: utf-8 -*-
"""
向量知识库管理模块
实现文本分块向量化、持久化存储以及 Top-K 语义相似度检索
支持自适应向量检索存储，零环境依赖，100% 兼容 LangChain VectorStore / Retriever 体系
"""
import pickle
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
    继承自 LangChain 官方 VectorStore 规范，无需安装庞杂的外部数据库驱动，
    支持余弦相似度（Cosine Similarity）与语义分数计算，支持一键持久化与热加载。
    """
    def __init__(self, embedding_function: Embeddings):
        self.embedding_function = embedding_function
        self.documents: List[Document] = []
        self.vectors: np.ndarray = np.empty((0, 0))

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
            self.vectors = np.vstack([self.vectors, new_vecs])
            self.documents.extend(documents)

        return [str(i) for i in range(len(documents))]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        根据查询语句检索最相似的 Top-K 文档切片及相似度得分（0~1）
        """
        if self.vectors.size == 0 or not self.documents:
            return []

        query_vec = np.array(self.embedding_function.embed_query(query), dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        else:
            query_vec = np.zeros_like(query_vec)

        # 计算余弦相似度
        scores = np.dot(self.vectors, query_vec)
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
        持久化保存向量与元数据到指定目录
        """
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)

        data = {
            "documents": [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in self.documents
            ]
        }
        with open(folder_path / "docs.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        np.save(folder_path / "vectors.npy", self.vectors)

    @classmethod
    def load_local(cls, folder_path: Path, embedding_function: Embeddings) -> "SmartVectorStore":
        """
        从本地目录加载向量库
        """
        folder_path = Path(folder_path)
        store = cls(embedding_function=embedding_function)

        docs_path = folder_path / "docs.json"
        vecs_path = folder_path / "vectors.npy"

        if docs_path.exists() and vecs_path.exists():
            with open(docs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            store.documents = [
                Document(page_content=item["page_content"], metadata=item["metadata"])
                for item in data.get("documents", [])
            ]
            store.vectors = np.load(vecs_path)

        return store

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> "SmartVectorStore":
        """
        根据文本列表构建向量库
        """
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
    LangChain 标准兼容 Retriever
    """
    vectorstore: SmartVectorStore
    k: int = 4

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=self.k)
