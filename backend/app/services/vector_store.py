"""Embeddings and retrieval.

Embeddings are a Hugging Face sentence-transformer running locally on CPU, via
LangChain's HuggingFaceEmbeddings. No API key, no rate limit, no network call in
the hot path. Retrieval goes to Pinecone if a key is set, otherwise to a local
numpy store with the same interface.
"""
from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from typing import Any

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import settings

_store = None
_store_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_embeddings():
    """all-MiniLM-L6-v2, ~90 MB, loaded once, normalised so cosine == dot."""
    from langchain_huggingface import HuggingFaceEmbeddings
    print("[vec] loading embedding model", settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embeddings().embed_documents(texts)


# ------------------------------------------------------------------ backends
class LocalStore:
    """Cosine search over an in-memory matrix, persisted to a .npz file."""

    def __init__(self, path: str):
        self.path = path
        self.vectors = np.zeros((0, settings.embedding_dim), dtype="float32")
        self.meta: list[dict[str, Any]] = []
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        blob = np.load(self.path, allow_pickle=False)
        self.vectors = blob["vectors"].astype("float32")
        self.meta = json.loads(str(blob["meta"]))
        print(f"[vec] local store loaded: {len(self.meta)} vectors")

    def _save(self):
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        np.savez(self.path, vectors=self.vectors,
                  meta=np.array(json.dumps(self.meta)))

    def add(self, namespace: str, ids: list[str], texts: list[str],
            metadatas: list[dict]):
        incoming = set(ids)
        keep = [i for i, m in enumerate(self.meta)
                if not (m["namespace"] == namespace and m["id"] in incoming)]
        if len(keep) != len(self.meta):
            self.vectors = self.vectors[keep]
            self.meta = [self.meta[i] for i in keep]

        new_vectors = np.array(embed_texts(texts), dtype="float32")
        self.vectors = np.vstack([self.vectors, new_vectors])
        for identifier, text, metadata in zip(ids, texts, metadatas):
            self.meta.append({"id": identifier, "namespace": namespace,
                               "text": text, **metadata})
        self._save()

    def search(self, namespace: str, query: str, k: int) -> list[tuple[Document, float]]:
        positions = [i for i, m in enumerate(self.meta) if m["namespace"] == namespace]
        if not positions:
            return []
        query_vector = np.array(get_embeddings().embed_query(query), dtype="float32")
        scores = self.vectors[positions] @ query_vector
        order = np.argsort(-scores)[: min(k, len(positions))]
        results = []
        for rank in order:
            meta = dict(self.meta[positions[int(rank)]])
            text = meta.pop("text", "")
            results.append((Document(page_content=text, metadata=meta),
                             float(scores[int(rank)])))
        return results


class PineconeStore:
    """One index, one LangChain vector store object per namespace."""

    def __init__(self):
        from pinecone import Pinecone, ServerlessSpec
        self.client = Pinecone(api_key=settings.pinecone_api_key)
        name = settings.pinecone_index_name
        existing = [info["name"] for info in self.client.list_indexes()]
        if name not in existing:
            print("[vec] creating pinecone index", name)
            self.client.create_index(
                name=name,
                dimension=settings.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud,
                                     region=settings.pinecone_region),
            )
        self.index = self.client.Index(name)
        self._stores: dict[str, Any] = {}

    def _namespace_store(self, namespace: str):
        from langchain_pinecone import PineconeVectorStore
        if namespace not in self._stores:
            self._stores[namespace] = PineconeVectorStore(
                index=self.index,
                embedding=get_embeddings(),
                text_key="text",
                namespace=namespace,
            )
        return self._stores[namespace]

    def add(self, namespace: str, ids: list[str], texts: list[str],
            metadatas: list[dict]):
        self._namespace_store(namespace).add_texts(
            texts=texts, metadatas=metadatas, ids=ids)

    def search(self, namespace: str, query: str, k: int) -> list[tuple[Document, float]]:
        return self._namespace_store(namespace).similarity_search_with_score(query, k=k)


def get_store():
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                if settings.pinecone_api_key:
                    try:
                        _store = PineconeStore()
                        print("[vec] backend = pinecone")
                    except Exception as exc:  # noqa: BLE001
                        print(f"[vec] pinecone unavailable ({exc}); using local store")
                        _store = LocalStore(settings.local_vector_path)
                else:
                    _store = LocalStore(settings.local_vector_path)
                    print("[vec] backend = local")
    return _store


# ---------------------------------------------------------------- public API
def add_texts(namespace: str, ids: list[str], texts: list[str],
              metadatas: list[dict]):
    get_store().add(namespace, ids, texts, metadatas)


def search(namespace: str, query: str, k: int) -> list[tuple[Document, float]]:
    return get_store().search(namespace, query, k)


# ------------------------------------------------- requirement #5: precedent
def retrieve_labelled(text: str, k: int | None = None) -> list[dict]:
    """Top-k previously-labelled feedback examples, for grounding the classifier."""
    k = k or settings.retrieval_k
    try:
        hits = search(settings.labelled_namespace, text, k)
    except Exception as exc:  # noqa: BLE001 - never let retrieval stop a classification
        print(f"[vec] retrieval failed, classifying ungrounded: {exc}")
        return []
    return [
        {
            "text": document.page_content,
            "label": document.metadata.get("label", "Good"),
            "score": round(float(score), 3),
        }
        for document, score in hits
    ]


def add_labelled(record_id: str, text: str, label: str, origin: str = "human"):
    add_texts(
        settings.labelled_namespace,
        ids=[record_id],
        texts=[text],
        metadatas=[{"label": label, "origin": origin}],
    )


# ------------------------------------- the separate company-knowledge feature
def add_knowledge(chunks: list[str], document_name: str, ids: list[str]):
    add_texts(
        settings.knowledge_namespace,
        ids=ids,
        texts=chunks,
        metadatas=[{"document_name": document_name, "chunk_id": i}
                   for i in range(len(chunks))],
    )


class NamespaceRetriever(BaseRetriever):
    """A real LangChain retriever over either backend, so rag_chat.py can compose
    it with LCEL instead of calling the store by hand."""

    namespace: str
    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        try:
            return [document for document, _ in search(self.namespace, query, self.k)]
        except Exception as exc:  # noqa: BLE001
            print(f"[vec] knowledge retrieval failed: {exc}")
            return []


def get_retriever(namespace: str, k: int = 4) -> BaseRetriever:
    return NamespaceRetriever(namespace=namespace, k=k)