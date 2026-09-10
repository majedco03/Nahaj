"""Hybrid course retrieval from persistent Chroma collections."""

from __future__ import annotations

from collections import Counter
from math import log
from pathlib import Path
import os
import re
from typing import Any, Mapping, Sequence

from .app_config import settings
from .rag_ingestion import CourseIngestor, chromadb


class CourseRetriever:
    """Combine BM25 and Chroma cosine rankings with reciprocal rank fusion."""

    _TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        client: Any | None = None,
        embedding_function: Any | None = None,
    ):
        self.persist_dir = Path(
            persist_dir
            or os.getenv("CHROMA_DIR", "")
            or (settings.upload_dir / ".chroma")
        ).resolve()
        self.client = client or self._create_client()
        self.embedding_function = embedding_function

    def _create_client(self) -> Any:
        if chromadb is None:
            raise RuntimeError("chromadb is required for RAG retrieval")
        return chromadb.PersistentClient(path=str(self.persist_dir))

    def retrieve(self, query: str, course_id: Any, top_k: int = 3) -> list[dict[str, Any]]:
        """Return at most ``top_k`` chunks from one course's collection."""

        if top_k <= 0 or not course_id:
            return []
        try:
            options = {"name": CourseIngestor.collection_name(course_id)}
            if self.embedding_function is not None:
                options["embedding_function"] = self.embedding_function
            collection = self.client.get_collection(**options)
            count = collection.count()
        except Exception:
            return []
        if not count:
            return []

        stored = collection.get(include=["documents", "metadatas"])
        ids = list(stored.get("ids") or [])
        documents = list(stored.get("documents") or [])
        metadatas = list(stored.get("metadatas") or [])
        records = {
            chunk_id: {
                **(metadatas[index] if index < len(metadatas) and metadatas[index] else {}),
                "id": chunk_id,
                "text": documents[index] or "",
            }
            for index, chunk_id in enumerate(ids)
            if index < len(documents)
        }
        if not records:
            return []

        record_ids = list(records)
        record_index = {chunk_id: index for index, chunk_id in enumerate(record_ids)}
        texts = [records[chunk_id]["text"] for chunk_id in record_ids]
        bm25_scores = self._bm25(query, texts)
        bm25_order = sorted(range(len(record_ids)), key=lambda index: (-bm25_scores[index], index))

        vector_scores: dict[str, float] = {}
        vector_order: list[str] = []
        try:
            vector_result = collection.query(
                query_texts=[query],
                n_results=len(record_ids),
                include=["distances"],
            )
            returned_ids = (vector_result.get("ids") or [[]])[0]
            distances = (vector_result.get("distances") or [[]])[0]
            for chunk_id, distance in zip(returned_ids, distances):
                vector_order.append(chunk_id)
                vector_scores[chunk_id] = 1.0 - float(distance)
        except Exception:
            # BM25 remains useful if a collection has no embedding function configured.
            vector_order = list(record_ids)

        rrf_scores = {chunk_id: 0.0 for chunk_id in record_ids}
        for rank, index in enumerate(bm25_order, start=1):
            rrf_scores[record_ids[index]] += 1 / (60 + rank)
        for rank, chunk_id in enumerate(vector_order, start=1):
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] += 1 / (60 + rank)

        selected_ids = sorted(
            record_ids,
            key=lambda chunk_id: (
                -rrf_scores[chunk_id],
                -bm25_scores[record_index[chunk_id]],
                -vector_scores.get(chunk_id, 0.0),
            ),
        )[:top_k]
        return [
            {
                **records[chunk_id],
                "bm25_score": bm25_scores[record_index[chunk_id]],
                "cosine_score": vector_scores.get(chunk_id, 0.0),
                "rrf_score": rrf_scores[chunk_id],
            }
            for chunk_id in selected_ids
        ]

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        return [token.lower() for token in cls._TOKEN_RE.findall(text)]

    @classmethod
    def _bm25(cls, query: str, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
        query_terms = cls._tokens(query)
        document_terms = [cls._tokens(text) for text in texts]
        frequencies = Counter(term for terms in document_terms for term in set(terms))
        average_length = sum(map(len, document_terms)) / max(len(document_terms), 1)
        scores: list[float] = []
        for terms in document_terms:
            counts = Counter(terms)
            length_factor = 1 - b + b * len(terms) / max(average_length, 1)
            score = 0.0
            for term in query_terms:
                if not counts[term]:
                    continue
                idf = log(1 + (len(texts) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
                score += idf * counts[term] * (k1 + 1) / (counts[term] + k1 * length_factor)
            scores.append(score)
        return scores
