"""
embeddings.py
-------------
Semantic similarity backend for comparing product names across languages.

Contains:
- EmbeddingCache                       – SQLite-backed cache so embeddings are computed once
- SemanticSimilarity                   – abstract base class
- LocalEmbeddingSemanticSimilarity     – sentence-transformers model (multilingual)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import List, Optional, Sequence

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding cache (SQLite)
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """Persistent key-value store for embedding vectors."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.commit()

    def get(self, key: str) -> Optional[np.ndarray]:
        row = self.conn.execute(
            "SELECT value FROM embeddings WHERE key = ?", (key,),
        ).fetchone()
        if not row:
            return None
        return np.array(json.loads(row[0]), dtype=np.float32)

    def set(self, key: str, value: np.ndarray) -> None:
        data = json.dumps(value.astype(float).tolist(), ensure_ascii=False)
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (key, value) VALUES (?, ?)",
            (key, data),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SemanticSimilarity:
    """Interface that all semantic backends implement."""

    def fit(self, texts: Sequence[str]) -> None:
        raise NotImplementedError

    def similarity(self, a_index: int, b_index: int) -> float:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Local sentence-transformers backend
# ---------------------------------------------------------------------------

_E5_MODEL_PREFIXES = ("intfloat/multilingual-e5-", "intfloat/e5-")


class LocalEmbeddingSemanticSimilarity(SemanticSimilarity):
    """Multilingual sentence-transformer model with SQLite embedding cache."""

    def __init__(self, model_name: str, cache_path: str) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required for multilingual matching. "
                "Install it with: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache = EmbeddingCache(cache_path)
        self.embeddings: List[np.ndarray] = []
        self._needs_prefix = any(
            model_name.startswith(p) for p in _E5_MODEL_PREFIXES
        )

    def _prepare_text(self, text: str) -> str:
        """E5 models expect a 'query: ' or 'passage: ' prefix."""
        if self._needs_prefix:
            return f"query: {text}"
        return text

    def fit(self, texts: Sequence[str]) -> None:
        self.embeddings = []
        missing_texts: List[str] = []
        missing_positions: List[int] = []

        for i, text in enumerate(texts):
            cached = self.cache.get(f"{self.model_name}::{text}")
            if cached is not None:
                self.embeddings.append(cached)
            else:
                self.embeddings.append(np.array([], dtype=np.float32))
                missing_texts.append(text)
                missing_positions.append(i)

        if missing_texts:
            logger.info("Computing embeddings for %d new texts", len(missing_texts))
            encode_texts = [self._prepare_text(t) for t in missing_texts]
            new_embeddings = self.model.encode(
                encode_texts, normalize_embeddings=True,
            )
            for pos, emb in zip(missing_positions, new_embeddings):
                emb_arr = np.asarray(emb, dtype=np.float32)
                self.embeddings[pos] = emb_arr
                self.cache.set(f"{self.model_name}::{texts[pos]}", emb_arr)

    def similarity(self, a_index: int, b_index: int) -> float:
        a, b = self.embeddings[a_index], self.embeddings[b_index]
        if a.size == 0 or b.size == 0:
            return 0.0
        value = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        return max(0.0, min(1.0, value))

    def close(self) -> None:
        self.cache.close()
