"""Semantic search over the saved index: cosine similarity with numpy, no vector DB needed."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import Embed, normalize
from .ingest import Chunk


class IndexNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embed: Embed):
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks = chunks
        self._vectors = normalize(np.asarray(vectors, dtype=np.float32))
        self._embed = embed

    @classmethod
    def load(cls, index_dir: Path, embed: Embed) -> "Retriever":
        vectors_path, chunks_path = index_dir / "vectors.npy", index_dir / "chunks.json"
        if not vectors_path.exists() or not chunks_path.exists():
            raise IndexNotFoundError(f"No search index in {index_dir}. Run `uv run ingest` first.")
        chunks = [Chunk(**raw) for raw in json.loads(chunks_path.read_text(encoding="utf-8"))]
        return cls(chunks, np.load(vectors_path), embed)

    def search(self, query: str, k: int = 4) -> list[SearchHit]:
        if not query.strip() or not self.chunks:
            return []
        query_vector = normalize(self._embed([query]))[0]
        scores = self._vectors @ query_vector
        top = np.argsort(-scores)[:k]
        return [SearchHit(self.chunks[i], float(scores[i])) for i in top]

    def doc_catalog(self) -> dict[str, str]:
        catalog: dict[str, str] = {}
        for chunk in self.chunks:
            catalog.setdefault(chunk.doc_id, chunk.doc_title)
        return catalog
