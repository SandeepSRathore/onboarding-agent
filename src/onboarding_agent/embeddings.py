"""Turning text into vectors. `Embed` is the only interface the rest of the app depends on."""
from collections.abc import Callable

import numpy as np

Embed = Callable[[list[str]], np.ndarray]


def openai_embedder(client, model: str, batch_size: int = 100) -> Embed:
    def embed(texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            response = client.embeddings.create(model=model, input=texts[start : start + batch_size])
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        return np.array(vectors, dtype=np.float32)

    return embed


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so a dot product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
