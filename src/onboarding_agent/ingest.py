"""Ingestion: split the handbook into heading-sized chunks, embed them, save the index."""
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import load_settings, openai_client
from .embeddings import Embed, openai_embedder

HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    heading: str
    text: str

    def embedding_text(self) -> str:
        return f"{self.doc_title} › {self.heading}\n{self.text}"


def chunk_markdown(doc_id: str, markdown: str, max_chars: int = 1500) -> list[Chunk]:
    """One chunk per H2/H3 section. The H1 is the doc title; long sections are split on blank lines."""
    doc_title = doc_id
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    buffer: list[str] = []
    in_fence = False

    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(line)
        if match is None:
            buffer.append(line)
            continue
        sections.append((heading, "\n".join(buffer).strip()))
        buffer = []
        level, title = len(match.group(1)), match.group(2)
        if level == 1:
            doc_title, heading = title, None
        else:
            heading = title
    sections.append((heading, "\n".join(buffer).strip()))

    chunks: list[Chunk] = []
    for section_heading, text in sections:
        if not text:
            continue
        for piece in _split(text, max_chars):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{len(chunks)}",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    heading=section_heading or doc_title,
                    text=piece,
                )
            )
    return chunks


def _split(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            pieces.append(current)
            current = paragraph
    if current:
        pieces.append(current)
    return pieces


def load_documents(kb_dir: Path) -> dict[str, str]:
    docs = {path.stem: path.read_text(encoding="utf-8") for path in sorted(kb_dir.glob("*.md"))}
    if not docs:
        raise FileNotFoundError(f"No .md files found in {kb_dir}")
    return docs


def build_index(kb_dir: Path, index_dir: Path, embed: Embed) -> list[Chunk]:
    chunks = [chunk for doc_id, markdown in load_documents(kb_dir).items() for chunk in chunk_markdown(doc_id, markdown)]
    vectors = embed([chunk.embedding_text() for chunk in chunks])
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "vectors.npy", vectors)
    (index_dir / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks], indent=2), encoding="utf-8")
    return chunks


def main() -> None:
    settings = load_settings()
    client = openai_client(settings)
    chunks = build_index(settings.kb_dir, settings.index_dir, openai_embedder(client, settings.embed_model))
    doc_count = len({chunk.doc_id for chunk in chunks})
    print(f"Indexed {len(chunks)} chunks from {doc_count} docs → {settings.index_dir}")
