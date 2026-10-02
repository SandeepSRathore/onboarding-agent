import json

import numpy as np
import pytest

from fakes import DIM, DOCS, fake_embed
from onboarding_agent.ingest import build_index, chunk_markdown


def test_splits_by_h2_and_uses_h1_as_title():
    chunks = chunk_markdown("01-setup", DOCS["01-setup"])
    assert [c.heading for c in chunks] == ["Install tools", "VPN"]
    assert all(c.doc_title == "Dev Setup" for c in chunks)
    assert [c.chunk_id for c in chunks] == ["01-setup#0", "01-setup#1"]


def test_hash_lines_inside_code_fences_are_not_headings():
    chunks = chunk_markdown("01-setup", DOCS["01-setup"])
    assert "# install everything" in chunks[0].text
    assert "make bootstrap" in chunks[0].text


def test_intro_text_before_first_h2_is_kept_under_doc_title():
    chunks = chunk_markdown("00-welcome", "# Welcome\n\nHello and welcome.\n\n## Day one\nBring your ID.\n")
    assert [(c.heading, c.text) for c in chunks] == [("Welcome", "Hello and welcome."), ("Day one", "Bring your ID.")]


def test_h4_and_hashtags_stay_body_text():
    chunks = chunk_markdown("d", "# T\n\n## S\n#### Note\n#nohashtag\n")
    assert len(chunks) == 1
    assert chunks[0].text == "#### Note\n#nohashtag"


def test_long_sections_are_split_on_paragraphs():
    paragraph = ("word " * 100).strip()  # 499 chars
    chunks = chunk_markdown("d", "# T\n\n## Big\n" + "\n\n".join([paragraph] * 5), max_chars=1200)
    assert len(chunks) == 3
    assert all(c.heading == "Big" for c in chunks)
    assert all(len(c.text) <= 1200 for c in chunks)


def test_embedding_text_includes_title_and_heading():
    chunk = chunk_markdown("02-leave", DOCS["02-leave"])[0]
    assert chunk.embedding_text().startswith("Leave Policy › Paid leave\n")


def test_build_index_writes_vectors_and_chunks(kb_dir, tmp_path):
    index_dir = tmp_path / "index"
    chunks = build_index(kb_dir, index_dir, fake_embed)
    assert len(chunks) == 4
    assert np.load(index_dir / "vectors.npy").shape == (4, DIM)
    saved = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    assert [c["doc_id"] for c in saved] == ["01-setup", "01-setup", "02-leave", "03-workflow"]


def test_build_index_fails_on_empty_knowledge_base(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No .md files"):
        build_index(tmp_path / "empty", tmp_path / "index", fake_embed)
