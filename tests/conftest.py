import pytest

from fakes import DOCS, fake_embed
from onboarding_agent.ingest import build_index
from onboarding_agent.retriever import Retriever


@pytest.fixture
def kb_dir(tmp_path):
    directory = tmp_path / "kb"
    directory.mkdir()
    for doc_id, markdown in DOCS.items():
        (directory / f"{doc_id}.md").write_text(markdown, encoding="utf-8")
    return directory


@pytest.fixture
def retriever(kb_dir, tmp_path):
    build_index(kb_dir, tmp_path / "index", fake_embed)
    return Retriever.load(tmp_path / "index", fake_embed)
