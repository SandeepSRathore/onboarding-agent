import pytest

from fakes import DOCS


@pytest.fixture
def kb_dir(tmp_path):
    directory = tmp_path / "kb"
    directory.mkdir()
    for doc_id, markdown in DOCS.items():
        (directory / f"{doc_id}.md").write_text(markdown, encoding="utf-8")
    return directory
