# Onboarding Buddy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a web-based AI agent that guides a fresher software engineer through onboarding at the fictional company "Nimbus Labs". It answers from a sample handbook with citations and tracks checklist progress.

**Architecture:** A hand-rolled tool-calling loop over the OpenAI Chat Completions API. The model can call five Python tools: `search_docs`, `read_doc`, `get_checklist`, `mark_task_done` and `find_contact`. Retrieval is numpy cosine similarity over OpenAI embeddings saved to disk. State (profile, progress, chat history) lives in SQLite. FastAPI serves a JSON API and a plain HTML/JS chat UI.

**Tech Stack:** Python ≥3.12, uv, `openai` (≥2.8), numpy, FastAPI + uvicorn, PyYAML, python-dotenv, pytest + httpx. Frontend: vanilla JS, with `marked` + `DOMPurify` from the jsDelivr CDN.

**Spec:** `docs/superpowers/specs/2026-10-02-onboarding-agent-design.md`

## Global Constraints

- No agent framework: no LangChain, LlamaIndex or OpenAI Agents SDK. The loop is our own code in `agent.py`.
- LLM calls use `client.chat.completions.create(model=..., messages=..., tools=TOOL_SCHEMAS, tool_choice=...)`.
- Embeddings use `client.embeddings.create(model=..., input=[...])`.
- Model choice lives in `.env`: `OPENAI_MODEL=gpt-5-mini` (cheap and reliable at tool calling; `gpt-5-nano` is the cheaper option to try), `OPENAI_REASONING_EFFORT=low` (sent as `reasoning_effort` only when non-empty, so it can be cleared for models without reasoning), and `OPENAI_EMBED_MODEL=text-embedding-3-small`. The code defaults mirror these, so a missing `.env` never falls back to an expensive model.
- The agent loop runs at most `MAX_AGENT_STEPS` (default 6) model calls that use tools. After that comes one final call with `tool_choice="none"`.
- No automated test may hit the network. Tests use `tests/fakes.py` (`fake_embed`, `FakeOpenAI`).
- Single local user: one employee profile per database. The server binds to `127.0.0.1`.
- All fictional email addresses use the `nimbuslabs.example` domain. No real company names as the employer.
- Handbook doc ids are fixed (see Task 9) because the checklist, eval set and tests reference them.

## Review Focus

1. **`#` lines inside fenced code blocks** (bash comments in setup guides) must stay body text, not become headings. Pinned in Task 2.
2. **Malformed tool calls from the model** (bad JSON, unknown tool, missing args, non-object args) must come back to the model as `{"error": ...}`, never crash the request. Pinned in Tasks 5 and 6.
3. **Start date today or in the future** must give "day 1" on the start date and "start in N day(s)" before it, never "day 0" or "day -2". Pinned in Tasks 4 and 6.
4. **Path-like `doc_id`s** (`../.env`) passed to `read_doc` must be rejected; only ids from the index are readable. Pinned in Task 5.
5. **OpenAI outage mid-chat** must return HTTP 502 with a friendly message and must not save a half-finished exchange to history. Pinned in Tasks 6 and 7.

---

## File Structure

```
onboarding-agent/
  pyproject.toml, .gitignore, .env.example, README.md
  knowledge_base/*.md              # 10 sample handbook docs (Task 9)
  data/checklist.yaml              # onboarding tasks (Task 9)
  data/contacts.json               # who to ask (Task 9)
  data/index/                      # generated: vectors.npy + chunks.json (gitignored)
  data/onboarding.db               # generated SQLite (gitignored)
  evals/golden.yaml                # retrieval eval questions (Task 9)
  src/onboarding_agent/
    __init__.py
    config.py        # Settings from env; PROJECT_ROOT; require_api_key
    embeddings.py    # Embed type, openai_embedder, normalize
    ingest.py        # Chunk, chunk_markdown, load_documents, build_index, main (CLI)
    retriever.py     # Retriever (load/search/doc_catalog), SearchHit, IndexNotFoundError
    store.py         # Employee (+ day_on), Store (SQLite)
    tools.py         # TOOL_SCHEMAS, ChecklistTask, Contact, ToolResult, Toolbox, loaders
    prompts.py       # build_system_prompt
    agent.py         # Agent loop, AgentReply
    server.py        # create_app, build_default_app, main (CLI)
    eval.py          # evaluate, main (CLI)
  web/index.html, web/app.js, web/styles.css
  tests/fakes.py, tests/conftest.py, tests/test_*.py
```

---

### Task 1: Project scaffold, config, embeddings, test fakes

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `src/onboarding_agent/__init__.py`, `src/onboarding_agent/config.py`, `src/onboarding_agent/embeddings.py`, `tests/fakes.py`
- Test: `tests/test_config_and_embeddings.py`

**Interfaces:**
- Produces:
  - `config.PROJECT_ROOT: Path`
  - `config.Settings` (frozen dataclass with fields `openai_api_key: str | None, model: str, embed_model: str, kb_dir: Path, data_dir: Path, index_dir: Path, db_path: Path, web_dir: Path, max_agent_steps: int, reasoning_effort: str | None`)
  - `config.load_settings() -> Settings`
  - `config.require_api_key(settings) -> str` (raises `SystemExit` if missing)
  - `embeddings.Embed = Callable[[list[str]], np.ndarray]`
  - `embeddings.openai_embedder(client, model: str, batch_size: int = 100) -> Embed`
  - `embeddings.normalize(vectors: np.ndarray) -> np.ndarray`
  - `tests/fakes.py`: `DIM`, `fake_embed`, `tool_call`, `completion`, `FakeOpenAI`, `DOCS`, `TODAY`, `CHECKLIST_ROWS`, `CONTACT_ROWS`

- [ ] **Step 1: Create project files**

`pyproject.toml`:
```toml
[project]
name = "onboarding-agent"
version = "0.1.0"
description = "Onboarding Buddy: an AI agent that guides new engineers using company docs"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
  "openai>=2.8",
  "numpy>=2.1",
  "fastapi>=0.115",
  "uvicorn>=0.32",
  "pyyaml>=6.0",
  "python-dotenv>=1.0",
]

[project.scripts]
ingest = "onboarding_agent.ingest:main"
serve = "onboarding_agent.server:main"
eval = "onboarding_agent.eval:main"

[dependency-groups]
dev = ["pytest>=8.3", "httpx>=0.27"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/onboarding_agent"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]
```

`.gitignore`:
```
.venv/
__pycache__/
.pytest_cache/
.env
data/index/
data/onboarding.db
```

`.env.example`:
```
OPENAI_API_KEY=sk-your-key-here
# Cheapest model that handles multi-tool calling reliably. Try gpt-5-nano to cut cost further.
OPENAI_MODEL=gpt-5-mini
# low keeps hidden reasoning tokens (and cost) down. Leave empty for models without reasoning.
OPENAI_REASONING_EFFORT=low
OPENAI_EMBED_MODEL=text-embedding-3-small
MAX_AGENT_STEPS=6
```

Also create the real `.env` (gitignored): `cp .env.example .env`. The user pastes their key into it later.

`src/onboarding_agent/__init__.py`:
```python
"""Onboarding Buddy: an AI agent that guides new engineers using company docs."""
```

Create a placeholder `README.md` containing only `# Onboarding Buddy`; Task 10 replaces it. Hatchling requires the file to exist.

- [ ] **Step 2: Write test fakes** in `tests/fakes.py`

```python
"""Network-free stand-ins for OpenAI, plus a tiny sample handbook used across tests."""
import hashlib
import re
from datetime import date
from types import SimpleNamespace

import numpy as np

DIM = 256
TODAY = date(2026, 10, 7)

DOCS = {
    "01-setup": (
        "# Dev Setup\n\n"
        "## Install tools\nInstall Docker and uv with Homebrew.\n\n"
        "```bash\n# install everything\nmake bootstrap\n```\n\n"
        "## VPN\nConnect to the Tailscale VPN before cloning repos.\n"
    ),
    "02-leave": "# Leave Policy\n\n## Paid leave\nYou get 24 days of paid leave per year, requested in PeopleHub.\n",
    "03-workflow": (
        "# Engineering Workflow\n\n"
        "## Pull requests\nEvery pull request needs one approval and green CI before squash merge.\n"
    ),
}

CHECKLIST_ROWS = [
    ("setup-laptop", 1, "Set up your laptop", "01-setup"),
    ("vpn", 1, "Connect to VPN", "01-setup"),
    ("first-pr", 1, "Open your first PR", "03-workflow"),
    ("leave-policy", 2, "Read the leave policy", "02-leave"),
]

CONTACT_ROWS = [
    ("IT Helpdesk", "IT support", "#it-help on Slack", ("laptop", "vpn", "access", "github", "password")),
    ("Neha Kapoor", "HR Business Partner", "hr@nimbuslabs.example", ("leave", "payroll", "benefits", "holidays")),
]


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic bag-of-words embedding: each word adds 1 to one of DIM hashed buckets."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
            out[row, bucket] += 1.0
    return out


def tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def completion(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAI:
    """Mimics `client.chat.completions.create`.

    Returns queued responses in order (raising any queued exception) and records each call's kwargs.
    """

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self._responses:
            raise AssertionError("FakeOpenAI ran out of queued responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
```

- [ ] **Step 3: Write the failing tests** in `tests/test_config_and_embeddings.py`

```python
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from onboarding_agent.config import PROJECT_ROOT, load_settings, require_api_key
from onboarding_agent.embeddings import normalize, openai_embedder


def test_settings_read_env_and_point_inside_project(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "my-model")
    monkeypatch.setenv("MAX_AGENT_STEPS", "3")
    settings = load_settings()
    assert settings.model == "my-model"
    assert settings.max_agent_steps == 3
    assert settings.kb_dir == PROJECT_ROOT / "knowledge_base"
    assert settings.index_dir == PROJECT_ROOT / "data" / "index"


def test_empty_reasoning_effort_means_none(monkeypatch):
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "")
    assert load_settings().reasoning_effort is None
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "medium")
    assert load_settings().reasoning_effort == "medium"


def test_require_api_key_exits_with_help_when_missing():
    settings = dataclasses.replace(load_settings(), openai_api_key=None)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        require_api_key(settings)


def test_normalize_handles_zero_rows():
    out = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_openai_embedder_batches_and_keeps_input_order():
    calls = []

    def create(model, input):
        calls.append(list(input))
        # Return items out of order to prove we sort by `index`.
        data = [SimpleNamespace(index=i, embedding=[float(len(t)), 1.0]) for i, t in reversed(list(enumerate(input)))]
        return SimpleNamespace(data=data)

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    out = openai_embedder(client, "m", batch_size=2)(["a", "bb", "ccc"])
    assert calls == [["a", "bb"], ["ccc"]]
    assert out[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert out.dtype == np.float32
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv sync && uv run pytest tests/test_config_and_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onboarding_agent.config'`

- [ ] **Step 5: Implement** `src/onboarding_agent/config.py`

```python
"""Runtime settings, read from environment variables (and a local .env file)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    model: str
    embed_model: str
    kb_dir: Path
    data_dir: Path
    index_dir: Path
    db_path: Path
    web_dir: Path
    max_agent_steps: int
    reasoning_effort: str | None


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    data_dir = PROJECT_ROOT / "data"
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        kb_dir=PROJECT_ROOT / "knowledge_base",
        data_dir=data_dir,
        index_dir=data_dir / "index",
        db_path=data_dir / "onboarding.db",
        web_dir=PROJECT_ROOT / "web",
        max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "6")),
        reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "low") or None,
    )


def require_api_key(settings: Settings) -> str:
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return settings.openai_api_key
```

`src/onboarding_agent/embeddings.py`:
```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config_and_embeddings.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit** (only if the user approved git; see the handoff)

```bash
git add -A && git commit -m "chore: scaffold project with config, embeddings and test fakes"
```

---

### Task 2: Markdown chunking and index building

**Files:**
- Create: `src/onboarding_agent/ingest.py`, `tests/conftest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `embeddings.Embed`, `config.load_settings`, `config.require_api_key`, `embeddings.openai_embedder`
- Produces:
  - `ingest.Chunk` (frozen dataclass with `chunk_id, doc_id, doc_title, heading, text: str`, plus `embedding_text() -> str`)
  - `ingest.chunk_markdown(doc_id: str, markdown: str, max_chars: int = 1500) -> list[Chunk]`
  - `ingest.load_documents(kb_dir: Path) -> dict[str, str]`
  - `ingest.build_index(kb_dir: Path, index_dir: Path, embed: Embed) -> list[Chunk]`, which writes `vectors.npy` and `chunks.json`
  - `ingest.main()`
  - `conftest` fixture `kb_dir`

- [ ] **Step 1: Write the shared fixture** in `tests/conftest.py`

```python
import pytest

from fakes import DOCS


@pytest.fixture
def kb_dir(tmp_path):
    directory = tmp_path / "kb"
    directory.mkdir()
    for doc_id, markdown in DOCS.items():
        (directory / f"{doc_id}.md").write_text(markdown, encoding="utf-8")
    return directory
```

- [ ] **Step 2: Write the failing tests** in `tests/test_ingest.py`

```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onboarding_agent.ingest'`

- [ ] **Step 4: Implement** `src/onboarding_agent/ingest.py`

```python
"""Ingestion: split the handbook into heading-sized chunks, embed them, save the index."""
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

from .config import load_settings, require_api_key
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
    client = OpenAI(api_key=require_api_key(settings))
    chunks = build_index(settings.kb_dir, settings.index_dir, openai_embedder(client, settings.embed_model))
    doc_count = len({chunk.doc_id for chunk in chunks})
    print(f"Indexed {len(chunks)} chunks from {doc_count} docs → {settings.index_dir}")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: chunk handbook markdown and build embedding index"
```

---

### Task 3: Retriever

**Files:**
- Create: `src/onboarding_agent/retriever.py`
- Modify: `tests/conftest.py` (add the `retriever` fixture)
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `ingest.Chunk`, `ingest.build_index`, `embeddings.Embed`, `embeddings.normalize`
- Produces:
  - `retriever.IndexNotFoundError(FileNotFoundError)`
  - `retriever.SearchHit` (frozen dataclass with `chunk: Chunk, score: float`)
  - `retriever.Retriever(chunks, vectors, embed)`, with `.chunks: list[Chunk]`, `Retriever.load(index_dir, embed)`, `.search(query: str, k: int = 4) -> list[SearchHit]` and `.doc_catalog() -> dict[str, str]` (doc_id → title, in index order)
  - `conftest` fixture `retriever`

- [ ] **Step 1: Add the fixture** to `tests/conftest.py` (append, and extend the imports)

```python
from fakes import fake_embed
from onboarding_agent.ingest import build_index
from onboarding_agent.retriever import Retriever


@pytest.fixture
def retriever(kb_dir, tmp_path):
    build_index(kb_dir, tmp_path / "index", fake_embed)
    return Retriever.load(tmp_path / "index", fake_embed)
```

- [ ] **Step 2: Write the failing tests** in `tests/test_retriever.py`

```python
import numpy as np
import pytest

from fakes import fake_embed
from onboarding_agent.retriever import IndexNotFoundError, Retriever


def test_search_ranks_relevant_chunk_first(retriever):
    hits = retriever.search("How many days of paid leave do I get?")
    assert hits[0].chunk.doc_id == "02-leave"
    assert hits[0].score > hits[-1].score


def test_search_finds_vpn_section(retriever):
    assert retriever.search("How do I connect to the VPN?")[0].chunk.heading == "VPN"


def test_search_respects_k_and_small_indexes(retriever):
    assert len(retriever.search("vpn", k=2)) == 2
    assert len(retriever.search("vpn", k=50)) == len(retriever.chunks) == 4


def test_blank_query_returns_nothing(retriever):
    assert retriever.search("   ") == []


def test_doc_catalog_lists_each_doc_once_in_order(retriever):
    assert retriever.doc_catalog() == {
        "01-setup": "Dev Setup",
        "02-leave": "Leave Policy",
        "03-workflow": "Engineering Workflow",
    }


def test_load_without_index_raises_helpful_error(tmp_path):
    with pytest.raises(IndexNotFoundError, match="uv run ingest"):
        Retriever.load(tmp_path / "missing", fake_embed)


def test_mismatched_chunks_and_vectors_rejected(retriever):
    with pytest.raises(ValueError):
        Retriever(retriever.chunks, np.zeros((1, 3)), fake_embed)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onboarding_agent.retriever'`

- [ ] **Step 4: Implement** `src/onboarding_agent/retriever.py`

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: add cosine-similarity retriever over saved index"
```

---

### Task 4: SQLite store and employee profile

**Files:**
- Create: `src/onboarding_agent/store.py`
- Modify: `tests/conftest.py` (add the `store` and `employee` fixtures)
- Test: `tests/test_store.py`

**Interfaces:**
- Produces:
  - `store.Employee` (frozen dataclass with `name, role, team: str, start_date: date`, plus `.day_on(today: date) -> int`; day 1 is the start date, and it is ≤0 before then)
  - `store.Store(db_path: Path | str)` (`":memory:"` allowed), with `.get_employee() -> Employee | None`, `.save_employee(e)`, `.completed_task_ids() -> set[str]`, `.mark_done(task_id)` (idempotent), `.add_message(role, content)` and `.recent_messages(limit: int = 20) -> list[dict[str, str]]` (oldest first)
  - `conftest` fixtures `store` and `employee` (Asha, Software Engineer, Platform, start 2026-10-05, which is day 3 on `TODAY`)

- [ ] **Step 1: Add the fixtures** to `tests/conftest.py`

```python
from datetime import date

from onboarding_agent.store import Employee, Store


@pytest.fixture
def store():
    return Store(":memory:")


@pytest.fixture
def employee():
    return Employee(name="Asha", role="Software Engineer", team="Platform", start_date=date(2026, 10, 5))
```

- [ ] **Step 2: Write the failing tests** in `tests/test_store.py`

```python
from dataclasses import replace
from datetime import date

from onboarding_agent.store import Store


def test_no_employee_initially(store):
    assert store.get_employee() is None


def test_employee_roundtrip_and_update(store, employee):
    store.save_employee(employee)
    assert store.get_employee() == employee
    store.save_employee(replace(employee, team="Growth"))
    assert store.get_employee().team == "Growth"


def test_day_on_counts_start_date_as_day_one(employee):
    assert employee.day_on(date(2026, 10, 5)) == 1
    assert employee.day_on(date(2026, 10, 7)) == 3
    assert employee.day_on(date(2026, 10, 4)) == 0
    assert employee.day_on(date(2026, 10, 3)) == -1


def test_mark_done_is_idempotent(store):
    store.mark_done("vpn")
    store.mark_done("vpn")
    store.mark_done("first-pr")
    assert store.completed_task_ids() == {"vpn", "first-pr"}


def test_recent_messages_returns_last_n_oldest_first(store):
    for i in range(5):
        store.add_message("user", f"q{i}")
        store.add_message("assistant", f"a{i}")
    assert store.recent_messages(limit=3) == [
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "q4"},
        {"role": "assistant", "content": "a4"},
    ]


def test_data_persists_across_instances(tmp_path, employee):
    path = tmp_path / "nested" / "onboarding.db"
    first = Store(path)
    first.save_employee(employee)
    first.mark_done("vpn")
    second = Store(path)
    assert second.get_employee() == employee
    assert second.completed_task_ids() == {"vpn"}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onboarding_agent.store'`

- [ ] **Step 4: Implement** `src/onboarding_agent/store.py`

```python
"""SQLite persistence: the new hire's profile, checklist progress and chat history."""
import sqlite3
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS employee (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    team TEXT NOT NULL,
    start_date TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS task_progress (
    task_id TEXT PRIMARY KEY,
    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass(frozen=True)
class Employee:
    name: str
    role: str
    team: str
    start_date: date

    def day_on(self, today: date) -> int:
        """Day 1 is the start date; zero or negative means they haven't started yet."""
        return (today - self.start_date).days + 1


class Store:
    def __init__(self, db_path: Path | str):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # FastAPI runs sync endpoints in a thread pool, so share one connection behind a lock.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)

    def get_employee(self) -> Employee | None:
        with self._lock:
            row = self._conn.execute("SELECT name, role, team, start_date FROM employee WHERE id = 1").fetchone()
        if row is None:
            return None
        name, role, team, start_date = row
        return Employee(name, role, team, date.fromisoformat(start_date))

    def save_employee(self, employee: Employee) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO employee (id, name, role, team, start_date) VALUES (1, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET name = excluded.name, role = excluded.role,
                   team = excluded.team, start_date = excluded.start_date""",
                (employee.name, employee.role, employee.team, employee.start_date.isoformat()),
            )

    def completed_task_ids(self) -> set[str]:
        with self._lock:
            return {row[0] for row in self._conn.execute("SELECT task_id FROM task_progress")}

    def mark_done(self, task_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR IGNORE INTO task_progress (task_id) VALUES (?)", (task_id,))

    def add_message(self, role: str, content: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))

    def recent_messages(self, limit: int = 20) -> list[dict[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: add SQLite store for profile, progress and chat history"
```

---

### Task 5: Agent tools (Toolbox)

**Files:**
- Create: `src/onboarding_agent/tools.py`
- Modify: `tests/conftest.py` (add the `toolbox` fixture)
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `Retriever.search`, `Retriever.doc_catalog`, the `Store` methods, `Employee.day_on`
- Produces:
  - `tools.TOOL_SCHEMAS: list[dict]` (Chat Completions `tools` format)
  - `tools.ChecklistTask` (frozen with `id: str, week: int, title: str, doc_ref: str | None = None`)
  - `tools.Contact` (frozen with `name, role, channel: str, topics: tuple[str, ...]`)
  - `tools.ToolResult` (with `content: str` (JSON), `sources: list[dict]`; each source is `{"doc_id","doc_title","section"}`)
  - `tools.load_checklist(path) -> list[ChecklistTask]` (YAML with a top-level `tasks:` list)
  - `tools.load_contacts(path) -> list[Contact]` (a JSON list)
  - `tools.Toolbox(retriever, store, checklist, contacts, kb_dir, today=date.today)`, with `.retriever`, `.tool_names: list[str]`, `.run(name: str, arguments_json: str) -> ToolResult` (never raises on bad input) and `.checklist_status() -> dict` (keys: `day: int | None, completed: int, total: int, tasks: list[dict], next_tasks: list[dict]`; each task dict has `id, week, title, doc_ref, done`)
  - `conftest` fixture `toolbox` (today = `TODAY`)

- [ ] **Step 1: Add the fixture** to `tests/conftest.py`

```python
from fakes import CHECKLIST_ROWS, CONTACT_ROWS, TODAY
from onboarding_agent.tools import ChecklistTask, Contact, Toolbox


@pytest.fixture
def toolbox(retriever, store, kb_dir):
    checklist = [ChecklistTask(*row) for row in CHECKLIST_ROWS]
    contacts = [Contact(*row) for row in CONTACT_ROWS]
    return Toolbox(retriever, store, checklist, contacts, kb_dir, today=lambda: TODAY)
```

- [ ] **Step 2: Write the failing tests** in `tests/test_tools.py`

```python
import json

from onboarding_agent.tools import TOOL_SCHEMAS, load_checklist, load_contacts


def run(toolbox, name, **arguments):
    result = toolbox.run(name, json.dumps(arguments))
    return json.loads(result.content), result


def test_schemas_match_handlers(toolbox):
    assert [s["function"]["name"] for s in TOOL_SCHEMAS] == toolbox.tool_names
    assert toolbox.tool_names == ["search_docs", "read_doc", "get_checklist", "mark_task_done", "find_contact"]


def test_search_docs_returns_sections_and_sources(toolbox):
    data, result = run(toolbox, "search_docs", query="paid leave days")
    assert data["results"][0]["doc_id"] == "02-leave"
    assert "24 days" in data["results"][0]["text"]
    assert result.sources[0] == {"doc_id": "02-leave", "doc_title": "Leave Policy", "section": "Paid leave"}


def test_search_docs_blank_query(toolbox):
    data, result = run(toolbox, "search_docs", query="  ")
    assert data["results"] == [] and result.sources == []


def test_read_doc_returns_full_markdown(toolbox):
    data, result = run(toolbox, "read_doc", doc_id="01-setup")
    assert "make bootstrap" in data["content"] and data["truncated"] is False
    assert result.sources == [{"doc_id": "01-setup", "doc_title": "Dev Setup", "section": "Dev Setup"}]


def test_read_doc_rejects_unknown_and_path_like_ids(toolbox):
    for bad in ["nope", "../.env", "01-setup.md"]:
        data, result = run(toolbox, "read_doc", doc_id=bad)
        assert "Valid ids: 01-setup, 02-leave, 03-workflow" in data["error"]
        assert result.sources == []


def test_get_checklist_reports_day_progress_and_next_tasks(toolbox, store, employee):
    store.save_employee(employee)
    store.mark_done("setup-laptop")
    data, _ = run(toolbox, "get_checklist")
    assert data["day"] == 3
    assert (data["completed"], data["total"]) == (1, 4)
    assert [t["id"] for t in data["next_tasks"]] == ["vpn", "first-pr", "leave-policy"]
    assert data["tasks"][0] == {"id": "setup-laptop", "week": 1, "title": "Set up your laptop", "doc_ref": "01-setup", "done": True}


def test_get_checklist_without_profile_has_no_day(toolbox):
    data, _ = run(toolbox, "get_checklist")
    assert data["day"] is None


def test_mark_task_done_updates_progress(toolbox, store):
    data, _ = run(toolbox, "mark_task_done", task_id="vpn")
    assert data["ok"] is True and data["task"] == "Connect to VPN"
    assert (data["completed"], data["total"]) == (1, 4)
    assert store.completed_task_ids() == {"vpn"}


def test_mark_task_done_unknown_id_lists_valid_ids(toolbox, store):
    data, _ = run(toolbox, "mark_task_done", task_id="fly-to-moon")
    assert "setup-laptop" in data["error"]
    assert store.completed_task_ids() == set()


def test_find_contact_matches_topic_words(toolbox):
    data, _ = run(toolbox, "find_contact", topic="My VPN is not working")
    assert data["matches"][0]["name"] == "IT Helpdesk"
    assert data["matches"][0]["channel"] == "#it-help on Slack"


def test_find_contact_without_match_returns_everyone(toolbox):
    data, _ = run(toolbox, "find_contact", topic="quantum physics")
    assert data["matches"] == []
    assert {c["name"] for c in data["all_contacts"]} == {"IT Helpdesk", "Neha Kapoor"}


def test_bad_tool_calls_return_errors_instead_of_raising(toolbox):
    cases = [
        ("delete_everything", "{}", "Unknown tool"),
        ("search_docs", "{not json", "not valid JSON"),
        ("search_docs", "[1, 2]", "JSON object"),
        ("search_docs", "{}", "Bad arguments"),
        ("search_docs", '{"query": "x", "extra": 1}', "Bad arguments"),
    ]
    for name, arguments, message in cases:
        data = json.loads(toolbox.run(name, arguments).content)
        assert message in data["error"], (name, arguments)


def test_empty_arguments_string_means_no_arguments(toolbox):
    assert "tasks" in json.loads(toolbox.run("get_checklist", "").content)


def test_loaders(tmp_path):
    (tmp_path / "c.yaml").write_text(
        "tasks:\n  - id: vpn\n    week: 1\n    title: Connect to VPN\n    doc_ref: 05-tools-and-access\n", encoding="utf-8"
    )
    (tmp_path / "c.json").write_text(
        '[{"name": "IT", "role": "Support", "channel": "#it-help", "topics": ["vpn"]}]', encoding="utf-8"
    )
    [task] = load_checklist(tmp_path / "c.yaml")
    assert (task.id, task.week, task.doc_ref) == ("vpn", 1, "05-tools-and-access")
    [contact] = load_contacts(tmp_path / "c.json")
    assert contact.topics == ("vpn",)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onboarding_agent.tools'`

- [ ] **Step 4: Implement** `src/onboarding_agent/tools.py`

```python
"""The agent's tools: the JSON schemas the model sees, plus the Python that runs them."""
import inspect
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import yaml

from .retriever import Retriever
from .store import Store

READ_DOC_MAX_CHARS = 12_000
SEARCH_RESULTS = 4


def _function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _function(
        "search_docs",
        "Semantic search over the Nimbus Labs handbook. Call this before answering any company-specific "
        "question. Returns the most relevant sections with their doc_id, title and section name.",
        {"query": {"type": "string", "description": "What to look for, as a question or keywords."}},
        ["query"],
    ),
    _function(
        "read_doc",
        "Read a whole handbook document, e.g. a step-by-step setup guide. Use a doc_id from the handbook list.",
        {"doc_id": {"type": "string", "description": "Document id such as '01-dev-environment-setup'."}},
        ["doc_id"],
    ),
    _function(
        "get_checklist",
        "Get the new hire's onboarding checklist: every task with its id and done status, "
        "progress counts, today's onboarding day number, and the next tasks to do.",
        {},
        [],
    ),
    _function(
        "mark_task_done",
        "Mark an onboarding checklist task as completed once the user says they finished it.",
        {"task_id": {"type": "string", "description": "Task id from get_checklist, e.g. 'vpn'."}},
        ["task_id"],
    ),
    _function(
        "find_contact",
        "Find the right person or channel to ask about a topic the handbook does not fully answer.",
        {"topic": {"type": "string", "description": "The topic or problem, e.g. 'laptop not booting'."}},
        ["topic"],
    ),
]


@dataclass(frozen=True)
class ChecklistTask:
    id: str
    week: int
    title: str
    doc_ref: str | None = None


@dataclass(frozen=True)
class Contact:
    name: str
    role: str
    channel: str
    topics: tuple[str, ...]


@dataclass
class ToolResult:
    content: str
    sources: list[dict] = field(default_factory=list)


def load_checklist(path: Path) -> list[ChecklistTask]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [ChecklistTask(**item) for item in raw["tasks"]]


def load_contacts(path: Path) -> list[Contact]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Contact(c["name"], c["role"], c["channel"], tuple(c["topics"])) for c in raw]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _error(message: str) -> ToolResult:
    return ToolResult(json.dumps({"error": message}))


class Toolbox:
    def __init__(
        self,
        retriever: Retriever,
        store: Store,
        checklist: list[ChecklistTask],
        contacts: list[Contact],
        kb_dir: Path,
        today: Callable[[], date] = date.today,
    ):
        self.retriever = retriever
        self.store = store
        self.checklist = checklist
        self.contacts = contacts
        self.kb_dir = Path(kb_dir)
        self._today = today
        self._handlers: dict[str, Callable[..., ToolResult]] = {
            "search_docs": self._search_docs,
            "read_doc": self._read_doc,
            "get_checklist": self._get_checklist,
            "mark_task_done": self._mark_task_done,
            "find_contact": self._find_contact,
        }

    @property
    def tool_names(self) -> list[str]:
        return list(self._handlers)

    def run(self, name: str, arguments_json: str) -> ToolResult:
        """Execute one tool call. Problems are reported back to the model as {"error": ...}, never raised."""
        handler = self._handlers.get(name)
        if handler is None:
            return _error(f"Unknown tool '{name}'. Available: {', '.join(self._handlers)}")
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as exc:
            return _error(f"Arguments were not valid JSON: {exc}")
        if not isinstance(arguments, dict):
            return _error("Arguments must be a JSON object.")
        try:
            inspect.signature(handler).bind(**arguments)
        except TypeError as exc:
            return _error(f"Bad arguments for {name}: {exc}")
        return handler(**arguments)

    def checklist_status(self) -> dict:
        done = self.store.completed_task_ids()
        employee = self.store.get_employee()
        tasks = [{**asdict(task), "done": task.id in done} for task in self.checklist]
        remaining = [task for task in tasks if not task["done"]]
        return {
            "day": employee.day_on(self._today()) if employee else None,
            "completed": len(tasks) - len(remaining),
            "total": len(tasks),
            "tasks": tasks,
            "next_tasks": remaining[:3],
        }

    def _search_docs(self, query: str) -> ToolResult:
        hits = self.retriever.search(str(query), k=SEARCH_RESULTS)
        results = [
            {
                "doc_id": hit.chunk.doc_id,
                "doc_title": hit.chunk.doc_title,
                "section": hit.chunk.heading,
                "score": round(hit.score, 3),
                "text": hit.chunk.text,
            }
            for hit in hits
        ]
        sources = [{key: r[key] for key in ("doc_id", "doc_title", "section")} for r in results]
        return ToolResult(json.dumps({"results": results}), sources)

    def _read_doc(self, doc_id: str) -> ToolResult:
        catalog = self.retriever.doc_catalog()
        doc_id = str(doc_id)
        if doc_id not in catalog:  # also blocks path tricks like "../.env"
            return _error(f"Unknown doc_id '{doc_id}'. Valid ids: {', '.join(catalog)}")
        text = (self.kb_dir / f"{doc_id}.md").read_text(encoding="utf-8")
        title = catalog[doc_id]
        payload = {
            "doc_id": doc_id,
            "doc_title": title,
            "content": text[:READ_DOC_MAX_CHARS],
            "truncated": len(text) > READ_DOC_MAX_CHARS,
        }
        return ToolResult(json.dumps(payload), [{"doc_id": doc_id, "doc_title": title, "section": title}])

    def _get_checklist(self) -> ToolResult:
        return ToolResult(json.dumps(self.checklist_status()))

    def _mark_task_done(self, task_id: str) -> ToolResult:
        task = next((t for t in self.checklist if t.id == str(task_id)), None)
        if task is None:
            valid = ", ".join(t.id for t in self.checklist)
            return _error(f"Unknown task_id '{task_id}'. Valid ids: {valid}")
        self.store.mark_done(task.id)
        status = self.checklist_status()
        return ToolResult(
            json.dumps(
                {
                    "ok": True,
                    "task": task.title,
                    "completed": status["completed"],
                    "total": status["total"],
                    "next_tasks": status["next_tasks"],
                }
            )
        )

    def _find_contact(self, topic: str) -> ToolResult:
        wanted = _words(str(topic))
        scored = []
        for contact in self.contacts:
            overlap = len(wanted & _words(" ".join([contact.name, contact.role, *contact.topics])))
            if overlap:
                scored.append((overlap, contact))
        scored.sort(key=lambda pair: -pair[0])
        matches = [asdict(contact) for _, contact in scored[:3]]
        if matches:
            return ToolResult(json.dumps({"matches": matches}))
        return ToolResult(
            json.dumps(
                {
                    "matches": [],
                    "all_contacts": [asdict(c) for c in self.contacts],
                    "note": "No direct match. Pick the most relevant contact, or suggest the onboarding buddy.",
                }
            )
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tools.py -v`
Expected: 14 passed

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: add agent tools for docs, checklist and contacts"
```

---

### Task 6: System prompt and the agent loop

**Files:**
- Create: `src/onboarding_agent/prompts.py`, `src/onboarding_agent/agent.py`
- Test: `tests/test_prompts.py`, `tests/test_agent.py`

**Interfaces:**
- Consumes: `Employee`, `Store`, `Toolbox.run`, `Toolbox.retriever.doc_catalog()`, `TOOL_SCHEMAS`
- Produces:
  - `prompts.build_system_prompt(employee: Employee | None, today: date, doc_catalog: dict[str, str]) -> str`
  - `agent.AgentReply` (with `text: str, sources: list[dict], tools_used: list[str]`)
  - `agent.Agent(client, model: str, toolbox: Toolbox, store: Store, max_steps: int = 6, reasoning_effort: str | None = None, today=date.today)`, with `.chat(user_message: str) -> AgentReply`. It saves the user and assistant messages only on success. OpenAI exceptions propagate.
  - `agent.STEP_LIMIT_NOTE`, `agent.EMPTY_REPLY_FALLBACK`

- [ ] **Step 1: Write the failing prompt tests** in `tests/test_prompts.py`

```python
from datetime import date

from fakes import TODAY
from onboarding_agent.prompts import build_system_prompt


def test_prompt_includes_profile_day_and_docs(employee):
    prompt = build_system_prompt(employee, TODAY, {"01-setup": "Dev Setup"})
    assert "Asha" in prompt and "Platform team" in prompt
    assert "day 3 of their onboarding" in prompt
    assert "- 01-setup: Dev Setup" in prompt
    assert "2026-10-07" in prompt


def test_prompt_before_start_date(employee):
    assert "they start in 2 day(s)" in build_system_prompt(employee, date(2026, 10, 3), {})


def test_prompt_on_start_date_is_day_one(employee):
    assert "day 1 of their onboarding" in build_system_prompt(employee, date(2026, 10, 5), {})


def test_prompt_without_profile(employee):
    assert "has not shared their profile" in build_system_prompt(None, TODAY, {})
```

- [ ] **Step 2: Write the failing agent tests** in `tests/test_agent.py`

```python
import json

import pytest
from openai import OpenAIError

from fakes import TODAY, FakeOpenAI, completion, tool_call
from onboarding_agent.agent import EMPTY_REPLY_FALLBACK, STEP_LIMIT_NOTE, Agent
from onboarding_agent.tools import TOOL_SCHEMAS


@pytest.fixture
def make_agent(toolbox, store, employee):
    store.save_employee(employee)

    def make(responses, max_steps=6, reasoning_effort=None):
        client = FakeOpenAI(responses)
        agent = Agent(
            client, "test-model", toolbox, store,
            max_steps=max_steps, reasoning_effort=reasoning_effort, today=lambda: TODAY,
        )
        return agent, client

    return make


def test_plain_answer_without_tools(make_agent):
    agent, client = make_agent([completion("Welcome aboard!")])
    reply = agent.chat("hi")
    assert (reply.text, reply.sources, reply.tools_used) == ("Welcome aboard!", [], [])
    call = client.calls[0]
    assert call["model"] == "test-model" and call["tools"] == TOOL_SCHEMAS and call["tool_choice"] == "auto"
    assert "Asha" in call["messages"][0]["content"] and "day 3" in call["messages"][0]["content"]
    assert call["messages"][-1] == {"role": "user", "content": "hi"}


def test_reasoning_effort_sent_only_when_configured(make_agent):
    agent, client = make_agent([completion("ok")])
    agent.chat("hi")
    assert "reasoning_effort" not in client.calls[0]
    agent, client = make_agent([completion("ok")], reasoning_effort="low")
    agent.chat("hi")
    assert client.calls[0]["reasoning_effort"] == "low"


def test_runs_tool_then_answers_with_sources(make_agent):
    agent, client = make_agent([
        completion(tool_calls=[tool_call("c1", "search_docs", '{"query": "paid leave"}')]),
        completion("You get 24 days [Leave Policy › Paid leave]."),
    ])
    reply = agent.chat("How much leave do I get?")
    assert reply.text.startswith("You get 24 days")
    assert reply.tools_used == ["search_docs"]
    assert reply.sources[0]["doc_id"] == "02-leave"
    assert len({s["doc_id"] for s in reply.sources}) == len(reply.sources)  # de-duplicated by doc
    assistant_turn, tool_turn = client.calls[1]["messages"][-2:]
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["tool_calls"][0] == {
        "id": "c1", "type": "function", "function": {"name": "search_docs", "arguments": '{"query": "paid leave"}'}
    }
    assert tool_turn["role"] == "tool" and tool_turn["tool_call_id"] == "c1"
    assert "24 days" in tool_turn["content"]


def test_multiple_tool_calls_in_one_turn_all_get_answered(make_agent, store):
    agent, client = make_agent([
        completion(tool_calls=[
            tool_call("a", "mark_task_done", '{"task_id": "vpn"}'),
            tool_call("b", "get_checklist", "{}"),
        ]),
        completion("Marked VPN as done!"),
    ])
    agent.chat("I connected to the VPN")
    assert store.completed_task_ids() == {"vpn"}
    tool_ids = [m["tool_call_id"] for m in client.calls[1]["messages"] if m["role"] == "tool"]
    assert tool_ids == ["a", "b"]


def test_tool_errors_go_back_to_the_model(make_agent):
    agent, client = make_agent([
        completion(tool_calls=[tool_call("x", "hack_the_planet", "{}")]),
        completion("Sorry, let me try something else."),
    ])
    reply = agent.chat("do something weird")
    assert reply.text == "Sorry, let me try something else."
    assert "Unknown tool" in json.loads(client.calls[1]["messages"][-1]["content"])["error"]


def test_step_limit_forces_a_final_answer_without_tools(make_agent):
    looping = completion(tool_calls=[tool_call("c", "get_checklist", "{}")])
    agent, client = make_agent([looping, looping, completion("Partial answer")], max_steps=2)
    reply = agent.chat("loop forever")
    assert reply.text == "Partial answer" + STEP_LIMIT_NOTE
    assert len(client.calls) == 3 and client.calls[-1]["tool_choice"] == "none"


def test_empty_model_reply_gets_fallback(make_agent):
    agent, _ = make_agent([completion("")])
    assert agent.chat("hello?").text == EMPTY_REPLY_FALLBACK


def test_history_is_saved_and_replayed(make_agent, store):
    agent, client = make_agent([completion("one"), completion("two")])
    agent.chat("first")
    agent.chat("second")
    assert client.calls[1]["messages"][1:] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "second"},
    ]
    assert len(store.recent_messages()) == 4


def test_api_failure_propagates_and_saves_nothing(make_agent, store):
    agent, _ = make_agent([OpenAIError("service down")])
    with pytest.raises(OpenAIError):
        agent.chat("hello")
    assert store.recent_messages() == []
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_prompts.py tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onboarding_agent.prompts'`

- [ ] **Step 4: Implement** `src/onboarding_agent/prompts.py`

```python
"""The system prompt: who the agent is, who it's helping, and the rules it must follow."""
from datetime import date

from .store import Employee

SYSTEM_TEMPLATE = """You are "Onboarding Buddy", a friendly guide for new software engineers joining Nimbus Labs.

{who}
Today's date: {today}.

How to help:
- The user is likely a fresher. Briefly explain company jargon and engineering terms the first time you use them.
- Be concise: short paragraphs, bullet lists and concrete next steps. Format with Markdown.
- For ANY question about Nimbus Labs (tools, processes, policies, people, codebase), call search_docs first. Use read_doc when you need a whole document, such as a step-by-step setup guide.
- Answer only from the handbook. Cite each fact like [Doc Title › Section]. Never invent policies, URLs, commands, numbers or credentials.
- If the handbook does not answer the question, say so plainly, then call find_contact and tell the user who to ask and how to reach them.
- When the user says they finished something, call get_checklist to find the matching task id, call mark_task_done, and confirm what you marked.
- When the user asks what to do next (or seems unsure), call get_checklist and suggest the next one or two tasks, pointing to the relevant doc.
- General programming questions (e.g. "what is a pull request?") can be answered from your own knowledge; then add how Nimbus Labs does it if the handbook covers it.

Handbook documents (doc_id: title):
{docs}"""


def build_system_prompt(employee: Employee | None, today: date, doc_catalog: dict[str, str]) -> str:
    if employee is None:
        who = "The new hire has not shared their profile yet."
    else:
        day = employee.day_on(today)
        when = f"today is day {day} of their onboarding" if day >= 1 else f"they start in {1 - day} day(s)"
        who = (
            f"You are helping {employee.name}, a new {employee.role} on the {employee.team} team; "
            f"{when} (start date {employee.start_date.isoformat()})."
        )
    docs = "\n".join(f"- {doc_id}: {title}" for doc_id, title in doc_catalog.items())
    return SYSTEM_TEMPLATE.format(who=who, today=today.isoformat(), docs=docs)
```

- [ ] **Step 5: Implement** `src/onboarding_agent/agent.py`

```python
"""The agent loop: call the model, run any tools it asks for, repeat until it answers."""
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from .prompts import build_system_prompt
from .store import Store
from .tools import TOOL_SCHEMAS, Toolbox

STEP_LIMIT_NOTE = "\n\n_(I hit my research step limit, so this answer may be incomplete. Try a more specific question.)_"
EMPTY_REPLY_FALLBACK = "Sorry, I couldn't come up with an answer. Could you rephrase your question?"


@dataclass
class AgentReply:
    text: str = ""
    sources: list[dict] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        client,
        model: str,
        toolbox: Toolbox,
        store: Store,
        max_steps: int = 6,
        reasoning_effort: str | None = None,
        today: Callable[[], date] = date.today,
    ):
        self.client = client
        self.model = model
        self.toolbox = toolbox
        self.store = store
        self.max_steps = max_steps
        self.reasoning_effort = reasoning_effort
        self._today = today

    def chat(self, user_message: str) -> AgentReply:
        system = build_system_prompt(self.store.get_employee(), self._today(), self.toolbox.retriever.doc_catalog())
        messages = [
            {"role": "system", "content": system},
            *self.store.recent_messages(),
            {"role": "user", "content": user_message},
        ]
        reply = AgentReply()

        for _ in range(self.max_steps):
            message = self._complete(messages, tool_choice="auto")
            if not message.tool_calls:
                reply.text = message.content or ""
                break
            messages.append(_assistant_turn(message))
            for call in message.tool_calls:
                result = self.toolbox.run(call.function.name, call.function.arguments)
                reply.tools_used.append(call.function.name)
                _merge_sources(reply.sources, result.sources)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result.content})
        else:
            # Out of steps: force a final text answer from what the model has gathered so far.
            final = self._complete(messages, tool_choice="none")
            reply.text = (final.content or "") + STEP_LIMIT_NOTE

        if not reply.text.strip():
            reply.text = EMPTY_REPLY_FALLBACK
        self.store.add_message("user", user_message)
        self.store.add_message("assistant", reply.text)
        return reply

    def _complete(self, messages: list[dict], tool_choice: str):
        options = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=TOOL_SCHEMAS, tool_choice=tool_choice, **options
        )
        return response.choices[0].message


def _assistant_turn(message) -> dict:
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in message.tool_calls
        ],
    }


def _merge_sources(sources: list[dict], new: list[dict]) -> None:
    seen = {s["doc_id"] for s in sources}
    for source in new:
        if source["doc_id"] not in seen:
            sources.append(source)
            seen.add(source["doc_id"])
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_prompts.py tests/test_agent.py -v`
Expected: 13 passed

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: add system prompt and hand-rolled tool-calling agent loop"
```

---

### Task 7: HTTP API server

**Files:**
- Create: `src/onboarding_agent/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `Agent`, `Store`, `Toolbox.checklist_status`, `Employee`, the config/loader functions, `Retriever.load`, `IndexNotFoundError`
- Produces:
  - `server.create_app(agent, store, toolbox, web_dir: Path | None = None) -> FastAPI`
  - `server.build_default_app() -> FastAPI`
  - `server.main()`
  - HTTP endpoints:
    - `GET /api/employee` → `{"employee": {...} | null}`
    - `POST /api/employee` with `{name, role?, team, start_date}` → `{"employee": {...}}`
    - `GET /api/checklist` → `checklist_status()`
    - `GET /api/history` → `{"messages": [...]}`
    - `POST /api/chat` with `{message}` → `{"reply", "sources"}`. Returns 409 when there is no profile, 422 for a blank message and 502 on an OpenAI failure.
    - Static `web/` is mounted at `/`.

- [ ] **Step 1: Write the failing tests** in `tests/test_server.py`

```python
import pytest
from fastapi.testclient import TestClient
from openai import OpenAIError

from fakes import TODAY, FakeOpenAI, completion, tool_call
from onboarding_agent.agent import Agent
from onboarding_agent.server import create_app

PROFILE = {"name": "Asha", "team": "Platform", "start_date": "2026-10-05"}


@pytest.fixture
def make_http(toolbox, store):
    def make(responses=(), web_dir=None):
        agent = Agent(FakeOpenAI(list(responses)), "test-model", toolbox, store, today=lambda: TODAY)
        return TestClient(create_app(agent, store, toolbox, web_dir=web_dir))

    return make


def test_profile_flow(make_http):
    http = make_http()
    assert http.get("/api/employee").json() == {"employee": None}
    saved = http.post("/api/employee", json=PROFILE)
    assert saved.status_code == 200
    assert saved.json()["employee"] == {**PROFILE, "role": "Software Engineer"}
    assert http.get("/api/employee").json()["employee"]["name"] == "Asha"


def test_profile_validation(make_http):
    http = make_http()
    assert http.post("/api/employee", json={**PROFILE, "name": "   "}).status_code == 422
    assert http.post("/api/employee", json={**PROFILE, "start_date": "next tuesday"}).status_code == 422


def test_chat_requires_profile(make_http):
    response = make_http().post("/api/chat", json={"message": "hi"})
    assert response.status_code == 409
    assert "profile" in response.json()["detail"]


def test_chat_returns_reply_and_sources(make_http):
    http = make_http([
        completion(tool_calls=[tool_call("c1", "search_docs", '{"query": "leave"}')]),
        completion("24 days."),
    ])
    http.post("/api/employee", json=PROFILE)
    body = http.post("/api/chat", json={"message": "How much leave?"}).json()
    assert body["reply"] == "24 days."
    assert body["sources"][0]["doc_id"] == "02-leave"
    assert http.get("/api/history").json()["messages"] == [
        {"role": "user", "content": "How much leave?"},
        {"role": "assistant", "content": "24 days."},
    ]


def test_blank_message_rejected(make_http):
    http = make_http()
    http.post("/api/employee", json=PROFILE)
    assert http.post("/api/chat", json={"message": "   "}).status_code == 422


def test_openai_failure_returns_502_and_saves_nothing(make_http):
    http = make_http([OpenAIError("down")])
    http.post("/api/employee", json=PROFILE)
    response = http.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 502
    assert "try again" in response.json()["detail"]
    assert http.get("/api/history").json()["messages"] == []


def test_checklist_endpoint(make_http):
    http = make_http()
    http.post("/api/employee", json=PROFILE)
    body = http.get("/api/checklist").json()
    assert (body["day"], body["completed"], body["total"]) == (3, 0, 4)


def test_serves_web_ui(make_http, tmp_path):
    (tmp_path / "index.html").write_text("<h1>Onboarding Buddy</h1>", encoding="utf-8")
    http = make_http(web_dir=tmp_path)
    assert "Onboarding Buddy" in http.get("/").text
    assert http.get("/api/employee").status_code == 200  # API routes still win over static files
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onboarding_agent.server'`

- [ ] **Step 3: Implement** `src/onboarding_agent/server.py`

```python
"""HTTP API for the web UI, plus the `serve` command that wires the real dependencies together."""
import os
from datetime import date
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from .agent import Agent
from .config import load_settings, require_api_key
from .embeddings import openai_embedder
from .retriever import IndexNotFoundError, Retriever
from .store import Employee, Store
from .tools import Toolbox, load_checklist, load_contacts


class EmployeeIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=100)
    role: str = Field(default="Software Engineer", min_length=1, max_length=100)
    team: str = Field(min_length=1, max_length=100)
    start_date: date


class ChatIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=4000)


def _employee_json(employee: Employee) -> dict:
    return {
        "name": employee.name,
        "role": employee.role,
        "team": employee.team,
        "start_date": employee.start_date.isoformat(),
    }


def create_app(agent: Agent, store: Store, toolbox: Toolbox, web_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Onboarding Buddy")

    @app.get("/api/employee")
    def get_employee():
        employee = store.get_employee()
        return {"employee": _employee_json(employee) if employee else None}

    @app.post("/api/employee")
    def save_employee(body: EmployeeIn):
        employee = Employee(**body.model_dump())
        store.save_employee(employee)
        return {"employee": _employee_json(employee)}

    @app.get("/api/checklist")
    def checklist():
        return toolbox.checklist_status()

    @app.get("/api/history")
    def history():
        return {"messages": store.recent_messages(limit=50)}

    @app.post("/api/chat")
    def chat(body: ChatIn):
        if store.get_employee() is None:
            raise HTTPException(409, "Please set up your profile first.")
        try:
            reply = agent.chat(body.message)
        except OpenAIError as exc:
            raise HTTPException(502, f"The AI service failed ({type(exc).__name__}). Please try again.") from exc
        return {"reply": reply.text, "sources": reply.sources}

    if web_dir is not None:
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


def build_default_app() -> FastAPI:
    settings = load_settings()
    client = OpenAI(api_key=require_api_key(settings))
    retriever = Retriever.load(settings.index_dir, openai_embedder(client, settings.embed_model))
    store = Store(settings.db_path)
    toolbox = Toolbox(
        retriever,
        store,
        load_checklist(settings.data_dir / "checklist.yaml"),
        load_contacts(settings.data_dir / "contacts.json"),
        settings.kb_dir,
    )
    agent = Agent(
        client, settings.model, toolbox, store,
        max_steps=settings.max_agent_steps, reasoning_effort=settings.reasoning_effort,
    )
    return create_app(agent, store, toolbox, web_dir=settings.web_dir)


def main() -> None:
    try:
        app = build_default_app()
    except IndexNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    port = int(os.getenv("PORT", "8000"))
    print(f"Onboarding Buddy running at http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add FastAPI server for profile, checklist and chat"
```

---

### Task 8: Web chat UI

**Files:**
- Create: `web/index.html`, `web/app.js`, `web/styles.css`

**Interfaces:**
- Consumes: the HTTP endpoints from Task 7. Team options must match the teams named in `knowledge_base/00-welcome.md` (Platform, Growth, Integrations).

There's no JS test harness (YAGNI for a learning project). Verify by running `node --check` and the manual flow in Task 10.

- [ ] **Step 1: Write** `web/index.html`

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Onboarding Buddy</title>
  <link rel="stylesheet" href="/styles.css">
  <script src="https://cdn.jsdelivr.net/npm/marked@15/marked.min.js" defer></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js" defer></script>
  <script src="/app.js" defer></script>
</head>
<body>
  <section id="setup" class="setup" hidden>
    <form id="setup-form" class="card">
      <h1>Welcome to Nimbus Labs 👋</h1>
      <p class="muted">I'm your Onboarding Buddy. Tell me a little about yourself so I can guide you.</p>
      <label>Your name <input name="name" required maxlength="100" autocomplete="name"></label>
      <label>Role <input name="role" value="Software Engineer" required maxlength="100"></label>
      <label>Team
        <select name="team" required>
          <option>Platform</option>
          <option>Growth</option>
          <option>Integrations</option>
        </select>
      </label>
      <label>Start date <input name="start_date" type="date" required></label>
      <button type="submit">Start onboarding</button>
      <p id="setup-error" class="error" hidden></p>
    </form>
  </section>

  <main id="app" class="app" hidden>
    <aside class="sidebar">
      <h2 id="greeting"></h2>
      <p id="day-label" class="muted"></p>
      <div class="progress"><div id="progress-bar"></div></div>
      <p id="progress-label" class="muted"></p>
      <div id="checklist"></div>
    </aside>
    <section class="chat">
      <div id="messages" class="messages" aria-live="polite"></div>
      <div id="suggestions" class="suggestions"></div>
      <form id="chat-form" class="composer">
        <textarea id="chat-input" rows="1" maxlength="4000" placeholder="Ask anything about onboarding…" aria-label="Message"></textarea>
        <button type="submit" id="send">Send</button>
      </form>
    </section>
  </main>
</body>
</html>
```

- [ ] **Step 2: Write** `web/app.js`

```js
const $ = (selector) => document.querySelector(selector);

const SUGGESTIONS = [
  "What should I do first?",
  "How do I set up my dev environment?",
  "How does code review work here?",
  "How many leave days do I get?",
];

let busy = false;

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (typeof body.detail === "string") throw new Error(body.detail);
    if (Array.isArray(body.detail)) throw new Error(body.detail.map((d) => d.msg).join("; "));
    throw new Error("Something went wrong. Please try again.");
  }
  return body;
}

function renderMarkdown(text) {
  if (window.marked && window.DOMPurify) return DOMPurify.sanitize(marked.parse(text));
  const div = document.createElement("div"); // CDN unavailable: show plain text safely
  div.textContent = text;
  return div.innerHTML.replace(/\n/g, "<br>");
}

function addMessage(kind, text, sources = []) {
  const bubble = document.createElement("div");
  bubble.className = `message ${kind}`;
  if (kind === "assistant") bubble.innerHTML = renderMarkdown(text);
  else bubble.textContent = text;
  if (sources.length) {
    const row = document.createElement("div");
    row.className = "sources";
    row.append("Sources:");
    for (const source of sources) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = source.doc_title;
      chip.title = source.section;
      row.append(chip);
    }
    bubble.append(row);
  }
  $("#messages").append(bubble);
  bubble.scrollIntoView({ block: "end", behavior: "smooth" });
  return bubble;
}

function showSuggestions() {
  $("#suggestions").replaceChildren(
    ...SUGGESTIONS.map((text) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = text;
      button.addEventListener("click", () => send(text));
      return button;
    }),
  );
}

async function refreshChecklist() {
  const status = await api("/api/checklist");
  $("#day-label").textContent =
    status.day >= 1 ? `Day ${status.day} of onboarding` : `You start in ${1 - status.day} day(s)`;
  const percent = status.total ? Math.round((100 * status.completed) / status.total) : 0;
  $("#progress-bar").style.width = `${percent}%`;
  $("#progress-label").textContent = `${status.completed} of ${status.total} tasks done`;

  const byWeek = new Map();
  for (const task of status.tasks) {
    if (!byWeek.has(task.week)) byWeek.set(task.week, []);
    byWeek.get(task.week).push(task);
  }
  const container = $("#checklist");
  container.replaceChildren();
  for (const [week, tasks] of byWeek) {
    const heading = document.createElement("h3");
    heading.textContent = `Week ${week}`;
    const list = document.createElement("ul");
    for (const task of tasks) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = task.done ? "task done" : "task";
      button.textContent = `${task.done ? "✓" : "○"} ${task.title}`;
      button.title = task.done ? "Done: click to ask about it again" : "Ask the buddy to help with this";
      button.addEventListener("click", () =>
        send(task.done ? `Remind me about: ${task.title}` : `Help me with: ${task.title}`),
      );
      item.append(button);
      list.append(item);
    }
    container.append(heading, list);
  }
}

async function send(text) {
  text = text.trim();
  if (!text || busy) return;
  busy = true;
  $("#send").disabled = true;
  $("#chat-input").value = "";
  $("#suggestions").replaceChildren();
  addMessage("user", text);
  const typing = addMessage("typing", "Thinking…");
  try {
    const { reply, sources } = await api("/api/chat", { method: "POST", body: JSON.stringify({ message: text }) });
    typing.remove();
    addMessage("assistant", reply, sources);
    await refreshChecklist(); // the agent may have marked a task done
  } catch (error) {
    typing.remove();
    addMessage("error", error.message);
  } finally {
    busy = false;
    $("#send").disabled = false;
    $("#chat-input").focus();
  }
}

async function showApp(employee) {
  $("#setup").hidden = true;
  $("#app").hidden = false;
  $("#greeting").textContent = `Hi ${employee.name} 👋`;
  await refreshChecklist();
  const { messages } = await api("/api/history");
  for (const message of messages) addMessage(message.role, message.content);
  if (!messages.length) {
    addMessage(
      "assistant",
      `Welcome to Nimbus Labs, **${employee.name}**! I'm your Onboarding Buddy. Ask me about your setup, ` +
        "our engineering workflow or company policies, or click a task on the left to get started.",
    );
    showSuggestions();
  }
}

$("#setup-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(event.target));
  try {
    const { employee } = await api("/api/employee", { method: "POST", body: JSON.stringify(data) });
    await showApp(employee);
  } catch (error) {
    $("#setup-error").textContent = error.message;
    $("#setup-error").hidden = false;
  }
});

$("#chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  send($("#chat-input").value);
});

$("#chat-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    send(event.target.value);
  }
});

(async function init() {
  const { employee } = await api("/api/employee");
  if (employee) {
    await showApp(employee);
  } else {
    $("#setup").hidden = false;
    $("input[name=start_date]").value = new Date().toLocaleDateString("en-CA"); // local YYYY-MM-DD
  }
})();
```

- [ ] **Step 3: Write** `web/styles.css`

```css
:root {
  --bg: #f6f7fb;
  --panel: #ffffff;
  --text: #1d2433;
  --muted: #6b7385;
  --border: #e3e6ef;
  --accent: #4f5bd5;
  --accent-text: #ffffff;
  --done: #2f9e6e;
  --error-bg: #fdecec;
  --error-text: #a12d2d;
  --radius: 12px;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color-scheme: light dark;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #12151c;
    --panel: #1b1f29;
    --text: #e6e9f0;
    --muted: #9aa3b5;
    --border: #2a3040;
    --accent: #7c86ff;
    --done: #4cc38a;
    --error-bg: #3a1d1f;
    --error-text: #f2a7a7;
  }
}

* { box-sizing: border-box; }
[hidden] { display: none !important; }
body { margin: 0; background: var(--bg); color: var(--text); }

input, select, textarea {
  font: inherit; color: var(--text); background: var(--bg);
  border: 1px solid var(--border); border-radius: 8px; padding: 10px;
}
button {
  font: inherit; cursor: pointer; border: 0; border-radius: 8px;
  padding: 10px 16px; background: var(--accent); color: var(--accent-text);
}
button:disabled { opacity: 0.6; cursor: wait; }
.muted { color: var(--muted); margin: 0 0 8px; font-size: 0.9rem; }
.error { color: var(--error-text); margin: 0; }

/* First-visit profile form */
.setup { display: grid; place-items: center; min-height: 100dvh; padding: 16px; }
.card {
  background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 28px; width: min(420px, 100%); display: grid; gap: 14px;
}
.card h1 { margin: 0; font-size: 1.4rem; }
.card label { display: grid; gap: 6px; font-size: 0.9rem; color: var(--muted); }

/* Main layout: checklist sidebar + chat */
.app { display: grid; grid-template-columns: 300px 1fr; height: 100dvh; }
.sidebar { background: var(--panel); border-right: 1px solid var(--border); padding: 20px; overflow-y: auto; }
.sidebar h2 { margin: 0 0 4px; font-size: 1.2rem; }
.sidebar h3 {
  font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); margin: 20px 0 6px;
}
.sidebar ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }
.progress { height: 8px; background: var(--border); border-radius: 99px; overflow: hidden; margin-bottom: 6px; }
#progress-bar { height: 100%; width: 0; background: var(--done); transition: width 0.4s; }
.task {
  width: 100%; text-align: left; background: none; color: var(--text);
  padding: 7px 8px; border-radius: 6px; font-size: 0.92rem;
}
.task:hover { background: var(--bg); }
.task.done { color: var(--muted); text-decoration: line-through; }

.chat { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 12px; }
.message {
  max-width: min(720px, 85%); padding: 12px 16px; border-radius: var(--radius);
  line-height: 1.5; overflow-wrap: anywhere;
}
.message.user { align-self: flex-end; background: var(--accent); color: var(--accent-text); }
.message.assistant { align-self: flex-start; background: var(--panel); border: 1px solid var(--border); }
.message.typing { align-self: flex-start; color: var(--muted); font-style: italic; }
.message.error { align-self: flex-start; background: var(--error-bg); color: var(--error-text); }
.message > :first-child { margin-top: 0; }
.message > :last-child { margin-bottom: 0; }
.message pre { background: var(--bg); padding: 10px; border-radius: 8px; overflow-x: auto; }
.message code { font-family: ui-monospace, Menlo, monospace; font-size: 0.88em; }
.sources { margin-top: 10px; font-size: 0.8rem; color: var(--muted); display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.chip { border: 1px solid var(--border); border-radius: 99px; padding: 2px 10px; }

.suggestions { display: flex; flex-wrap: wrap; gap: 8px; padding: 0 24px 12px; }
.suggestions button {
  background: var(--panel); color: var(--accent); border: 1px solid var(--border);
  font-size: 0.88rem; padding: 6px 12px; border-radius: 99px;
}
.composer { display: flex; gap: 10px; padding: 16px 24px; border-top: 1px solid var(--border); background: var(--panel); }
.composer textarea { flex: 1; resize: none; min-height: 44px; max-height: 160px; }

@media (max-width: 760px) {
  .app { grid-template-columns: 1fr; grid-template-rows: auto 1fr; }
  .sidebar { border-right: 0; border-bottom: 1px solid var(--border); max-height: 35dvh; padding: 16px; }
  .messages, .composer { padding: 16px; }
  .suggestions { padding: 0 16px 12px; }
}
```

- [ ] **Step 3b: Pin the CDN scripts with Subresource Integrity**

In `index.html`, replace the two CDN `<script>` tags with exact versions (look up the latest `marked` 15.x and `dompurify` 3.x on npm) and add SRI hashes so a compromised CDN can't inject code. Compute each hash:

```bash
curl -sL https://cdn.jsdelivr.net/npm/marked@<exact>/marked.min.js | openssl dgst -sha384 -binary | openssl base64 -A
curl -sL https://cdn.jsdelivr.net/npm/dompurify@<exact>/dist/purify.min.js | openssl dgst -sha384 -binary | openssl base64 -A
```

Then use them in the tags:

```html
<script src="https://cdn.jsdelivr.net/npm/marked@<exact>/marked.min.js" integrity="sha384-<hash>" crossorigin="anonymous" defer></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@<exact>/dist/purify.min.js" integrity="sha384-<hash>" crossorigin="anonymous" defer></script>
```

If a script fails to load, `renderMarkdown` falls back to escaped plain text, so the UI still works.

- [ ] **Step 4: Syntax-check the JS**

Run: `node --check web/app.js`
Expected: no output, exit code 0

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add web chat UI with checklist sidebar"
```

---

### Task 9: Sample handbook, checklist, contacts and golden questions

**Files:**
- Create: `knowledge_base/00-welcome.md` … `knowledge_base/09-faq.md` (10 files), `data/checklist.yaml`, `data/contacts.json`, `evals/golden.yaml`
- Test: `tests/test_knowledge_base.py`

**Interfaces:**
- Consumes: `chunk_markdown`, `load_checklist`, `load_contacts`, `PROJECT_ROOT`
- Produces: the real content used by `serve`, `ingest` and `eval`. Doc ids are exactly those listed below.

- [ ] **Step 1: Write the failing content tests** in `tests/test_knowledge_base.py`

```python
import re

import yaml

from onboarding_agent.config import PROJECT_ROOT
from onboarding_agent.ingest import chunk_markdown
from onboarding_agent.tools import load_checklist, load_contacts

KB = PROJECT_ROOT / "knowledge_base"
DATA = PROJECT_ROOT / "data"
EXPECTED_DOCS = {
    "00-welcome", "01-dev-environment-setup", "02-engineering-workflow", "03-coding-standards",
    "04-architecture-overview", "05-tools-and-access", "06-hr-policies", "07-first-90-days",
    "08-glossary", "09-faq",
}


def doc_ids():
    return {path.stem for path in KB.glob("*.md")}


def test_handbook_has_expected_docs():
    assert doc_ids() == EXPECTED_DOCS


def test_every_doc_has_a_title_and_several_sections():
    for path in KB.glob("*.md"):
        chunks = chunk_markdown(path.stem, path.read_text(encoding="utf-8"))
        assert chunks[0].doc_title != path.stem, f"{path.name} needs an H1 title"
        assert len({c.heading for c in chunks}) >= 3, f"{path.name} needs at least 3 sections"


def test_checklist_ids_unique_and_refs_exist():
    tasks = load_checklist(DATA / "checklist.yaml")
    assert len(tasks) == 15
    assert len({t.id for t in tasks}) == len(tasks)
    assert {t.doc_ref for t in tasks} <= doc_ids()
    assert {t.week for t in tasks} == {1, 2, 3, 4}


def test_contacts_are_complete():
    contacts = load_contacts(DATA / "contacts.json")
    assert len(contacts) == 8
    assert all(c.name and c.role and c.channel and c.topics for c in contacts)


def test_golden_questions_point_at_real_docs():
    cases = yaml.safe_load((PROJECT_ROOT / "evals" / "golden.yaml").read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 12
    assert {case["expect_doc"] for case in cases} <= doc_ids()


def test_fictional_emails_only():
    email = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")  # Slack handles like "@arjun.mehta" don't match
    for path in [*KB.glob("*.md"), DATA / "contacts.json"]:
        for address in email.findall(path.read_text(encoding="utf-8")):
            assert address.endswith("@nimbuslabs.example"), f"{path.name}: {address}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_knowledge_base.py -v`
Expected: FAIL (`assert set() == {...}`, because the knowledge base doesn't exist yet)

- [ ] **Step 3: Write `data/checklist.yaml`**

```yaml
tasks:
  - {id: accounts, week: 1, title: "Sign in to Okta, Google Workspace and Slack", doc_ref: 05-tools-and-access}
  - {id: meet-buddy, week: 1, title: "Meet your onboarding buddy and manager", doc_ref: 00-welcome}
  - {id: peoplehub-profile, week: 1, title: "Complete your PeopleHub profile and payroll details", doc_ref: 06-hr-policies}
  - {id: github-access, week: 1, title: "Get access to the nimbus GitHub organization", doc_ref: 05-tools-and-access}
  - {id: vpn, week: 1, title: "Install and connect to the Tailscale VPN", doc_ref: 05-tools-and-access}
  - {id: dev-environment, week: 1, title: "Set up your development environment", doc_ref: 01-dev-environment-setup}
  - {id: run-locally, week: 1, title: "Run flow-api and flow-web locally", doc_ref: 01-dev-environment-setup}
  - {id: read-workflow, week: 1, title: "Read the engineering workflow (branches, PRs, reviews)", doc_ref: 02-engineering-workflow}
  - {id: read-architecture, week: 2, title: "Read the architecture overview", doc_ref: 04-architecture-overview}
  - {id: coding-standards, week: 2, title: "Read the coding standards and install pre-commit hooks", doc_ref: 03-coding-standards}
  - {id: first-pr, week: 2, title: "Ship your first PR (a good-first-issue ticket)", doc_ref: 02-engineering-workflow}
  - {id: first-ticket, week: 3, title: "Pick up your first sprint ticket", doc_ref: 07-first-90-days}
  - {id: review-prs, week: 3, title: "Review two teammates' pull requests", doc_ref: 02-engineering-workflow}
  - {id: watch-release, week: 4, title: "Watch a release-train deploy to production", doc_ref: 02-engineering-workflow}
  - {id: day30-checkin, week: 4, title: "Prepare for your 30-day check-in", doc_ref: 07-first-90-days}
```

- [ ] **Step 4: Write `data/contacts.json`**

```json
[
  {"name": "Arjun Mehta", "role": "Onboarding Buddy (Senior Software Engineer, Platform)", "channel": "Slack @arjun.mehta",
   "topics": ["onboarding", "buddy", "codebase", "code", "questions", "help", "pr", "review", "setup", "stuck"]},
  {"name": "Priya Sharma", "role": "Engineering Manager", "channel": "Slack @priya.sharma (weekly 1:1 on Mondays)",
   "topics": ["manager", "goals", "checkin", "check", "performance", "feedback", "probation", "career", "team"]},
  {"name": "IT Helpdesk", "role": "IT support", "channel": "#it-help on Slack or it@nimbuslabs.example",
   "topics": ["laptop", "vpn", "tailscale", "okta", "password", "access", "github", "account", "accounts", "hardware", "sso", "1password", "mfa"]},
  {"name": "Neha Kapoor", "role": "HR Business Partner", "channel": "hr@nimbuslabs.example or PeopleHub",
   "topics": ["leave", "holidays", "holiday", "payroll", "salary", "pay", "benefits", "insurance", "policy", "documents", "probation"]},
  {"name": "Platform Infra team", "role": "DevOps and CI/CD owners", "channel": "#devops on Slack",
   "topics": ["deploy", "deployment", "ci", "pipeline", "aws", "staging", "production", "docker", "release", "infrastructure", "build"]},
  {"name": "Security team", "role": "Security and compliance", "channel": "#security on Slack or security@nimbuslabs.example",
   "topics": ["security", "phishing", "secrets", "secret", "credentials", "leak", "incident", "data", "privacy"]},
  {"name": "On-call engineer", "role": "Current production on-call (rotates weekly)", "channel": "#incidents on Slack, page via PagerDuty",
   "topics": ["outage", "incident", "alert", "down", "oncall", "pagerduty", "broken", "production"]},
  {"name": "Finance team", "role": "Expenses and reimbursements", "channel": "finance@nimbuslabs.example",
   "topics": ["expenses", "expense", "reimbursement", "travel", "invoice", "budget", "equipment"]}
]
```

- [ ] **Step 5: Write `evals/golden.yaml`**

```yaml
cases:
  - {question: "How do I set up my laptop for development?", expect_doc: 01-dev-environment-setup}
  - {question: "Which command starts the backend locally?", expect_doc: 01-dev-environment-setup}
  - {question: "What should I name my git branch?", expect_doc: 02-engineering-workflow}
  - {question: "How many approvals does a pull request need?", expect_doc: 02-engineering-workflow}
  - {question: "When do production deploys happen?", expect_doc: 02-engineering-workflow}
  - {question: "What linter and formatter do we use for Python?", expect_doc: 03-coding-standards}
  - {question: "What database does Nimbus Flow use?", expect_doc: 04-architecture-overview}
  - {question: "How do I get access to GitHub?", expect_doc: 05-tools-and-access}
  - {question: "How do I connect to the VPN?", expect_doc: 05-tools-and-access}
  - {question: "How many paid leave days do I get per year?", expect_doc: 06-hr-policies}
  - {question: "When is salary paid each month?", expect_doc: 06-hr-policies}
  - {question: "What is expected of me in my first 30 days?", expect_doc: 07-first-90-days}
  - {question: "What does release train mean?", expect_doc: 08-glossary}
  - {question: "Who is my onboarding buddy?", expect_doc: 00-welcome}
```

- [ ] **Step 6: Write the 10 handbook docs**

Each doc is Markdown with one `# Title` (H1), at least 3 `##` sections, and 60–150 lines of friendly, concrete prose written for a fresher. Each fact lives in exactly one doc. Other docs link to it in prose ("see *Tools & Access*") and don't restate it. The FAQ answers in 1–2 sentences and names the doc to read. Use **exactly** the facts below so the checklist, contacts and eval stay consistent. Commands go in fenced code blocks with `#` comments; the chunker handles these.

| doc_id / H1 title | Required `##` sections and facts |
|---|---|
| `00-welcome` / **Welcome to Nimbus Labs** | **About us:** Nimbus Labs, a ~120-person startup building **Nimbus Flow**, workflow-automation SaaS for small businesses. **Mission & values:** "Customers first, ship small, own it, be kind". **Teams:** Platform (core API & infra), Growth (signup, billing, web app), Integrations (third-party connectors). **Who's who:** Engineering Manager Priya Sharma; your onboarding buddy is **Arjun Mehta** (Senior SWE, Platform); HR partner Neha Kapoor. **Your first day:** 10:00 welcome session, laptop pickup from IT, lunch with your team, and a buddy intro at 15:00. |
| `01-dev-environment-setup` / **Development Environment Setup** | **Prerequisites:** macOS laptop from IT; install Homebrew, Git, Python 3.12 via `uv`, Node 22 via `fnm`, Docker Desktop. **Clone the repos:** `nimbus/flow-api`, `nimbus/flow-web`, `nimbus/infra` (needs GitHub access and the VPN; see Tools & Access). **Secrets:** copy `.env.example` to `.env.local` and fill values from the 1Password vault "Engineering – Dev". **Run flow-api:** `make bootstrap` then `make dev` (API at http://localhost:8080, Postgres + Redis via `docker compose up -d`). **Run flow-web:** `npm ci` then `npm run dev` (http://localhost:5173). **Troubleshooting:** port already in use, Docker not running, migrations (`make migrate`). |
| `02-engineering-workflow` / **Engineering Workflow** | **Branching:** trunk-based on `main`; branch name `feat/FLOW-123-short-description` (also `fix/`, `chore/`). **Commits:** Conventional Commits (`feat:`, `fix:`, …). **Pull requests:** keep them small (<400 lines); fill in the PR template; **1 approval** needed (**2** for the `infra` repo); CI must be green; **squash merge**. **Code review:** respond within 1 working day; be kind and specific; how to review as a newcomer. **CI/CD & releases:** GitHub Actions runs tests on every PR; merges auto-deploy to **staging**; production ships via the **release train on Tuesdays and Thursdays at 14:00 IST**; no production deploys on Friday after 14:00. **Good first issues:** Jira label `good-first-issue`. |
| `03-coding-standards` / **Coding Standards** | **Python:** `ruff` for linting and formatting, `mypy --strict` for new modules, type hints required, pytest, ≥80% coverage on new code. **TypeScript/React:** ESLint + Prettier, function components + hooks, no `any`. **Naming & structure:** snake_case for Python, camelCase for TS; small functions. **Tests:** every bug fix needs a regression test. **Pre-commit hooks:** `pre-commit install` once per repo. **Logging & errors:** structured logging via `structlog`; never log secrets or personal data. |
| `04-architecture-overview` / **Architecture Overview** | **System at a glance:** `flow-web` (React + TypeScript SPA) → `flow-api` (Python 3.12, FastAPI) → **PostgreSQL 16** (primary DB) + **Redis** (cache and job queue); `flow-runner` workers execute customer workflows; `connectors` service (owned by Integrations). **Hosting:** AWS: ECS Fargate, RDS for Postgres, S3 for file storage, CloudFront. **Request flow:** a step-by-step path of one API request. **Multi-tenancy:** every table has `tenant_id`; always filter by it. **Observability:** Datadog for metrics/logs, Sentry for errors, PagerDuty for alerts. |
| `05-tools-and-access` / **Tools & Access** | **Day-one accounts:** Okta SSO (set up MFA with Okta Verify), Google Workspace, Slack. **GitHub:** request access to the `nimbus` org by posting your GitHub username in **#it-help**; approved within 1 working day. **VPN:** install **Tailscale**, sign in with Okta; required for internal tools and cloning private repos. **Passwords:** 1Password; never share secrets in Slack. **Other tools:** Jira (tickets), Confluence (design docs), Datadog, Sentry. **AWS access:** read-only console via Okta after week 1; production write access only after on-call training. **Laptop:** issued by IT; report problems in #it-help. |
| `06-hr-policies` / **HR Policies** | **Working hours:** core hours **11:00–16:00 IST**; hybrid: in office **Tuesday–Thursday**. **Leave:** **24 days paid leave per year**, 12 days sick leave, plus public holidays from the yearly list; apply in **PeopleHub** at least 1 week ahead for planned leave. **Payroll:** salary paid on the **last working day of each month**; add bank details in PeopleHub during week 1. **Probation:** 6 months, with check-ins at 30/60/90 days. **Expenses:** submit within 30 days with receipts (finance@nimbuslabs.example). **Code of conduct & harassment reporting:** contact HR (Neha Kapoor). |
| `07-first-90-days` / **Your First 90 Days** | **Week 1:** accounts, dev setup, run locally, meet buddy. **Weeks 2–4 (first 30 days):** ship your first PR from a `good-first-issue`, pick up your first sprint ticket, review teammates' PRs; 30-day check-in with your manager. **Days 31–60:** own a small feature end to end with a design note. **Days 61–90:** shadow the on-call rotation for one week; 90-day review. **Rituals:** daily standup at 10:30 IST (15 min), two-week sprints, sprint planning on Mondays, retro every other Friday. **How to ask for help:** try for 30 minutes, then ask in your team channel or your buddy. |
| `08-glossary` / **Glossary** | At least 3 sections (e.g. **Company terms**, **Engineering terms**, **Process terms**). Define: Flow, Runner, Tenant, Connector, Release train (the scheduled Tue/Thu production deploy), Staging, SLO, On-call, Standup, Sprint, PR, CI/CD, Squash merge, Feature flag. |
| `09-faq` / **FAQ** | At least 3 sections (e.g. **Setup & access**, **Working here**, **Engineering**), 12+ Q&As, each a short answer pointing to the right doc. Cover: GitHub access, VPN not connecting, where secrets live, when payday is, how to apply for leave, PR approvals needed, when deploys happen, who to ask when stuck, what to do if you break something (tell #incidents; blameless culture). |

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_knowledge_base.py -v`
Expected: 6 passed

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "docs: add sample Nimbus Labs handbook, checklist, contacts and eval set"
```

---

### Task 10: Retrieval eval, README and end-to-end check

**Files:**
- Create: `src/onboarding_agent/eval.py`, and replace `README.md`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `Retriever`, `load_settings`, `require_api_key`, `openai_embedder`, `PROJECT_ROOT`
- Produces: `eval.evaluate(retriever, cases: list[dict], k: int = 4) -> tuple[float, list[dict]]` and `eval.main()` (exits 1 when the hit rate is below 0.8)

- [ ] **Step 1: Write the failing test** in `tests/test_eval.py`

```python
from onboarding_agent.eval import evaluate


def test_evaluate_reports_hit_rate_and_rows(retriever):
    cases = [
        {"question": "How many days of paid leave?", "expect_doc": "02-leave"},
        {"question": "How many days of paid leave?", "expect_doc": "99-missing"},
    ]
    rate, rows = evaluate(retriever, cases, k=1)
    assert rate == 0.5
    assert [row["hit"] for row in rows] == [True, False]
    assert rows[0]["found"] == ["02-leave"]


def test_evaluate_empty_cases(retriever):
    assert evaluate(retriever, []) == (0.0, [])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onboarding_agent.eval'`

- [ ] **Step 3: Implement** `src/onboarding_agent/eval.py`

```python
"""Retrieval eval: for each golden question, does search surface the expected handbook doc?"""
import sys

import yaml
from openai import OpenAI

from .config import PROJECT_ROOT, load_settings, require_api_key
from .embeddings import openai_embedder
from .retriever import Retriever

PASS_THRESHOLD = 0.8


def evaluate(retriever: Retriever, cases: list[dict], k: int = 4) -> tuple[float, list[dict]]:
    rows = []
    for case in cases:
        found = [hit.chunk.doc_id for hit in retriever.search(case["question"], k=k)]
        rows.append({"question": case["question"], "expected": case["expect_doc"], "found": found,
                     "hit": case["expect_doc"] in found})
    rate = sum(row["hit"] for row in rows) / len(rows) if rows else 0.0
    return rate, rows


def main() -> None:
    settings = load_settings()
    client = OpenAI(api_key=require_api_key(settings))
    retriever = Retriever.load(settings.index_dir, openai_embedder(client, settings.embed_model))
    cases = yaml.safe_load((PROJECT_ROOT / "evals" / "golden.yaml").read_text(encoding="utf-8"))["cases"]
    rate, rows = evaluate(retriever, cases)
    for row in rows:
        print(f"[{'PASS' if row['hit'] else 'FAIL'}] {row['question']}")
        if not row["hit"]:
            print(f"        expected {row['expected']}, got {row['found']}")
    print(f"\nHit rate @4: {rate:.0%} ({sum(r['hit'] for r in rows)}/{len(rows)})")
    sys.exit(0 if rate >= PASS_THRESHOLD else 1)
```

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -v`
Expected: 69 passed, with no network access

- [ ] **Step 5: Write `README.md`**

Sections:
1. **What it is.** One paragraph, plus the architecture diagram from the spec.
2. **Quick start:**
   ```bash
   cp .env.example .env   # add OPENAI_API_KEY
   uv sync
   uv run ingest          # embed knowledge_base/ → data/index/
   uv run serve           # http://127.0.0.1:8000
   ```
3. **How the agent works.** Walk through `agent.py`'s loop in 5 numbered steps, plus a table of the 5 tools and what each returns.
4. **Use your own company docs.** Replace `knowledge_base/*.md` (H1 title + `##` sections), edit `data/checklist.yaml` and `data/contacts.json`, re-run `uv run ingest`, and delete `data/onboarding.db` to start over.
5. **Testing:** `uv run pytest` (offline, uses fakes) and `uv run eval` (real API, retrieval hit rate).
6. **Project layout.** Copy the File Structure tree from this plan.
7. **Ideas for next steps:** streaming, auth/multiple employees, Slack bot, PDF ingestion, a framework comparison.

- [ ] **Step 6: End-to-end check with the real API** (needs `OPENAI_API_KEY` in `.env`)

Run: `uv run ingest`. Expected output: `Indexed N chunks from 10 docs → …/data/index`.
Run: `uv run eval`. Expected: hit rate ≥ 80%, exit 0. If it's below 80%, improve the failing docs' wording (don't change the questions), re-ingest and re-run.
Run: `uv run serve`, then in the browser at http://127.0.0.1:8000:
1. The profile form appears. Submit name "Test", team Platform, today's date. The sidebar shows "Day 1 of onboarding" and "0 of 15 tasks done".
2. Ask "How do I connect to the VPN?". The answer mentions Tailscale and Okta, cites Tools & Access, and shows a source chip.
3. Say "I've connected to the VPN". The agent confirms, and the sidebar shows "1 of 15" with VPN struck through.
4. Ask "What's the office wifi password?". The agent says the handbook doesn't cover it and points to IT Helpdesk (#it-help).
5. Reload the page. The chat history and progress are still there.
6. Stop the server, run `OPENAI_API_KEY=bad uv run serve`, and send a message. The UI shows an inline error bubble with "Please try again".

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: add retrieval eval and README"
```
