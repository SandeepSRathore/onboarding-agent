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
