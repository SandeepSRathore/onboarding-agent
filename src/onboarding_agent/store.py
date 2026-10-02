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
