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
