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
