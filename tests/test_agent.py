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
