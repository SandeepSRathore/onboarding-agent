import httpx
import pytest
from fastapi.testclient import TestClient
from openai import AuthenticationError, OpenAIError

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


def test_bad_api_key_says_how_to_fix_it(make_http):
    response_401 = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    http = make_http([AuthenticationError("Incorrect API key provided", response=response_401, body=None)])
    http.post("/api/employee", json=PROFILE)
    response = http.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 502
    assert "OPENAI_API_KEY" in response.json()["detail"]
    assert "try again" not in response.json()["detail"]
