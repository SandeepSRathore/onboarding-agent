"""HTTP API for the web UI, plus the `serve` command that wires the real dependencies together."""
import os
from datetime import date
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from openai import AuthenticationError, OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from .agent import Agent
from .config import load_settings, openai_client
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
        except AuthenticationError as exc:
            raise HTTPException(502, "OpenAI rejected the API key. Check OPENAI_API_KEY in .env and restart the server.") from exc
        except OpenAIError as exc:
            raise HTTPException(502, f"The AI service failed ({type(exc).__name__}). Please try again.") from exc
        return {"reply": reply.text, "sources": reply.sources}

    if web_dir is not None:
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


def build_default_app() -> FastAPI:
    settings = load_settings()
    client = openai_client(settings)
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
