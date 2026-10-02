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
