import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from onboarding_agent.config import OPENAI_TIMEOUT_SECONDS, PROJECT_ROOT, load_settings, openai_client, require_api_key
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


def test_openai_client_fails_fast_instead_of_hanging():
    settings = dataclasses.replace(load_settings(), openai_api_key="sk-test")
    client = openai_client(settings)
    assert client.timeout == OPENAI_TIMEOUT_SECONDS == 60
    assert client.max_retries == 1
