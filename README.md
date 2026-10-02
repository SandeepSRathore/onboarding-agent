# Onboarding Buddy

An AI agent that guides a new (fresher) software engineer through onboarding at a company, using the company's own docs. It answers questions with citations to the handbook, knows which onboarding day you're on, suggests what to do next, and ticks off checklist tasks when you tell it you've finished them.

It's built **without an agent framework**: the tool-calling loop is about 100 lines of plain Python (`src/onboarding_agent/agent.py`), so every moving part is visible. The sample handbook belongs to a fictional company, *Nimbus Labs*.

```
Browser (chat + checklist sidebar)
        │  HTTP/JSON
        ▼
FastAPI server ──► Agent loop ──► OpenAI (function calling)
                       │
                       ├─ search_docs(query)          → Retriever (numpy cosine over embeddings)
                       ├─ read_doc(doc_id)            → knowledge_base/*.md
                       ├─ get_checklist()             → SQLite progress + checklist.yaml
                       ├─ mark_task_done(task_id)     → SQLite
                       └─ find_contact(topic)         → contacts.json
```

## Quick start

You need [uv](https://docs.astral.sh/uv/) and an OpenAI API key.

```bash
cp .env.example .env   # then paste your OPENAI_API_KEY into .env
uv sync
uv run ingest          # embed knowledge_base/ → data/index/
uv run serve           # open http://127.0.0.1:8000
```

The model is configured in `.env`. The default is `gpt-5-mini` with `OPENAI_REASONING_EFFORT=low`, which is cheap and reliable at tool calling. To cut cost further, try `OPENAI_MODEL=gpt-5-nano`. For a model without reasoning support, leave `OPENAI_REASONING_EFFORT` empty.

## How the agent works

Each chat message runs this loop (`Agent.chat`):

1. **Build the conversation:** a system prompt (rules, the new hire's profile, "today is day N", the list of handbook docs), the recent chat history, and the new message.
2. **Call the model** with the five tool definitions (`TOOL_SCHEMAS`).
3. **If the model asks for tools,** run each one (`Toolbox.run`), append the results as `tool` messages, and go back to step 2.
4. **If it answers with text,** that's the reply. The docs the tools touched become the source chips in the UI.
5. **Safety net:** after `MAX_AGENT_STEPS` rounds of tool use, make one final call with `tool_choice="none"` to force an answer.

| Tool | What it returns |
|---|---|
| `search_docs(query)` | The 4 most similar handbook sections (doc id, title, section, text, score) |
| `read_doc(doc_id)` | A whole handbook document (only ids from the index are allowed) |
| `get_checklist()` | Every task with its done status, progress counts, onboarding day, and the next 3 tasks |
| `mark_task_done(task_id)` | Confirmation plus updated progress |
| `find_contact(topic)` | The best-matching people or channels to ask, or everyone if nothing matches |

Bad tool calls (unknown tool, invalid JSON, wrong arguments) are returned to the model as `{"error": ...}` so it can recover, instead of crashing the request.

## Use your own company docs

1. Replace `knowledge_base/*.md` with your docs. Give each file one `# Title` and `##` sections; each section becomes a searchable chunk. The file name (without `.md`) is the doc id.
2. Edit `data/checklist.yaml` (tasks with `id`, `week`, `title`, `doc_ref`) and `data/contacts.json`.
3. Update the company name in `src/onboarding_agent/prompts.py` and the team list in `web/index.html`.
4. Re-run `uv run ingest`.
5. To start over as a new hire, delete `data/onboarding.db`.

## Testing

```bash
uv run pytest    # 69 offline tests: OpenAI is replaced by fakes in tests/fakes.py
uv run eval      # real API: does search find the right doc for each question in evals/golden.yaml?
```

`uv run eval` prints PASS/FAIL per question and exits non-zero if fewer than 80% of the expected docs are found in the top 4 results.

## Project layout

```
knowledge_base/*.md          sample handbook (10 docs)
data/checklist.yaml          onboarding tasks
data/contacts.json           who to ask about what
evals/golden.yaml            retrieval eval questions
src/onboarding_agent/
  config.py                  settings from .env
  embeddings.py              OpenAI embeddings + normalization
  ingest.py                  markdown → chunks → embedding index   (uv run ingest)
  retriever.py               cosine-similarity search over the index
  store.py                   SQLite: profile, progress, chat history
  tools.py                   tool schemas + implementations
  prompts.py                 system prompt
  agent.py                   the tool-calling loop
  server.py                  FastAPI API + static UI                (uv run serve)
  eval.py                    retrieval eval                         (uv run eval)
web/                         index.html, app.js, styles.css
tests/                       pytest suite + fakes
```

## Ideas for next steps

- Stream responses token by token
- Support multiple employees with login
- A Slack bot front end
- PDF and Confluence ingestion
- Rebuild the loop with the OpenAI Agents SDK and compare the two
