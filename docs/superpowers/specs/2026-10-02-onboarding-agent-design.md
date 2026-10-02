# Onboarding Buddy — Design Spec

Date: 2026-10-02

## 1. Understanding

**What you asked for**
- An AI agent that helps a **fresher software engineer** get through onboarding at a company.
- It guides them using the company's **existing docs**.

**Decisions you made**
- Goal: **learning / portfolio** project. Clarity matters more than scale.
- Docs: **generate a realistic sample handbook** for a fictional company; the real docs can be swapped in later.
- Interface: **web chat UI** with an onboarding-checklist sidebar.
- Stack: **Python + OpenAI**.
- Architecture: **a hand-rolled tool-calling agent** (no agent framework), so every moving part is visible.

**Assumptions (correct me if any are wrong)**
- Single-machine app. No login/auth; the new hire enters a name, team and start date on first visit.
- The fictional company is "Nimbus Labs", a small SaaS startup (Python/FastAPI backend, React frontend, Postgres, AWS).
- Answers must be grounded in the docs with citations. If the docs don't cover a question, the agent says so and suggests who to ask. It does not invent policy.
- You provide an `OPENAI_API_KEY`. The model is set in `.env`: `OPENAI_MODEL=gpt-5-mini` with `OPENAI_REASONING_EFFORT=low`, the cheapest model that handles multi-tool calling reliably. `gpt-5-nano` is a one-line swap to cut cost further.

**Success criteria**
1. A new hire can ask things like "How do I set up my dev environment?" or "What's the PR review process?" and get a correct, cited answer.
2. The agent knows where the new hire is in onboarding (day N, checklist progress) and proactively suggests the next step.
3. The new hire can say "I've finished setting up VPN" and the checklist updates.
4. A small eval script shows the agent retrieves the right doc for a set of golden questions.

## 2. Architecture

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
                       └─ find_contact(topic|name)    → contacts.json
```

**The agent loop** (`agent.py`, about 100 lines):
1. Build messages: a system prompt (role, rules, employee profile, "today is day N"), the recent chat history, and the new user message.
2. Call OpenAI with the tool schemas.
3. If the model returns tool calls, execute each one, append the results, and go back to step 2. Stop after at most 6 iterations.
4. Return the final text plus the list of sources that were used (for citation chips in the UI).

## 3. Components

| Unit | Purpose | Depends on |
|---|---|---|
| `knowledge_base/*.md` | ~10 sample handbook docs: welcome, dev-env setup, engineering workflow (git/PR/review), coding standards, architecture overview, tools & access, HR policies (leave, hours, payroll), first-90-days plan, glossary, FAQ | — |
| `data/checklist.yaml` | Onboarding tasks: `id`, `week`, `title`, `doc_ref` | — |
| `data/contacts.json` | People/teams and what to contact them for (IT, HR, buddy, manager, on-call) | — |
| `config.py` | Reads the env vars (`OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_REASONING_EFFORT`, `OPENAI_EMBED_MODEL`) and paths | — |
| `ingest.py` | Splits Markdown into chunks by heading (with a size cap), embeds them, and saves `data/index/` (vectors `.npy` + `chunks.json`). Run with `uv run ingest`. | OpenAI embeddings |
| `retriever.py` | Loads the index, embeds the query, and returns the top-k chunks with `doc_id`, `heading` and score | index, OpenAI |
| `store.py` | SQLite: the `employee` profile, `task_progress`, and `messages` (chat history) | sqlite3 (stdlib) |
| `tools.py` | Tool JSON schemas + implementations + a dispatcher | retriever, store, data files |
| `prompts.py` | System prompt template | — |
| `agent.py` | The loop described above | OpenAI client (injected, so tests can mock it), tools |
| `server.py` | `POST /api/employee`, `GET /api/checklist`, `POST /api/chat`, and serves the static `web/` files | agent, store |
| `web/` | `index.html` + `app.js` + `styles.css`: a profile form on first visit, then chat with citation chips and a checklist sidebar with progress (week 1–4) | server |

## 4. Agent behaviour (system prompt rules)
- You are "Onboarding Buddy" for Nimbus Labs. Be friendly and concise, and remember the user is a fresher, so explain jargon.
- Always use `search_docs` before answering a company-specific question. Cite the sources as `[doc title › section]`.
- If the docs don't answer the question, say so and call `find_contact` to suggest who to ask. Never invent policies, URLs or credentials.
- When the user says they finished something, call `mark_task_done`. When they ask "what next?", call `get_checklist` and suggest the next one or two tasks.

## 5. Error handling
- Missing API key or missing index → a clear startup error message ("run `uv run ingest` first").
- OpenAI error or timeout → the API returns 502 with a friendly message, and the UI shows it inline.
- An unknown tool or bad tool arguments → an error string is returned to the model as the tool result, so it can recover; the server doesn't crash.
- Loop cap reached → return the best partial answer with a note.

## 6. Testing
- `pytest` unit tests: chunking, retrieval ranking (with fake embeddings), store/progress, tool dispatch, and the agent loop with a **mocked OpenAI client** (no network).
- `uv run eval`: about 12 golden questions → checks that the expected `doc_id` appears among the retrieved sources (uses the real API).
- Manual check: run the server, do a first-visit flow, ask 3 questions, mark a task done, and confirm the sidebar updates.

## 7. Out of scope (YAGNI)
- Auth, multiple orgs, Slack integration, streaming responses, vector DBs, PDF ingestion. Each one is a possible follow-up.
