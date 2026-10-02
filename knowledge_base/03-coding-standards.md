# Coding Standards

Consistent code is easier to read, review and change. Most of these rules are enforced automatically by tools, so you rarely have to think about them. Install the pre-commit hooks once and let the tools do the work.

## Python (flow-api)

- **Linting and formatting:** we use **`ruff`** for both. Don't argue about formatting; `ruff format` decides.
- **Type checking:** **`mypy --strict`** is required for all new modules. Older modules are being migrated gradually.
- **Type hints** are required on every function signature.
- **Tests:** we use **`pytest`**. New code needs **at least 80% test coverage**. CI reports coverage on every PR.
- **Python version:** 3.12. Feel free to use modern syntax such as `match` statements and `X | None` types.

```bash
# run the same checks CI runs, before you push
ruff check . && ruff format --check . && mypy src && pytest
```

## TypeScript and React (flow-web)

- **Linting and formatting:** **ESLint + Prettier**, configured in the repo.
- **Components:** write **function components with hooks**. We don't write new class components.
- **No `any`.** If you don't know a type, use `unknown` and narrow it. Ask for help with tricky generics.
- **Tests:** Vitest and React Testing Library. Test what the user sees and does, not implementation details.

## Naming and structure

- **Python:** `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- **TypeScript:** `camelCase` for functions and variables, `PascalCase` for components and types.
- **Small functions.** If a function needs a comment explaining each section, split it into smaller functions with good names.
- **Name things after what they mean,** not how they work: `active_tenants`, not `tenant_list_2`.

## Tests

- **Every bug fix needs a regression test.** First write a test that fails because of the bug, then fix it. That proves the fix works and stops the bug coming back.
- Keep tests fast and independent: no real network calls, and no shared state between tests.
- Test names describe behavior: `test_retry_stops_after_three_attempts`.

## Pre-commit hooks

Pre-commit hooks run the formatters and linters automatically every time you commit, so CI rarely fails on style. Install them **once per repository**:

```bash
# run inside each repo (flow-api, flow-web) after cloning
pre-commit install
```

If a hook changes a file, just `git add` it again and re-run your commit.

## Logging and errors

- Use **structured logging via `structlog`** in Python: log events with key/value fields, e.g. `log.info("flow_run_started", flow_id=flow.id, tenant_id=tenant.id)`.
- **Never log secrets or personal data.** That includes passwords, API tokens, email addresses and customer data. When in doubt, leave it out and ask the Security team.
- Raise specific exceptions with helpful messages. Don't silently catch and ignore errors; Sentry can only alert us about errors it sees.
