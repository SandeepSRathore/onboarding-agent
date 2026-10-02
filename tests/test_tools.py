import json

from onboarding_agent.tools import TOOL_SCHEMAS, load_checklist, load_contacts


def run(toolbox, name, **arguments):
    result = toolbox.run(name, json.dumps(arguments))
    return json.loads(result.content), result


def test_schemas_match_handlers(toolbox):
    assert [s["function"]["name"] for s in TOOL_SCHEMAS] == toolbox.tool_names
    assert toolbox.tool_names == ["search_docs", "read_doc", "get_checklist", "mark_task_done", "find_contact"]


def test_search_docs_returns_sections_and_sources(toolbox):
    data, result = run(toolbox, "search_docs", query="paid leave days")
    assert data["results"][0]["doc_id"] == "02-leave"
    assert "24 days" in data["results"][0]["text"]
    assert result.sources[0] == {"doc_id": "02-leave", "doc_title": "Leave Policy", "section": "Paid leave"}


def test_search_docs_blank_query(toolbox):
    data, result = run(toolbox, "search_docs", query="  ")
    assert data["results"] == [] and result.sources == []


def test_read_doc_returns_full_markdown(toolbox):
    data, result = run(toolbox, "read_doc", doc_id="01-setup")
    assert "make bootstrap" in data["content"] and data["truncated"] is False
    assert result.sources == [{"doc_id": "01-setup", "doc_title": "Dev Setup", "section": "Dev Setup"}]


def test_read_doc_rejects_unknown_and_path_like_ids(toolbox):
    for bad in ["nope", "../.env", "01-setup.md"]:
        data, result = run(toolbox, "read_doc", doc_id=bad)
        assert "Valid ids: 01-setup, 02-leave, 03-workflow" in data["error"]
        assert result.sources == []


def test_get_checklist_reports_day_progress_and_next_tasks(toolbox, store, employee):
    store.save_employee(employee)
    store.mark_done("setup-laptop")
    data, _ = run(toolbox, "get_checklist")
    assert data["day"] == 3
    assert (data["completed"], data["total"]) == (1, 4)
    assert [t["id"] for t in data["next_tasks"]] == ["vpn", "first-pr", "leave-policy"]
    assert data["tasks"][0] == {"id": "setup-laptop", "week": 1, "title": "Set up your laptop", "doc_ref": "01-setup", "done": True}


def test_get_checklist_without_profile_has_no_day(toolbox):
    data, _ = run(toolbox, "get_checklist")
    assert data["day"] is None


def test_mark_task_done_updates_progress(toolbox, store):
    data, _ = run(toolbox, "mark_task_done", task_id="vpn")
    assert data["ok"] is True and data["task"] == "Connect to VPN"
    assert (data["completed"], data["total"]) == (1, 4)
    assert store.completed_task_ids() == {"vpn"}


def test_mark_task_done_unknown_id_lists_valid_ids(toolbox, store):
    data, _ = run(toolbox, "mark_task_done", task_id="fly-to-moon")
    assert "setup-laptop" in data["error"]
    assert store.completed_task_ids() == set()


def test_find_contact_matches_topic_words(toolbox):
    data, _ = run(toolbox, "find_contact", topic="My VPN is not working")
    assert data["matches"][0]["name"] == "IT Helpdesk"
    assert data["matches"][0]["channel"] == "#it-help on Slack"


def test_find_contact_without_match_returns_everyone(toolbox):
    data, _ = run(toolbox, "find_contact", topic="quantum physics")
    assert data["matches"] == []
    assert {c["name"] for c in data["all_contacts"]} == {"IT Helpdesk", "Neha Kapoor"}


def test_bad_tool_calls_return_errors_instead_of_raising(toolbox):
    cases = [
        ("delete_everything", "{}", "Unknown tool"),
        ("search_docs", "{not json", "not valid JSON"),
        ("search_docs", "[1, 2]", "JSON object"),
        ("search_docs", "{}", "Bad arguments"),
        ("search_docs", '{"query": "x", "extra": 1}', "Bad arguments"),
    ]
    for name, arguments, message in cases:
        data = json.loads(toolbox.run(name, arguments).content)
        assert message in data["error"], (name, arguments)


def test_empty_arguments_string_means_no_arguments(toolbox):
    assert "tasks" in json.loads(toolbox.run("get_checklist", "").content)


def test_loaders(tmp_path):
    (tmp_path / "c.yaml").write_text(
        "tasks:\n  - id: vpn\n    week: 1\n    title: Connect to VPN\n    doc_ref: 05-tools-and-access\n", encoding="utf-8"
    )
    (tmp_path / "c.json").write_text(
        '[{"name": "IT", "role": "Support", "channel": "#it-help", "topics": ["vpn"]}]', encoding="utf-8"
    )
    [task] = load_checklist(tmp_path / "c.yaml")
    assert (task.id, task.week, task.doc_ref) == ("vpn", 1, "05-tools-and-access")
    [contact] = load_contacts(tmp_path / "c.json")
    assert contact.topics == ("vpn",)
