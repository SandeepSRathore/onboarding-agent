from dataclasses import replace
from datetime import date

from onboarding_agent.store import Store


def test_no_employee_initially(store):
    assert store.get_employee() is None


def test_employee_roundtrip_and_update(store, employee):
    store.save_employee(employee)
    assert store.get_employee() == employee
    store.save_employee(replace(employee, team="Growth"))
    assert store.get_employee().team == "Growth"


def test_day_on_counts_start_date_as_day_one(employee):
    assert employee.day_on(date(2026, 10, 5)) == 1
    assert employee.day_on(date(2026, 10, 7)) == 3
    assert employee.day_on(date(2026, 10, 4)) == 0
    assert employee.day_on(date(2026, 10, 3)) == -1


def test_mark_done_is_idempotent(store):
    store.mark_done("vpn")
    store.mark_done("vpn")
    store.mark_done("first-pr")
    assert store.completed_task_ids() == {"vpn", "first-pr"}


def test_recent_messages_returns_last_n_oldest_first(store):
    for i in range(5):
        store.add_message("user", f"q{i}")
        store.add_message("assistant", f"a{i}")
    assert store.recent_messages(limit=3) == [
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "q4"},
        {"role": "assistant", "content": "a4"},
    ]


def test_data_persists_across_instances(tmp_path, employee):
    path = tmp_path / "nested" / "onboarding.db"
    first = Store(path)
    first.save_employee(employee)
    first.mark_done("vpn")
    second = Store(path)
    assert second.get_employee() == employee
    assert second.completed_task_ids() == {"vpn"}
