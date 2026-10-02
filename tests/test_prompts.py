from datetime import date

from fakes import TODAY
from onboarding_agent.prompts import build_system_prompt


def test_prompt_includes_profile_day_and_docs(employee):
    prompt = build_system_prompt(employee, TODAY, {"01-setup": "Dev Setup"})
    assert "Asha" in prompt and "Platform team" in prompt
    assert "day 3 of their onboarding" in prompt
    assert "- 01-setup: Dev Setup" in prompt
    assert "2026-10-07" in prompt


def test_prompt_before_start_date(employee):
    assert "they start in 2 day(s)" in build_system_prompt(employee, date(2026, 10, 3), {})


def test_prompt_on_start_date_is_day_one(employee):
    assert "day 1 of their onboarding" in build_system_prompt(employee, date(2026, 10, 5), {})


def test_prompt_without_profile(employee):
    assert "has not shared their profile" in build_system_prompt(None, TODAY, {})
