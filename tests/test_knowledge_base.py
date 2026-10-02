import re

import yaml

from onboarding_agent.config import PROJECT_ROOT
from onboarding_agent.ingest import chunk_markdown
from onboarding_agent.tools import load_checklist, load_contacts

KB = PROJECT_ROOT / "knowledge_base"
DATA = PROJECT_ROOT / "data"
EXPECTED_DOCS = {
    "00-welcome", "01-dev-environment-setup", "02-engineering-workflow", "03-coding-standards",
    "04-architecture-overview", "05-tools-and-access", "06-hr-policies", "07-first-90-days",
    "08-glossary", "09-faq",
}


def doc_ids():
    return {path.stem for path in KB.glob("*.md")}


def test_handbook_has_expected_docs():
    assert doc_ids() == EXPECTED_DOCS


def test_every_doc_has_a_title_and_several_sections():
    for path in KB.glob("*.md"):
        chunks = chunk_markdown(path.stem, path.read_text(encoding="utf-8"))
        assert chunks[0].doc_title != path.stem, f"{path.name} needs an H1 title"
        assert len({c.heading for c in chunks}) >= 3, f"{path.name} needs at least 3 sections"


def test_checklist_ids_unique_and_refs_exist():
    tasks = load_checklist(DATA / "checklist.yaml")
    assert len(tasks) == 15
    assert len({t.id for t in tasks}) == len(tasks)
    assert {t.doc_ref for t in tasks} <= doc_ids()
    assert {t.week for t in tasks} == {1, 2, 3, 4}


def test_contacts_are_complete():
    contacts = load_contacts(DATA / "contacts.json")
    assert len(contacts) == 8
    assert all(c.name and c.role and c.channel and c.topics for c in contacts)


def test_golden_questions_point_at_real_docs():
    cases = yaml.safe_load((PROJECT_ROOT / "evals" / "golden.yaml").read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 12
    assert {case["expect_doc"] for case in cases} <= doc_ids()


def test_fictional_emails_only():
    email = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")  # Slack handles like "@arjun.mehta" don't match
    for path in [*KB.glob("*.md"), DATA / "contacts.json"]:
        for address in email.findall(path.read_text(encoding="utf-8")):
            if address.startswith("git@"):  # SSH clone URLs like git@github.com:nimbus/flow-api.git
                continue
            assert address.endswith("@nimbuslabs.example"), f"{path.name}: {address}"
