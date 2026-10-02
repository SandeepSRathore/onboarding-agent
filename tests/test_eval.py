from onboarding_agent.eval import evaluate


def test_evaluate_reports_hit_rate_and_rows(retriever):
    cases = [
        {"question": "How many days of paid leave?", "expect_doc": "02-leave"},
        {"question": "How many days of paid leave?", "expect_doc": "99-missing"},
    ]
    rate, rows = evaluate(retriever, cases, k=1)
    assert rate == 0.5
    assert [row["hit"] for row in rows] == [True, False]
    assert rows[0]["found"] == ["02-leave"]


def test_evaluate_empty_cases(retriever):
    assert evaluate(retriever, []) == (0.0, [])
