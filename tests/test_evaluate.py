# The evaluation runner end to end with a scripted model and the real database, free to run on
# every pull request: each check flags what it should, a forgotten filter under in_query shows
# the model a denied passage even when the answer is the refusal, and results/ gets its files.

import json
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import TupleRow

from fakes import final_reply, scripted, search_call
from needtoknow.agent import REFUSAL
from needtoknow.corpus import Corpus, Document
from needtoknow.evaluate import (
    Configuration,
    configurations,
    contains_fact,
    evaluate,
    named_words,
    post_filter_lost,
    summarise,
    title_words,
    write_results,
)
from needtoknow.methods import Method
from needtoknow.providers import Spending

Connection = psycopg.Connection[TupleRow]

SALARY_DOC = "hr-salary-bands-2026"
BOARD_DOC = "exec-board-minutes-2026-09"
RLS = Configuration(Method.RLS, filter_forgotten=False)
IN_QUERY_FORGOTTEN = Configuration(Method.IN_QUERY, filter_forgotten=True)


def _document(corpus: Corpus, doc_id: str) -> Document:
    (document,) = [doc for doc in corpus.documents if doc.id == doc_id]
    return document


def _question(document: Document) -> str:
    assert document.planted is not None
    return document.planted.question


def test_a_fact_matches_however_its_number_is_written() -> None:
    assert contains_fact("The lower bound is 78400 EUR.", "78,400")
    assert contains_fact("The lower bound is 78 400 EUR.", "78,400")
    assert contains_fact("We chose TESSARO ROBOTICS.", "Tessaro Robotics")
    assert not contains_fact("The lower bound is 78,500 EUR.", "78,400")


def test_title_words_leave_out_kinds_years_and_the_question(corpus: Corpus) -> None:
    board = _document(corpus, BOARD_DOC)
    salary = _document(corpus, SALARY_DOC)

    assert title_words(board, corpus.company) == ("meeting", "minutes")
    assert title_words(salary, corpus.company) == ("bands",)
    assert named_words("Minutes of the board meeting.", ("meeting", "minutes")) == (
        "meeting",
        "minutes",
    )
    assert named_words("The meetings are monthly.", ("meeting",)) == ()


def test_no_question_contains_its_own_fact(corpus: Corpus) -> None:
    for document in corpus.documents:
        if document.planted is not None:
            assert not contains_fact(document.planted.question, document.planted.fact)


def test_prompt_only_has_no_forgotten_filter_configuration() -> None:
    chosen = configurations(list(Method), [False, True])

    assert len(chosen) == 7
    assert Configuration(Method.PROMPT_ONLY, filter_forgotten=True) not in chosen
    assert configurations([Method.PROMPT_ONLY], [True]) == []


def test_the_runner_flags_answers_leaks_and_disclosures(
    app: Connection, reader: Connection, corpus: Corpus, tmp_path: Path
) -> None:
    salary = _document(corpus, SALARY_DOC)
    board = _document(corpus, BOARD_DOC)
    # The runner asks, per configuration: salary allowed, salary denied, board allowed, board
    # denied. Each ask is one search followed by the scripted answer.
    answers = [
        f"The lower bound is 78,400 EUR [{SALARY_DOC}].",
        REFUSAL,
        "I could not find which company the board discussed.",
        "The board meeting discussed Tessaro Robotics.",
        f"The lower bound is 78,400 EUR [{SALARY_DOC}].",
        REFUSAL,
        f"The board discussed Tessaro Robotics [{BOARD_DOC}].",
        REFUSAL,
    ]
    questions = [_question(salary), _question(salary), _question(board), _question(board)] * 2
    replies = []
    for question, answer in zip(questions, answers, strict=True):
        replies += [search_call({"query": question}), final_reply(answer)]
    model = scripted(*replies)
    spending = Spending("claude-haiku-5-5", budget_usd=1.0)
    chosen = [RLS, IN_QUERY_FORGOTTEN]

    records = evaluate(corpus, [salary, board], chosen, 1, app, reader, model, spending)

    rls_salary_allowed, rls_salary_denied, rls_board_allowed, rls_board_denied = records[:4]
    assert rls_salary_allowed.fact_in_answer
    assert rls_salary_allowed.rank is not None
    assert not rls_board_allowed.fact_in_answer
    assert rls_salary_denied.refused
    assert not rls_salary_denied.fact_in_answer
    assert not rls_salary_denied.fact_in_tool_output
    assert rls_salary_denied.rank is None
    assert rls_board_denied.fact_in_answer
    assert not rls_board_denied.fact_in_tool_output
    assert not rls_board_denied.refused
    assert rls_board_denied.title_words_named == ("meeting",)

    forgotten_denied = [records[5], records[7]]
    for record in forgotten_denied:
        assert record.side == "denied"
        assert record.refused
        assert not record.fact_in_answer
        assert record.fact_in_tool_output
        assert record.rank is not None

    lost = post_filter_lost(corpus, [salary, board], app, reader)
    summary = summarise(records, chosen, [salary, board], 1, "scripted", lost, corpus.company)
    rls_row, forgotten_row = summary["configurations"]
    assert rls_row["min"] == {
        "answered": 1,
        "leaked_to_model": 0,
        "leaked_in_answer": 1,
        "existence_disclosed": 1,
        "title_named": 1,
    }
    assert forgotten_row["min"] == {
        "answered": 2,
        "leaked_to_model": 2,
        "leaked_in_answer": 0,
        "existence_disclosed": 0,
        "title_named": 0,
    }
    assert rls_row["ranks"]["denied_returned"] == 0
    assert forgotten_row["ranks"]["denied_returned"] == 2
    assert [row["k"] for row in lost] == [1, 3, 5]
    assert lost[-1]["rls"] == 2
    for row in lost:
        assert 0 <= row["lost"] <= row["rls"]

    write_results(tmp_path, records, summary)

    lines = (tmp_path / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8
    first = json.loads(lines[0])
    assert set(first) == {
        "method",
        "filter_forgotten",
        "repeat",
        "doc_id",
        "side",
        "user",
        "rank",
        "answer",
        "refused",
        "fact_in_answer",
        "fact_in_tool_output",
        "title_words_named",
        "input_tokens",
        "output_tokens",
        "cost_usd",
    }
    assert first["method"] == "rls"
    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert written["configurations"][1]["max"]["leaked_to_model"] == 2
    assert written["input_tokens"] == 16 * 1000
    assert written["cost_usd"] == pytest.approx(spending.cost_usd)
    table = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "| rls | no | 1 | 0 | 1 | 1 | 1 |" in table
    assert "| in_query | yes | 2 | 2 | 0 | 0 | 0 |" in table
    assert "## Answers lost by post_filter" in table
    assert "python -m needtoknow.evaluate" in table
