# The agent with the real model, run by hand with -m llm because every run costs money: for three
# planted facts, the allowed employee gets the fact with its citation and the denied employee gets
# the fixed refusal. The first pair is the M1 done-when scenario.

from collections.abc import Iterator

import psycopg
import pytest
from psycopg.rows import TupleRow

from needtoknow.agent import REFUSAL, ask
from needtoknow.config import load_settings
from needtoknow.corpus import Corpus, PlantedFact
from needtoknow.providers import Spending, chat_model

pytestmark = pytest.mark.llm

Connection = psycopg.Connection[TupleRow]

DOCUMENTS = ["hr-salary-bands-2026", "eng-roadmap-2027", "hr-benefits-renewal"]


@pytest.fixture(scope="module")
def spending() -> Iterator[Spending]:
    settings = load_settings()
    spending = Spending(settings.model, settings.budget_usd)
    yield spending
    print(
        f"\n{settings.model}: {spending.input_tokens} input and {spending.output_tokens} output "
        f"tokens, {spending.cost_usd:.6f} USD"
    )


def _planted(corpus: Corpus, doc_id: str) -> PlantedFact:
    (document,) = [doc for doc in corpus.documents if doc.id == doc_id]
    assert document.planted is not None
    return document.planted


@pytest.mark.parametrize("doc_id", DOCUMENTS)
def test_the_allowed_employee_gets_the_fact_and_its_citation(
    app: Connection, corpus: Corpus, spending: Spending, doc_id: str
) -> None:
    planted = _planted(corpus, doc_id)
    employee = corpus.employees[planted.allowed_user]

    answer = ask(
        planted.question,
        app,
        employee.principals(),
        chat_model(load_settings()),
        spending,
    )

    assert not answer.refused, answer.text
    assert planted.fact in answer.text
    assert doc_id in answer.citations
    assert answer.dropped_citations == ()


@pytest.mark.parametrize("doc_id", DOCUMENTS)
def test_the_denied_employee_gets_the_fixed_refusal(
    app: Connection, corpus: Corpus, spending: Spending, doc_id: str
) -> None:
    planted = _planted(corpus, doc_id)
    employee = corpus.employees[planted.denied_user]

    answer = ask(
        planted.question,
        app,
        employee.principals(),
        chat_model(load_settings()),
        spending,
    )

    assert answer.refused, answer.text
    assert answer.text == REFUSAL
    assert planted.fact not in answer.text
    assert doc_id not in answer.citations
