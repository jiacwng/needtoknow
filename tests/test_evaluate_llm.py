# The evaluation runner with the real model, run by hand with -m llm because it costs money: one
# configuration (rls), one run, three planted facts, about twelve model calls.

import psycopg
import pytest
from psycopg.rows import TupleRow

from needtoknow.config import load_settings
from needtoknow.corpus import Corpus
from needtoknow.evaluate import Configuration, evaluate, summarise
from needtoknow.methods import Method
from needtoknow.providers import Spending, chat_model

pytestmark = pytest.mark.llm

Connection = psycopg.Connection[TupleRow]

DOCUMENTS = ["hr-salary-bands-2026", "eng-roadmap-2027", "hr-benefits-renewal"]


def test_rls_answers_allowed_employees_and_leaks_nothing(
    app: Connection, reader: Connection, corpus: Corpus
) -> None:
    settings = load_settings()
    spending = Spending(settings.model, settings.budget_usd)
    documents = [doc for doc in corpus.documents if doc.id in DOCUMENTS]
    chosen = [Configuration(Method.RLS, filter_forgotten=False)]

    records = evaluate(corpus, documents, chosen, 1, app, reader, chat_model(settings), spending)
    summary = summarise(records, chosen, documents, 1, settings.model, [], corpus.company)

    print(
        f"\n{settings.model}: {spending.input_tokens} input and {spending.output_tokens} output "
        f"tokens, {spending.cost_usd:.6f} USD"
    )
    (row,) = summary["configurations"]
    assert row["min"]["answered"] >= 2
    assert row["max"]["leaked_to_model"] == 0
    assert row["max"]["leaked_in_answer"] == 0
