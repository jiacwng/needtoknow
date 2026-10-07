# The agent with a scripted model and the real database: the search tool runs with the asker's
# principals and nothing the model sends can change them, citations the search never returned
# are dropped, and the round cap, the budget and the price table stop runaway or unpriced calls.

import re
from dataclasses import replace

import psycopg
import pytest
from langchain_core.messages import ToolMessage
from psycopg.rows import TupleRow

from fakes import USAGE, ScriptedModel, final_reply, scripted, search_call
from needtoknow.agent import (
    MAX_TOOL_ROUNDS,
    REFUSAL,
    SEARCH_RESULTS,
    SYSTEM_PROMPT,
    Answer,
    ask,
)
from needtoknow.config import load_settings
from needtoknow.corpus import Corpus
from needtoknow.methods import Method
from needtoknow.providers import BudgetExceeded, Spending, chat_model, price_of

Connection = psycopg.Connection[TupleRow]

QUESTION = "What is the lower bound of the 2026 salary band for senior engineers?"
SALARY_DOC = "hr-salary-bands-2026"
MODEL = "claude-haiku-5-5"


def _ask(app: Connection, corpus: Corpus, user: str, model: ScriptedModel) -> Answer:
    principals = corpus.employees[user].principals()
    return ask(QUESTION, app, principals, model, Spending(MODEL, budget_usd=1.0))


def _tool_results(model: ScriptedModel) -> list[ToolMessage]:
    last_prompt = model.prompts[-1]
    return [message for message in last_prompt if isinstance(message, ToolMessage)]


def _tagged_ids(result: ToolMessage) -> set[str]:
    assert isinstance(result.content, str)
    return set(re.findall(r"^\[([a-z0-9-]+)\]$", result.content, flags=re.MULTILINE))


def test_the_allowed_user_gets_the_fact_with_its_citation(app: Connection, corpus: Corpus) -> None:
    model = scripted(
        search_call({"query": QUESTION}),
        final_reply(f"The lower bound is 78,400 EUR [{SALARY_DOC}]."),
    )

    answer = _ask(app, corpus, "nadia", model)

    (result,) = _tool_results(model)
    assert SALARY_DOC in _tagged_ids(result)
    assert "78,400" in result.text
    assert answer.citations == (SALARY_DOC,)
    assert answer.dropped_citations == ()
    assert not answer.refused


@pytest.mark.parametrize("user", ["sofia", "julie", "ben", "nadia"])
def test_the_model_sees_only_documents_the_asker_may_read(
    app: Connection, corpus: Corpus, user: str
) -> None:
    model = scripted(search_call({"query": QUESTION}), final_reply(REFUSAL))

    _ask(app, corpus, user, model)

    (result,) = _tool_results(model)
    documents = {doc.id: doc for doc in corpus.documents}
    returned = _tagged_ids(result)
    assert len(returned) >= 1
    for doc_id in returned:
        assert corpus.can_read(user, documents[doc_id])
    if not corpus.can_read(user, documents[SALARY_DOC]):
        assert "78,400" not in result.text


def test_the_model_cannot_pass_principals_to_the_tool(app: Connection, corpus: Corpus) -> None:
    model = scripted(
        search_call({"query": QUESTION, "principals": ["group:hr", "group:executives"]}),
        final_reply(REFUSAL),
    )

    answer = _ask(app, corpus, "sofia", model)

    (result,) = _tool_results(model)
    assert result.status == "error"
    assert _tagged_ids(result) == set()
    assert answer.refused


def test_a_citation_the_search_never_returned_is_dropped(app: Connection, corpus: Corpus) -> None:
    model = scripted(
        search_call({"query": QUESTION}),
        final_reply(
            f"The lower bound is 78,400 EUR [{SALARY_DOC}] [board-secret-plan]. "
            f"It was approved in January [{SALARY_DOC}, exec-fundraising]."
        ),
    )

    answer = _ask(app, corpus, "nadia", model)

    assert answer.citations == (SALARY_DOC,)
    assert answer.dropped_citations == ("board-secret-plan", "exec-fundraising")
    assert answer.text == (
        f"The lower bound is 78,400 EUR [{SALARY_DOC}]. It was approved in January [{SALARY_DOC}]."
    )


def test_a_denied_document_cited_by_the_model_is_dropped(app: Connection, corpus: Corpus) -> None:
    model = scripted(
        search_call({"query": QUESTION}), final_reply(f"See the salary bands [{SALARY_DOC}].")
    )

    answer = _ask(app, corpus, "sofia", model)

    assert answer.citations == ()
    assert answer.dropped_citations == (SALARY_DOC,)
    assert SALARY_DOC not in answer.text


@pytest.mark.parametrize("text", [REFUSAL, f"  {REFUSAL}\n", ""])
def test_the_fixed_refusal_is_recognised(app: Connection, corpus: Corpus, text: str) -> None:
    model = scripted(search_call({"query": QUESTION}), final_reply(text))

    answer = _ask(app, corpus, "sofia", model)

    assert answer.refused
    assert answer.text == REFUSAL
    assert answer.citations == ()


def test_the_round_cap_stops_a_model_that_keeps_searching(app: Connection, corpus: Corpus) -> None:
    model = scripted(*[search_call({"query": QUESTION}) for _ in range(10)])

    answer = _ask(app, corpus, "nadia", model)

    assert len(model.prompts) == MAX_TOOL_ROUNDS + 1
    assert len(_tool_results(model)) == MAX_TOOL_ROUNDS
    assert answer.refused
    assert answer.text == REFUSAL


def test_tokens_and_cost_are_counted_per_request_and_per_process(
    app: Connection, corpus: Corpus
) -> None:
    spending = Spending(MODEL, budget_usd=1.0)
    principals = corpus.employees["nadia"].principals()

    for _ in range(2):
        model = scripted(search_call({"query": QUESTION}), final_reply(REFUSAL))
        answer = ask(QUESTION, app, principals, model, spending)

    assert answer.input_tokens == 2 * USAGE["input_tokens"]
    assert answer.output_tokens == 2 * USAGE["output_tokens"]
    # 2,000 input tokens at 0.10 USD and 200 output tokens at 0.50 USD per million.
    assert answer.cost_usd == pytest.approx(0.0003)
    assert spending.input_tokens == 4 * USAGE["input_tokens"]
    assert spending.cost_usd == pytest.approx(0.0006)


def test_the_budget_refuses_a_call_once_it_is_spent(app: Connection, corpus: Corpus) -> None:
    # One scripted call costs 0.00015 USD, so the second call finds the budget spent.
    spending = Spending(MODEL, budget_usd=0.0001)
    model = scripted(search_call({"query": QUESTION}), final_reply(REFUSAL))
    principals = corpus.employees["nadia"].principals()

    with pytest.raises(BudgetExceeded, match="budget is 0.0001 USD"):
        ask(QUESTION, app, principals, model, spending)

    assert len(model.prompts) == 1
    assert spending.input_tokens == USAGE["input_tokens"]


def test_a_zero_budget_refuses_the_first_call(app: Connection, corpus: Corpus) -> None:
    spending = Spending(MODEL, budget_usd=0.0)
    model = scripted(final_reply(REFUSAL))

    with pytest.raises(BudgetExceeded):
        ask(QUESTION, app, corpus.employees["nadia"].principals(), model, spending)

    assert model.prompts == []


@pytest.mark.parametrize("model_name", ["claude-haiku-4-5", "gpt-5", ""])
def test_an_unknown_model_is_refused(model_name: str) -> None:
    with pytest.raises(ValueError, match="no price for model"):
        price_of(model_name)
    with pytest.raises(ValueError, match="no price for model"):
        Spending(model_name, budget_usd=1.0)
    with pytest.raises(ValueError, match="no price for model"):
        chat_model(replace(load_settings(), model=model_name))


def test_the_price_table_matches_the_list_prices() -> None:
    assert price_of("claude-haiku-5-5").cost(1_000_000, 1_000_000) == pytest.approx(0.60)
    assert price_of("claude-sonnet-5-5").cost(1_000_000, 1_000_000) == pytest.approx(12.00)
    assert price_of("claude-opus-5-5").cost(1_000_000, 1_000_000) == pytest.approx(24.00)


@pytest.mark.parametrize("method", [Method.IN_QUERY, Method.POST_FILTER])
def test_the_application_filters_hide_denied_documents_from_the_model(
    reader: Connection, corpus: Corpus, method: Method
) -> None:
    model = scripted(search_call({"query": QUESTION}), final_reply(REFUSAL))
    principals = corpus.employees["sofia"].principals()

    ask(QUESTION, reader, principals, model, Spending(MODEL, budget_usd=1.0), method)

    (result,) = _tool_results(model)
    documents = {doc.id: doc for doc in corpus.documents}
    returned = _tagged_ids(result)
    assert len(returned) >= 1
    for doc_id in returned:
        assert corpus.can_read("sofia", documents[doc_id])
    assert "78,400" not in result.text
    assert model.prompts[0][0].text == SYSTEM_PROMPT


def test_prompt_only_labels_every_passage_and_names_the_asker(
    reader: Connection, corpus: Corpus
) -> None:
    model = scripted(search_call({"query": QUESTION}), final_reply(REFUSAL))
    principals = corpus.employees["sofia"].principals()

    ask(QUESTION, reader, principals, model, Spending(MODEL, budget_usd=1.0), Method.PROMPT_ONLY)

    (result,) = _tool_results(model)
    salary = next(doc for doc in corpus.documents if doc.id == SALARY_DOC)
    assert f"[{SALARY_DOC}] (readers: {', '.join(sorted(salary.readers))})\n" in result.text
    assert "78,400" in result.text
    labels = re.findall(r"^\[[a-z0-9-]+\] \(readers: [^)]+\)$", result.text, flags=re.MULTILINE)
    assert len(labels) == SEARCH_RESULTS
    system_prompt = model.prompts[0][0].text
    assert "group:everyone, group:sales, user:sofia" in system_prompt
    assert "Use only passages whose readers include one of those principals" in system_prompt


def test_the_default_method_is_rls_with_unlabelled_passages(
    app: Connection, corpus: Corpus
) -> None:
    model = scripted(search_call({"query": QUESTION}), final_reply(REFUSAL))

    _ask(app, corpus, "nadia", model)

    (result,) = _tool_results(model)
    assert "(readers:" not in result.text
    assert model.prompts[0][0].text == SYSTEM_PROMPT


def test_a_method_on_the_wrong_role_is_refused(app: Connection, corpus: Corpus) -> None:
    model = scripted(search_call({"query": QUESTION}), final_reply(REFUSAL))
    principals = corpus.employees["sofia"].principals()

    with pytest.raises(ValueError, match="prompt_only searches as needtoknow_reader"):
        ask(QUESTION, app, principals, model, Spending(MODEL, budget_usd=1.0), Method.PROMPT_ONLY)
