# The question-answering agent: a LangGraph loop in which the model searches the documents the
# asker may read, then answers with the ids of the documents it used. Development check:
# PYTHONPATH=src .venv/bin/python -m needtoknow.agent --user nadia "your question"

import argparse
import logging
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import psycopg
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from needtoknow import retrieval
from needtoknow.config import load_settings
from needtoknow.corpus import load_corpus
from needtoknow.db import connect_app
from needtoknow.ingest import CORPUS
from needtoknow.providers import Spending, chat_model

REFUSAL = "I found nothing you have access to on this."
MAX_TOOL_ROUNDS = 3
SEARCH_RESULTS = 5

SYSTEM_PROMPT = f"""You answer questions from employees using their company's documents.
Call search_documents to find passages. It returns only documents the employee asking may read,
so you can share anything in them. Answer only from the passages it returns.
Cite every claim with the document id in square brackets, for example [some-document-id].
If the passages do not contain the answer, reply exactly: {REFUSAL}"""

SEARCH_TOOL = {
    "name": "search_documents",
    "description": "Search the company documents and return the passages closest to the query, "
    "each tagged with its document id.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to search for."}},
        "required": ["query"],
        "additionalProperties": False,
    },
}

_CITATION = re.compile(r"( ?)\[([a-z0-9-]+(?:, *[a-z0-9-]+)*)\]")

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[str, ...]
    dropped_citations: tuple[str, ...]
    refused: bool
    input_tokens: int
    output_tokens: int
    cost_usd: float


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_rounds: int
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]


def ask(
    question: str,
    app: psycopg.Connection[Any],
    principals: frozenset[str],
    model: BaseChatModel,
    spending: Spending,
) -> Answer:
    returned: set[str] = set()

    # The principals are fixed here, outside the model's reach: the tool takes only a query.
    def search_documents(query: str) -> str:
        hits = retrieval.search(app, principals, query, SEARCH_RESULTS)
        returned.update(hit.doc_id for hit in hits)
        return _format_hits(hits)

    graph = _build_graph(model, search_documents, spending)
    final = graph.invoke(
        {
            "messages": [HumanMessage(question)],
            "tool_rounds": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    )
    return _answer(final, returned, spending)


def _build_graph(
    model: BaseChatModel, search_documents: Callable[[str], str], spending: Spending
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    model_with_tools = model.bind_tools([SEARCH_TOOL], strict=True)

    def call_model(state: AgentState) -> dict[str, Any]:
        spending.check()
        reply = model_with_tools.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        if reply.usage_metadata is None:
            raise RuntimeError("the model reply carries no token usage, so it cannot be priced")
        input_tokens = reply.usage_metadata["input_tokens"]
        output_tokens = reply.usage_metadata["output_tokens"]
        spending.record(input_tokens, output_tokens)
        return {"messages": [reply], "input_tokens": input_tokens, "output_tokens": output_tokens}

    def call_tools(state: AgentState) -> dict[str, Any]:
        reply = state["messages"][-1]
        calls = reply.tool_calls if isinstance(reply, AIMessage) else []
        results = [_run_tool(call, search_documents) for call in calls]
        return {"messages": results, "tool_rounds": state["tool_rounds"] + 1}

    def after_model(state: AgentState) -> str:
        reply = state["messages"][-1]
        if isinstance(reply, AIMessage) and reply.tool_calls:
            if state["tool_rounds"] < MAX_TOOL_ROUNDS:
                return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["tools", END])
    graph.add_edge("tools", "model")
    return graph.compile()


def _run_tool(call: ToolCall, search_documents: Callable[[str], str]) -> ToolMessage:
    call_id = call["id"] or ""
    query = call["args"].get("query")
    if call["name"] != SEARCH_TOOL["name"] or set(call["args"]) != {"query"}:
        error = f"unknown tool or arguments, call {SEARCH_TOOL['name']} with a query only"
        return ToolMessage(error, tool_call_id=call_id, status="error")
    if not isinstance(query, str) or not query.strip():
        return ToolMessage("query must be a non-empty string", tool_call_id=call_id, status="error")
    return ToolMessage(search_documents(query), tool_call_id=call_id)


def _format_hits(hits: list[retrieval.Hit]) -> str:
    if not hits:
        return "No documents found."
    return "\n\n".join(f"[{hit.doc_id}]\n{hit.text}" for hit in hits)


def _answer(final: dict[str, Any], returned: set[str], spending: Spending) -> Answer:
    input_tokens = final["input_tokens"]
    output_tokens = final["output_tokens"]
    cost_usd = spending.price.cost(input_tokens, output_tokens)

    reply = final["messages"][-1]
    text = ""
    if isinstance(reply, AIMessage) and not reply.tool_calls:
        text = reply.text.strip()
    refused = not text or text == REFUSAL
    citations: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    if refused:
        text = REFUSAL
    else:
        text, citations, dropped = _check_citations(text, returned)
    if dropped:
        _log.warning("dropped citations the search never returned: %s", ", ".join(dropped))
    return Answer(
        text=text,
        citations=citations,
        dropped_citations=dropped,
        refused=refused,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


def _check_citations(text: str, returned: set[str]) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    kept: list[str] = []
    dropped: list[str] = []

    def keep_returned(match: re.Match[str]) -> str:
        cited = [doc_id.strip() for doc_id in match.group(2).split(",")]
        verified = []
        for doc_id in cited:
            if doc_id in returned:
                verified.append(doc_id)
                kept.append(doc_id)
            else:
                dropped.append(doc_id)
        if not verified:
            return ""
        return match.group(1) + "".join(f"[{doc_id}]" for doc_id in verified)

    checked = _CITATION.sub(keep_returned, text)
    return checked, tuple(dict.fromkeys(kept)), tuple(dict.fromkeys(dropped))


def main() -> None:
    corpus = load_corpus(CORPUS)
    parser = argparse.ArgumentParser(description="Ask a question as one corpus employee.")
    parser.add_argument("--user", required=True, choices=sorted(corpus.employees))
    parser.add_argument("question")
    args = parser.parse_args()

    settings = load_settings()
    spending = Spending(settings.model, settings.budget_usd)
    principals = corpus.employees[args.user].principals()
    with connect_app(settings) as app:
        answer = ask(args.question, app, principals, chat_model(settings), spending)

    print(answer.text)
    print(f"citations: {', '.join(answer.citations) or 'none'}")
    if answer.dropped_citations:
        print(f"dropped citations: {', '.join(answer.dropped_citations)}")
    print(
        f"{settings.model}: {answer.input_tokens} input and {answer.output_tokens} output "
        f"tokens, {answer.cost_usd:.6f} USD"
    )


if __name__ == "__main__":
    main()
