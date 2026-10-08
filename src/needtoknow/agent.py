# The question-answering agent: a LangGraph loop in which the model searches and opens the
# documents the asker may read and looks up colleagues, then answers with the ids of the
# documents it used. Development check:
# PYTHONPATH=src .venv/bin/python -m needtoknow.agent --user nadia "your question"

import argparse
import logging
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
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
from needtoknow.methods import Method, Results, search
from needtoknow.providers import Spending, chat_model

REFUSAL = "I found nothing you have access to on this."
NOT_OPEN = "No document with that id is open to you."
NO_COLLEAGUE = "No colleague matches."
MAX_TOOL_ROUNDS = 4
SEARCH_RESULTS = 5
COLLEAGUE_RESULTS = 5

SYSTEM_PROMPT = f"""You answer questions from employees using their company's documents.
Call search_documents to find passages. It returns only documents the employee asking may read,
so you can share anything in them. Answer only from the passages it returns.
Cite every claim with the document id in square brackets, for example [some-document-id].
If the passages do not contain the answer, reply exactly: {REFUSAL}"""

# Under prompt_only the search returns every document, and only this instruction stands between
# the asker and the passages they may not read.
LABELLED_SYSTEM_PROMPT = f"""You answer questions from employees using their company's documents.
Call search_documents to find passages. It returns passages from every document, each labelled
with the readers allowed to see it. The employee asking holds these principals: {{principals}}.
Use only passages whose readers include one of those principals, and treat every other passage
as if it did not exist. Answer only from the passages you may use.
Cite every claim with the document id in square brackets, for example [some-document-id].
If the passages you may use do not contain the answer, reply exactly: {REFUSAL}"""

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

OPEN_DOCUMENT_TOOL = {
    "name": "open_document",
    "description": "Return the title and full text of one company document, given the document "
    "id that search_documents tagged it with. Use it when the passages leave out what you need.",
    "input_schema": {
        "type": "object",
        "properties": {"doc_id": {"type": "string", "description": "The document id."}},
        "required": ["doc_id"],
        "additionalProperties": False,
    },
}

FIND_COLLEAGUE_TOOL = {
    "name": "find_colleague",
    "description": "Look up colleagues in the company directory by part of their name, job "
    f"title or department. Returns up to {COLLEAGUE_RESULTS} people, each with their title, "
    "departments and manager.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A name, title or department to match."}
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

Tools = dict[str, tuple[str, Callable[[str], str]]]

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
    # Every tool result the model was shown. The system prompt and the question are the rest of
    # its input, so a fact absent from these never reached the model.
    tool_outputs: tuple[str, ...] = field(repr=False)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_rounds: int
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]


def ask(
    question: str,
    connection: psycopg.Connection[Any],
    principals: frozenset[str],
    model: BaseChatModel,
    spending: Spending,
    method: Method = Method.RLS,
    filter_forgotten: bool = False,
) -> Answer:
    returned: set[str] = set()
    outputs: list[str] = []

    # The principals are fixed here, outside the model's reach: each tool takes only one string.
    def search_documents(query: str) -> str:
        results = search(connection, principals, query, SEARCH_RESULTS, method, filter_forgotten)
        returned.update(hit.doc_id for hit in results.hits)
        outputs.append(_format_hits(results))
        return outputs[-1]

    def open_document(doc_id: str) -> str:
        document = retrieval.open_document(connection, principals, doc_id)
        if document is None:
            outputs.append(NOT_OPEN)
        else:
            returned.add(doc_id)
            title, body = document
            outputs.append(f"[{doc_id}]\n{title}\n\n{body.strip()}")
        return outputs[-1]

    def find_colleague(query: str) -> str:
        colleagues = retrieval.find_colleagues(connection, principals, query, COLLEAGUE_RESULTS)
        outputs.append("\n".join(map(_format_colleague, colleagues)) or NO_COLLEAGUE)
        return outputs[-1]

    tools: Tools = {
        "search_documents": ("query", search_documents),
        "open_document": ("doc_id", open_document),
        "find_colleague": ("query", find_colleague),
    }
    graph = _build_graph(model, _system_prompt(method, principals), tools, spending)
    final = graph.invoke(
        {
            "messages": [HumanMessage(question)],
            "tool_rounds": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    )
    return _answer(final, returned, tuple(outputs), spending)


def _system_prompt(method: Method, principals: frozenset[str]) -> str:
    if method is Method.PROMPT_ONLY:
        return LABELLED_SYSTEM_PROMPT.format(principals=", ".join(sorted(principals)))
    return SYSTEM_PROMPT


def _build_graph(
    model: BaseChatModel,
    system_prompt: str,
    tools: Tools,
    spending: Spending,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    model_with_tools = model.bind_tools(
        [SEARCH_TOOL, OPEN_DOCUMENT_TOOL, FIND_COLLEAGUE_TOOL], strict=True
    )

    def call_model(state: AgentState) -> dict[str, Any]:
        spending.check()
        reply = model_with_tools.invoke([SystemMessage(system_prompt), *state["messages"]])
        if reply.usage_metadata is None:
            raise RuntimeError("the model reply carries no token usage, so it cannot be priced")
        input_tokens = reply.usage_metadata["input_tokens"]
        output_tokens = reply.usage_metadata["output_tokens"]
        spending.record(input_tokens, output_tokens)
        return {"messages": [reply], "input_tokens": input_tokens, "output_tokens": output_tokens}

    def call_tools(state: AgentState) -> dict[str, Any]:
        reply = state["messages"][-1]
        calls = reply.tool_calls if isinstance(reply, AIMessage) else []
        results = [_run_tool(call, tools) for call in calls]
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


def _run_tool(call: ToolCall, tools: Tools) -> ToolMessage:
    call_id = call["id"] or ""
    if call["name"] not in tools:
        error = f"unknown tool, call one of {', '.join(tools)}"
        return ToolMessage(error, tool_call_id=call_id, status="error")
    argument, run = tools[call["name"]]
    if set(call["args"]) != {argument}:
        error = f"wrong arguments, call {call['name']} with a {argument} only"
        return ToolMessage(error, tool_call_id=call_id, status="error")
    value = call["args"][argument]
    if not isinstance(value, str) or not value.strip():
        error = f"{argument} must be a non-empty string"
        return ToolMessage(error, tool_call_id=call_id, status="error")
    return ToolMessage(run(value), tool_call_id=call_id)


def _format_colleague(colleague: retrieval.Colleague) -> str:
    line = f"{colleague.name}, {colleague.title}, {' and '.join(colleague.groups)}"
    if colleague.manager is not None:
        line += f", reports to {colleague.manager}"
    return line


def _format_hits(results: Results) -> str:
    if not results.hits:
        return "No documents found."
    passages = []
    for hit in results.hits:
        label = f"[{hit.doc_id}]"
        if hit.doc_id in results.readers:
            label += f" (readers: {', '.join(results.readers[hit.doc_id])})"
        passages.append(f"{label}\n{hit.text}")
    return "\n\n".join(passages)


def _answer(
    final: dict[str, Any], returned: set[str], tool_outputs: tuple[str, ...], spending: Spending
) -> Answer:
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
        tool_outputs=tool_outputs,
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
