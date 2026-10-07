# retrieval.search against the real database and the real embedding model: each planted fact is
# found by the employee allowed to read it and never by the one denied, every hit is a document
# the asker may read, and an employee with narrow access still gets k results.

import psycopg
import pytest
from psycopg import sql
from psycopg.abc import Query
from psycopg.rows import TupleRow

from needtoknow import methods
from needtoknow.corpus import Corpus, Document, load_corpus
from needtoknow.db import as_user
from needtoknow.embed import embed_query
from needtoknow.ingest import CORPUS
from needtoknow.methods import POST_FILTER_FACTOR, Method, Results
from needtoknow.retrieval import _NEAREST_CHUNKS, MAX_RESULTS, search

Connection = psycopg.Connection[TupleRow]

PLANTED = [doc for doc in load_corpus(CORPUS).documents if doc.planted is not None]


def test_the_corpus_plants_thirty_facts() -> None:
    assert len(PLANTED) == 30


@pytest.mark.parametrize("doc", PLANTED, ids=[doc.id for doc in PLANTED])
def test_planted_fact_is_found_only_by_the_allowed_user(
    app: Connection, corpus: Corpus, doc: Document
) -> None:
    assert doc.planted is not None
    allowed = corpus.employees[doc.planted.allowed_user]
    denied = corpus.employees[doc.planted.denied_user]

    allowed_hits = search(app, allowed.principals(), doc.planted.question, 5)
    denied_hits = search(app, denied.principals(), doc.planted.question, 5)

    assert doc.id in [hit.doc_id for hit in allowed_hits]
    assert doc.id not in [hit.doc_id for hit in denied_hits]
    documents = {document.id: document for document in corpus.documents}
    for employee, hits in ((allowed, allowed_hits), (denied, denied_hits)):
        assert len(hits) == 5
        for hit in hits:
            assert corpus.can_read(employee.id, documents[hit.doc_id])


def test_hits_come_nearest_first_with_their_title(app: Connection, corpus: Corpus) -> None:
    hits = search(app, corpus.employees["elena"].principals(), "remote work rules", MAX_RESULTS)
    titles = {doc.id: doc.title for doc in corpus.documents}

    assert len(hits) == MAX_RESULTS
    assert [hit.distance for hit in hits] == sorted(hit.distance for hit in hits)
    for hit in hits:
        assert hit.title == titles[hit.doc_id]
        assert hit.text.startswith(hit.title + "\n\n")


def test_the_narrowest_employee_still_gets_k_results(app: Connection, corpus: Corpus) -> None:
    def readable(employee_id: str) -> int:
        return sum(corpus.can_read(employee_id, doc) for doc in corpus.documents)

    narrowest = min(corpus.employees, key=readable)
    principals = corpus.employees[narrowest].principals()
    # With hnsw.ef_search at 1 the index hands over one candidate per round, and the nearest
    # chunks to this question are ones the employee cannot read. Without the iterative scan
    # set in search, row-level security leaves zero rows here.
    app.execute("SET hnsw.ef_search = 1")
    try:
        hits = search(app, principals, "salary bands and payroll", 5)
    finally:
        app.execute("RESET hnsw.ef_search")

    assert narrowest == "julie"
    assert len(hits) == 5


def test_the_query_reads_the_hnsw_index_and_needs_the_iterative_scan(
    app: Connection, corpus: Corpus
) -> None:
    query = sql.SQL(_NEAREST_CHUNKS).format(permitted=sql.SQL(""))
    params = {"query": embed_query("salary bands and payroll"), "limit": 5}
    rows = {}
    with as_user(app, corpus.employees["julie"].principals()):
        plan = [row[0] for row in app.execute(sql.SQL("EXPLAIN ") + query, params).fetchall()]
        app.execute("SET LOCAL hnsw.ef_search = 1")
        for mode in ("off", "strict_order"):
            app.execute("SELECT set_config('hnsw.iterative_scan', %s, true)", [mode])
            rows[mode] = len(app.execute(query, params).fetchall())

    assert any("chunks_embedding" in line for line in plan)
    assert rows["off"] < 5
    assert rows["strict_order"] == 5


@pytest.mark.parametrize("k", [0, -1, MAX_RESULTS + 1])
def test_k_outside_the_range_is_refused(app: Connection, corpus: Corpus, k: int) -> None:
    with pytest.raises(ValueError, match="k must be between 1 and 20"):
        search(app, corpus.employees["julie"].principals(), "holidays", k)


# The enforcement methods the evaluation compares. rls runs as needtoknow_app, the other three as
# needtoknow_reader, which the database lets read every row.
FILTERING = [Method.RLS, Method.IN_QUERY, Method.POST_FILTER]
SALARY_QUESTION = "What is the lower bound of the 2026 salary band for senior engineers?"


def _connection(method: Method, app: Connection, reader: Connection) -> Connection:
    return app if method is Method.RLS else reader


def _rank(results: Results, doc_id: str) -> int | None:
    for position, hit in enumerate(results.hits, start=1):
        if hit.doc_id == doc_id:
            return position
    return None


def _report(method: Method, ranks: dict[str, int | None]) -> None:
    found = [rank for rank in ranks.values() if rank is not None]
    counts = ", ".join(f"top-{k} {sum(rank <= k for rank in found)}" for k in (1, 3, 5))
    print(f"\n{method}: {counts} of {len(ranks)}")


@pytest.mark.parametrize("method", FILTERING)
def test_a_filtering_method_returns_only_readable_documents(
    app: Connection, reader: Connection, corpus: Corpus, method: Method
) -> None:
    connection = _connection(method, app, reader)
    documents = {document.id: document for document in corpus.documents}
    ranks: dict[str, int | None] = {}
    for doc in PLANTED:
        assert doc.planted is not None
        for user_id in (doc.planted.allowed_user, doc.planted.denied_user):
            principals = corpus.employees[user_id].principals()
            results = methods.search(connection, principals, doc.planted.question, 5, method)
            for hit in results.hits:
                assert corpus.can_read(user_id, documents[hit.doc_id])
            if user_id == doc.planted.allowed_user:
                ranks[doc.id] = _rank(results, doc.id)
            else:
                assert _rank(results, doc.id) is None

    _report(method, ranks)
    assert all(rank is not None for rank in ranks.values()), ranks


def test_prompt_only_returns_forbidden_documents_with_their_readers(
    reader: Connection, corpus: Corpus
) -> None:
    documents = {document.id: document for document in corpus.documents}
    ranks: dict[str, int | None] = {}
    for doc in PLANTED:
        assert doc.planted is not None
        principals = corpus.employees[doc.planted.denied_user].principals()
        results = methods.search(reader, principals, doc.planted.question, 5, Method.PROMPT_ONLY)
        assert len(results.hits) == 5
        for hit in results.hits:
            assert results.readers[hit.doc_id] == tuple(sorted(documents[hit.doc_id].readers))
        ranks[doc.id] = _rank(results, doc.id)

    _report(Method.PROMPT_ONLY, ranks)
    assert all(rank is not None for rank in ranks.values()), ranks


def test_a_forgotten_filter_leaks_under_in_query_and_post_filter_but_not_under_rls(
    app: Connection, reader: Connection, corpus: Corpus
) -> None:
    leaks: dict[Method, int] = {}
    for method in FILTERING:
        connection = _connection(method, app, reader)
        leaks[method] = 0
        for doc in PLANTED:
            assert doc.planted is not None
            principals = corpus.employees[doc.planted.denied_user].principals()
            results = methods.search(
                connection, principals, doc.planted.question, 5, method, filter_forgotten=True
            )
            if _rank(results, doc.id) is not None:
                leaks[method] += 1
        print(f"\n{method}, filter forgotten: {leaks[method]} of {len(PLANTED)} facts leak")

    assert leaks[Method.RLS] == 0
    assert leaks[Method.IN_QUERY] == len(PLANTED)
    assert leaks[Method.POST_FILTER] == len(PLANTED)


def test_julie_reads_the_salary_bands_under_in_query_only_when_the_filter_is_forgotten(
    app: Connection, reader: Connection, corpus: Corpus
) -> None:
    julie = corpus.employees["julie"].principals()

    def found(method: Method, filter_forgotten: bool) -> bool:
        connection = _connection(method, app, reader)
        results = methods.search(connection, julie, SALARY_QUESTION, 5, method, filter_forgotten)
        return _rank(results, "hr-salary-bands-2026") is not None

    assert found(Method.IN_QUERY, filter_forgotten=True)
    assert not found(Method.IN_QUERY, filter_forgotten=False)
    assert not found(Method.RLS, filter_forgotten=True)
    assert not found(Method.RLS, filter_forgotten=False)


def test_post_filter_reports_the_hits_it_drops(reader: Connection, corpus: Corpus) -> None:
    julie = corpus.employees["julie"].principals()
    documents = {document.id: document for document in corpus.documents}
    k = 5

    filtered = methods.search(reader, julie, SALARY_QUESTION, k, Method.POST_FILTER)
    # prompt_only at k * POST_FILTER_FACTOR reads the same chunks that post_filter reads.
    everything = methods.search(
        reader, julie, SALARY_QUESTION, k * POST_FILTER_FACTOR, Method.PROMPT_ONLY
    )
    readable = []
    for hit in everything.hits:
        if corpus.can_read("julie", documents[hit.doc_id]):
            readable.append(hit)

    assert filtered.dropped > 0
    assert filtered.dropped == len(everything.hits) - len(readable)
    assert filtered.hits == readable[:k]


def test_post_filter_can_return_fewer_than_k(reader: Connection, corpus: Corpus) -> None:
    julie = corpus.employees["julie"].principals()
    results = methods.search(reader, julie, SALARY_QUESTION, MAX_RESULTS, Method.POST_FILTER)

    assert results.dropped > 0
    assert len(results.hits) < MAX_RESULTS


@pytest.mark.parametrize(
    ("method", "fixture"),
    [(Method.RLS, "reader"), (Method.IN_QUERY, "app"), (Method.PROMPT_ONLY, "app")],
)
def test_a_method_refuses_the_wrong_role(
    request: pytest.FixtureRequest, corpus: Corpus, method: Method, fixture: str
) -> None:
    connection = request.getfixturevalue(fixture)
    with pytest.raises(ValueError, match=f"{method} searches as"):
        methods.search(connection, corpus.employees["julie"].principals(), "holidays", 5, method)


def test_the_reader_role_sees_every_row(reader: Connection, corpus: Corpus) -> None:
    row = reader.execute("SELECT count(*) FROM documents").fetchone()
    assert row == (len(corpus.documents),)


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO documents (id, title, body) VALUES ('x', 'x', 'x')",
        "UPDATE documents SET title = 'x'",
        "DELETE FROM doc_access",
        "TRUNCATE chunks",
    ],
)
def test_the_reader_role_cannot_write(reader: Connection, statement: Query) -> None:
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        reader.execute(statement)
