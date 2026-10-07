# retrieval.search against the real database and the real embedding model: each planted fact is
# found by the employee allowed to read it and never by the one denied, every hit is a document
# the asker may read, and an employee with narrow access still gets k results.

import psycopg
import pytest
from psycopg.rows import TupleRow

from needtoknow.corpus import Corpus, Document, load_corpus
from needtoknow.db import as_user
from needtoknow.embed import embed_query
from needtoknow.ingest import CORPUS
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
    params = {"query": embed_query("salary bands and payroll"), "k": 5}
    rows = {}
    with as_user(app, corpus.employees["julie"].principals()):
        plan = [row[0] for row in app.execute("EXPLAIN " + _NEAREST_CHUNKS, params).fetchall()]
        app.execute("SET LOCAL hnsw.ef_search = 1")
        for mode in ("off", "strict_order"):
            app.execute("SELECT set_config('hnsw.iterative_scan', %s, true)", [mode])
            rows[mode] = len(app.execute(_NEAREST_CHUNKS, params).fetchall())

    assert any("chunks_embedding" in line for line in plan)
    assert rows["off"] < 5
    assert rows["strict_order"] == 5


@pytest.mark.parametrize("k", [0, -1, MAX_RESULTS + 1])
def test_k_outside_the_range_is_refused(app: Connection, corpus: Corpus, k: int) -> None:
    with pytest.raises(ValueError, match="k must be between 1 and 20"):
        search(app, corpus.employees["julie"].principals(), "holidays", k)
