# Ingestion against the real database and the real embedding model: chunks keep every paragraph
# and planted fact, every chunk is embedded, and nearest-neighbour search under row-level
# security finds a planted fact for the allowed user and never for the denied user.

import numpy as np
import psycopg
from pgvector import Vector
from psycopg.rows import TupleRow

from needtoknow.corpus import EVERYONE, Corpus, Document
from needtoknow.db import as_user
from needtoknow.embed import DIMENSIONS, embed_query
from needtoknow.ingest import CHUNK_CHARS, split_into_chunks

Connection = psycopg.Connection[TupleRow]

LONG_DOCUMENT = Document(
    id="long",
    title="Long",
    readers=(EVERYONE,),
    body="\n\n".join(["word " * 50] * 6 + ["long " * 200] + ["short one", "short two"]),
    planted=None,
)


def _paragraphs(body: str) -> list[str]:
    return [paragraph.strip() for paragraph in body.split("\n\n") if paragraph.strip()]


def _nearest(app: Connection, principals: frozenset[str], question: str) -> list[str]:
    with as_user(app, principals):
        # Without it, the index hands over at most hnsw.ef_search candidates and row-level
        # security can filter them below the limit.
        app.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
        rows = app.execute(
            "SELECT doc_id FROM chunks ORDER BY embedding <=> %s LIMIT 5", [embed_query(question)]
        ).fetchall()
    return [row[0] for row in rows]


def test_a_long_document_is_split_and_short_paragraphs_are_merged() -> None:
    chunks = split_into_chunks(LONG_DOCUMENT)
    assert [chunk.text.count("\n\n") for chunk in chunks] == [3, 3, 1, 2]
    assert len(chunks[2].text) > CHUNK_CHARS


def test_chunks_fit_the_budget_and_keep_every_paragraph_in_order(corpus: Corpus) -> None:
    for doc in (*corpus.documents, LONG_DOCUMENT):
        chunks = split_into_chunks(doc)
        heading = doc.title + "\n\n"
        kept = []
        for position, chunk in enumerate(chunks):
            assert chunk.doc_id == doc.id
            assert chunk.position == position
            assert chunk.text.startswith(heading)
            group = chunk.text.removeprefix(heading).split("\n\n")
            assert all(group)
            assert len(chunk.text) <= CHUNK_CHARS or len(group) == 1
            kept.extend(group)
        assert kept == _paragraphs(doc.body)


def test_every_planted_fact_is_in_a_chunk_of_its_document(corpus: Corpus) -> None:
    for doc in corpus.documents:
        if doc.planted is not None:
            texts = [chunk.text for chunk in split_into_chunks(doc)]
            assert any(doc.planted.fact in text for text in texts), doc.id


def test_every_document_has_embedded_chunks(owner: Connection, corpus: Corpus) -> None:
    rows = owner.execute("SELECT doc_id, embedding FROM chunks").fetchall()
    assert {doc_id for doc_id, _ in rows} == {doc.id for doc in corpus.documents}
    for _, embedding in rows:
        assert isinstance(embedding, Vector)
        assert embedding.dimensions() == DIMENSIONS
        assert np.isfinite(embedding.to_numpy()).all()


def test_search_finds_planted_facts_only_for_allowed_users(app: Connection, corpus: Corpus) -> None:
    planted = [doc for doc in corpus.documents if doc.planted is not None]
    found = []
    for doc in planted:
        assert doc.planted is not None
        allowed = corpus.employees[doc.planted.allowed_user].principals()
        denied = corpus.employees[doc.planted.denied_user].principals()
        if doc.id in _nearest(app, allowed, doc.planted.question):
            found.append(doc.id)
        denied_results = _nearest(app, denied, doc.planted.question)
        assert len(denied_results) == 5
        assert doc.id not in denied_results

    assert len(planted) == 30
    assert len(found) >= 0.9 * len(planted)
