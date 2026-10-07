# Ingestion against the real database and the real embedding model: chunks keep every paragraph
# and planted fact, and every chunk is embedded. Search over the chunks is in test_retrieval.py.

import numpy as np
import psycopg
from pgvector import Vector
from psycopg.rows import TupleRow

from needtoknow.corpus import EVERYONE, Corpus, Document
from needtoknow.embed import DIMENSIONS
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
