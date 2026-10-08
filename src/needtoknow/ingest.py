# Splits each document into chunks, embeds them, and writes documents, access rows, chunks and
# the employee directory in one transaction. `python -m needtoknow.ingest` loads corpus/ into a
# freshly created schema.

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

from needtoknow.config import load_provision_settings, load_settings
from needtoknow.corpus import Corpus, Document, load_corpus
from needtoknow.embed import embed_passages
from needtoknow.provision import (
    connect_admin,
    connect_owner,
    create_schema,
    load_documents,
    load_employees,
    prepare_database,
)

CORPUS = Path(__file__).resolve().parents[2] / "corpus"
CHUNK_CHARS = 800


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    position: int
    text: str


@dataclass(frozen=True)
class Ingested:
    documents: int
    access_rows: int
    chunks: int


def split_into_chunks(document: Document) -> list[Chunk]:
    groups: list[list[str]] = []
    for paragraph in _paragraphs(document.body):
        if groups and len(_chunk_text(document.title, [*groups[-1], paragraph])) <= CHUNK_CHARS:
            groups[-1].append(paragraph)
        else:
            groups.append([paragraph])
    return [
        Chunk(doc_id=document.id, position=position, text=_chunk_text(document.title, group))
        for position, group in enumerate(groups)
    ]


def ingest(owner: psycopg.Connection[Any], corpus: Corpus) -> Ingested:
    chunks = [chunk for document in corpus.documents for chunk in split_into_chunks(document)]
    # Embedding takes seconds, so it runs before the transaction opens and holds no locks.
    embeddings = embed_passages([chunk.text for chunk in chunks])
    with owner.transaction(), owner.cursor() as cursor:
        load_documents(owner, corpus)
        load_employees(owner, corpus)
        cursor.executemany(
            "INSERT INTO chunks (doc_id, position, text, embedding) VALUES (%s, %s, %s, %s)",
            [
                (chunk.doc_id, chunk.position, chunk.text, embedding)
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ],
        )
    return Ingested(
        documents=len(corpus.documents),
        access_rows=sum(len(document.readers) for document in corpus.documents),
        chunks=len(chunks),
    )


def main() -> None:
    settings = load_settings()
    provision = load_provision_settings()
    corpus = load_corpus(CORPUS)
    with connect_admin(settings, provision) as admin:
        prepare_database(admin, settings, provision)
    with connect_owner(settings, provision) as owner:
        create_schema(owner)
        ingested = ingest(owner, corpus)
    print(
        f"ingested {ingested.documents} documents, {ingested.access_rows} access rows "
        f"and {ingested.chunks} chunks"
    )


def _paragraphs(body: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", body) if paragraph.strip()]


def _chunk_text(title: str, group: list[str]) -> str:
    return "\n\n".join([title, *group])


if __name__ == "__main__":
    main()
