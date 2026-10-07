# The four ways the evaluation enforces document permissions on a search, so that one set of
# questions can compare them. Only rls ships: the API always searches through retrieval.search.

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import psycopg

from needtoknow import retrieval
from needtoknow.db import APP_ROLE, READER_ROLE
from needtoknow.embed import embed_query

# post_filter reads k * 4 chunks before dropping forbidden ones, so a narrow reader still has a
# chance of k permitted hits. The hits it loses anyway are what the evaluation reports.
POST_FILTER_FACTOR = 4


class Method(StrEnum):
    RLS = "rls"
    IN_QUERY = "in_query"
    POST_FILTER = "post_filter"
    PROMPT_ONLY = "prompt_only"


@dataclass(frozen=True)
class Results:
    hits: list[retrieval.Hit]
    dropped: int = 0
    readers: dict[str, tuple[str, ...]] = field(default_factory=dict)


# rls sends the same SQL as in_query with its filter forgotten, so filter_forgotten changes
# nothing there: the policies of needtoknow_app still filter every row.
def search(
    connection: psycopg.Connection[Any],
    principals: frozenset[str],
    query: str,
    k: int,
    method: Method,
    filter_forgotten: bool = False,
) -> Results:
    role = APP_ROLE if method is Method.RLS else READER_ROLE
    if connection.info.user != role:
        raise ValueError(f"{method} searches as {role}, not as {connection.info.user}")
    if method is Method.RLS:
        return Results(retrieval.search(connection, principals, query, k))

    retrieval.check_k(k)
    embedding = embed_query(query)
    with connection.transaction():
        if method is Method.IN_QUERY:
            permitted = None if filter_forgotten else principals
            return Results(retrieval.nearest_chunks(connection, embedding, k, permitted))
        limit = k * POST_FILTER_FACTOR if method is Method.POST_FILTER else k
        hits = retrieval.nearest_chunks(connection, embedding, limit)
        readers = _readers(connection, {hit.doc_id for hit in hits})

    if method is Method.PROMPT_ONLY:
        return Results(hits, readers={hit.doc_id: readers[hit.doc_id] for hit in hits})
    kept = []
    for hit in hits:
        if filter_forgotten or not principals.isdisjoint(readers[hit.doc_id]):
            kept.append(hit)
    return Results(kept[:k], dropped=len(hits) - len(kept))


def _readers(connection: psycopg.Connection[Any], doc_ids: set[str]) -> dict[str, tuple[str, ...]]:
    rows = connection.execute(
        "SELECT doc_id, principal FROM doc_access WHERE doc_id = ANY(%s) ORDER BY principal",
        [sorted(doc_ids)],
    ).fetchall()
    readers: dict[str, list[str]] = {doc_id: [] for doc_id in doc_ids}
    for doc_id, principal in rows:
        readers[doc_id].append(principal)
    return {doc_id: tuple(names) for doc_id, names in readers.items()}
