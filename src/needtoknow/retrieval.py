# Nearest-neighbour search over the document chunks. search() is what the API runs: it searches
# as one employee, so row-level security decides which chunks can come back.

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql

from needtoknow.db import as_user
from needtoknow.embed import Vector, embed_query

MAX_RESULTS = 20

_NEAREST_CHUNKS = """
SELECT chunks.doc_id, documents.title, chunks.text, chunks.embedding <=> %(query)s AS distance
FROM chunks
JOIN documents ON documents.id = chunks.doc_id
{permitted}
ORDER BY distance
LIMIT %(limit)s
"""

_PERMITTED = """
WHERE EXISTS (
    SELECT 1 FROM doc_access a
    WHERE a.doc_id = chunks.doc_id AND a.principal = ANY(%(principals)s)
)
"""


@dataclass(frozen=True)
class Hit:
    doc_id: str
    title: str
    text: str
    distance: float


def search(
    app: psycopg.Connection[Any], principals: Iterable[str], query: str, k: int
) -> list[Hit]:
    check_k(k)
    embedding = embed_query(query)
    with as_user(app, principals):
        return nearest_chunks(app, embedding, k)


def check_k(k: int) -> None:
    if not 1 <= k <= MAX_RESULTS:
        raise ValueError(f"k must be between 1 and {MAX_RESULTS}, got {k}")


# Runs inside the caller's transaction. With principals, the query itself keeps only chunks of
# documents one of them may read; without, it returns the nearest chunks the role can see.
def nearest_chunks(
    connection: psycopg.Connection[Any],
    embedding: Vector,
    limit: int,
    principals: Collection[str] | None = None,
) -> list[Hit]:
    # The HNSW index hands over at most hnsw.ef_search candidates before row-level security or
    # the WHERE clause filters them. Iterative scan keeps reading the index until limit rows pass.
    connection.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
    permitted = sql.SQL(_PERMITTED) if principals is not None else sql.SQL("")
    query = sql.SQL(_NEAREST_CHUNKS).format(permitted=permitted)
    parameters = {"query": embedding, "limit": limit, "principals": sorted(principals or [])}
    rows = connection.execute(query, parameters).fetchall()
    return [Hit(doc_id=row[0], title=row[1], text=row[2], distance=row[3]) for row in rows]
