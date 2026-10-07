# Nearest-neighbour search over the document chunks, run as one employee so that row-level
# security decides which chunks can come back.

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import psycopg

from needtoknow.db import as_user
from needtoknow.embed import embed_query

MAX_RESULTS = 20

_NEAREST_CHUNKS = """
SELECT chunks.doc_id, documents.title, chunks.text, chunks.embedding <=> %(query)s AS distance
FROM chunks
JOIN documents ON documents.id = chunks.doc_id
ORDER BY distance
LIMIT %(k)s
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
    if not 1 <= k <= MAX_RESULTS:
        raise ValueError(f"k must be between 1 and {MAX_RESULTS}, got {k}")
    embedding = embed_query(query)
    with as_user(app, principals):
        # The HNSW index hands over at most hnsw.ef_search candidates before row-level security
        # filters them. Iterative scan keeps reading the index until k permitted rows are found.
        app.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
        rows = app.execute(_NEAREST_CHUNKS, {"query": embedding, "k": k}).fetchall()
    return [Hit(doc_id=row[0], title=row[1], text=row[2], distance=row[3]) for row in rows]
