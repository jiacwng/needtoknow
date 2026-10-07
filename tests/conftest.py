# Database fixtures: an admin connection creates the roles and tables, the owner ingests the
# real corpus with its embeddings, and the tests read through the app role.

from collections.abc import Iterator

import psycopg
import pytest
from psycopg.rows import TupleRow

from needtoknow.config import load_settings
from needtoknow.corpus import Corpus, load_corpus
from needtoknow.db import (
    connect_admin,
    connect_app,
    connect_owner,
    create_schema,
    prepare_database,
)
from needtoknow.ingest import CORPUS, ingest


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return load_corpus(CORPUS)


@pytest.fixture(scope="session")
def owner(corpus: Corpus) -> Iterator[psycopg.Connection[TupleRow]]:
    settings = load_settings()
    try:
        admin = connect_admin(settings)
    except psycopg.OperationalError as error:
        pytest.fail(
            f"cannot reach PostgreSQL at {settings.db_host}:{settings.db_port}, "
            f"start it with: docker compose up -d --wait db\n{error}",
            pytrace=False,
        )
    with admin:
        prepare_database(admin, settings)
    with connect_owner(settings) as connection:
        create_schema(connection)
        ingest(connection, corpus)
        yield connection


@pytest.fixture(scope="session")
def app(owner: psycopg.Connection[TupleRow]) -> Iterator[psycopg.Connection[TupleRow]]:
    with connect_app(load_settings()) as connection:
        yield connection
