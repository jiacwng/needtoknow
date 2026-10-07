# PostgreSQL access: the roles and tables, loading the corpus, and running queries as one
# employee so that row-level security filters every read.

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.pq import TransactionStatus
from psycopg.rows import TupleRow

from needtoknow.config import Settings
from needtoknow.corpus import Corpus

OWNER_ROLE = "needtoknow_owner"
APP_ROLE = "needtoknow_app"
SCHEMA = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"

_CREATE_ROLES = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'needtoknow_owner') THEN
        CREATE ROLE needtoknow_owner;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'needtoknow_app') THEN
        CREATE ROLE needtoknow_app;
    END IF;
END
$$
"""


def connect_admin(settings: Settings) -> psycopg.Connection[TupleRow]:
    return _connect(settings, settings.admin_user, settings.admin_password)


def connect_owner(settings: Settings) -> psycopg.Connection[TupleRow]:
    return _connect(settings, OWNER_ROLE, settings.owner_password)


def connect_app(settings: Settings) -> psycopg.Connection[TupleRow]:
    return _connect(settings, APP_ROLE, settings.app_password)


def prepare_database(admin: psycopg.Connection[Any], settings: Settings) -> None:
    admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
    admin.execute(_CREATE_ROLES)
    for role, password in (
        (OWNER_ROLE, settings.owner_password),
        (APP_ROLE, settings.app_password),
    ):
        # ALTER ROLE takes no bind parameters, so the password is quoted by psycopg instead.
        admin.execute(
            sql.SQL("ALTER ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD {}").format(
                sql.Identifier(role), sql.Literal(password)
            )
        )
    admin.execute("GRANT USAGE, CREATE ON SCHEMA public TO needtoknow_owner")
    admin.execute("GRANT USAGE ON SCHEMA public TO needtoknow_app")


def create_schema(owner: psycopg.Connection[Any]) -> None:
    with owner.transaction():
        owner.execute(SCHEMA.read_text(encoding="utf-8"))


def load_documents(owner: psycopg.Connection[Any], corpus: Corpus) -> None:
    with owner.transaction(), owner.cursor() as cursor:
        cursor.execute("TRUNCATE chunks, doc_access, documents")
        cursor.executemany(
            "INSERT INTO documents (id, title, body) VALUES (%s, %s, %s)",
            [(doc.id, doc.title, doc.body) for doc in corpus.documents],
        )
        cursor.executemany(
            "INSERT INTO doc_access (doc_id, principal) VALUES (%s, %s)",
            [(doc.id, reader) for doc in corpus.documents for reader in doc.readers],
        )


@contextmanager
def as_user(app: psycopg.Connection[Any], principals: Iterable[str]) -> Iterator[None]:
    names = sorted(principals)
    for name in names:
        if not name or "," in name:
            raise ValueError(f"principal {name!r} is empty or contains a comma")
    # Inside an open transaction this block would only be a savepoint, and the identity would
    # outlive it until the outer transaction ends.
    if app.info.transaction_status != TransactionStatus.IDLE:
        raise RuntimeError("as_user needs a connection with no open transaction")
    with app.transaction():
        app.execute("SELECT set_config('app.principals', %s, true)", [",".join(names)])
        yield


def _connect(settings: Settings, user: str, password: str) -> psycopg.Connection[TupleRow]:
    return psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=user,
        password=password,
        autocommit=True,
    )
