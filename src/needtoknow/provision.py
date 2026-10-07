# Sets up the database: the admin creates the owner and app roles, the owner creates the tables
# and loads the documents. Only ingestion and the tests run this; the API never does.

from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import TupleRow

from needtoknow.config import ProvisionSettings, Settings
from needtoknow.corpus import Corpus
from needtoknow.db import APP_ROLE, connect

OWNER_ROLE = "needtoknow_owner"
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


def connect_admin(settings: Settings, provision: ProvisionSettings) -> psycopg.Connection[TupleRow]:
    return connect(settings, provision.admin_user, provision.admin_password)


# The admin connects before the vector extension exists, so only the owner and app register it.
def connect_owner(settings: Settings, provision: ProvisionSettings) -> psycopg.Connection[TupleRow]:
    connection = connect(settings, OWNER_ROLE, provision.owner_password)
    register_vector(connection)
    return connection


def prepare_database(
    admin: psycopg.Connection[Any], settings: Settings, provision: ProvisionSettings
) -> None:
    admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
    admin.execute(_CREATE_ROLES)
    for role, password in (
        (OWNER_ROLE, provision.owner_password),
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
