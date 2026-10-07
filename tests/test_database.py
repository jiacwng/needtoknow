# Row-level security against the real database: each employee sees exactly the documents the
# corpus grants them, nothing is visible without an identity, and the app role cannot write.

import psycopg
import pytest
from psycopg import sql
from psycopg.abc import Query
from psycopg.pq import TransactionStatus
from psycopg.rows import TupleRow

from needtoknow.corpus import Corpus
from needtoknow.db import APP_ROLE, as_user
from needtoknow.ingest import ingest, split_into_chunks
from needtoknow.provision import OWNER_ROLE

Connection = psycopg.Connection[TupleRow]


def _ids(connection: Connection, query: Query) -> set[str]:
    return {row[0] for row in connection.execute(query).fetchall()}


def _count(connection: Connection, query: Query) -> int:
    row = connection.execute(query).fetchone()
    assert row is not None
    return int(row[0])


def test_each_employee_sees_exactly_their_documents(app: Connection, corpus: Corpus) -> None:
    expected = {}
    seen_documents = {}
    seen_chunks = {}
    for employee in corpus.employees.values():
        expected[employee.id] = {
            doc.id for doc in corpus.documents if corpus.can_read(employee.id, doc)
        }
        with as_user(app, employee.principals()):
            seen_documents[employee.id] = _ids(app, "SELECT id FROM documents")
            seen_chunks[employee.id] = _ids(app, "SELECT doc_id FROM chunks")

    assert len(expected) * len(corpus.documents) == 480
    assert seen_documents == expected
    assert seen_chunks == expected


@pytest.mark.parametrize("table", ["documents", "doc_access", "chunks"])
def test_no_identity_sees_no_rows(app: Connection, owner: Connection, table: str) -> None:
    query = sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
    assert _count(owner, query) > 0
    assert _count(app, query) == 0


def test_app_role_is_not_privileged_and_owns_nothing(app: Connection) -> None:
    role = app.execute(
        "SELECT rolname, rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
    ).fetchone()
    owned = _count(
        app,
        "SELECT count(*) FROM pg_class "
        "WHERE relowner = (SELECT oid FROM pg_roles WHERE rolname = current_user)",
    )
    assert role == (APP_ROLE, False, False)
    assert owned == 0


def test_every_table_has_row_level_security(owner: Connection) -> None:
    tables = owner.execute(
        "SELECT relname, relowner::regrole::text, relrowsecurity FROM pg_class "
        "WHERE relnamespace = 'public'::regnamespace AND relkind = 'r' ORDER BY relname"
    ).fetchall()
    assert tables == [
        ("chunks", OWNER_ROLE, True),
        ("doc_access", OWNER_ROLE, True),
        ("documents", OWNER_ROLE, True),
    ]


def test_identity_ends_with_the_transaction(app: Connection, corpus: Corpus) -> None:
    with as_user(app, corpus.employees["julie"].principals()):
        assert _count(app, "SELECT count(*) FROM documents") > 0
    assert _count(app, "SELECT count(*) FROM documents") == 0


def test_identity_ends_when_the_transaction_fails(app: Connection, corpus: Corpus) -> None:
    with (
        pytest.raises(psycopg.errors.DivisionByZero),
        as_user(app, corpus.employees["julie"].principals()),
    ):
        app.execute("SELECT 1 / 0")
    assert app.info.transaction_status == TransactionStatus.IDLE
    assert _count(app, "SELECT count(*) FROM documents") == 0


def test_identity_cannot_be_set_inside_an_open_transaction(app: Connection, corpus: Corpus) -> None:
    with (
        as_user(app, corpus.employees["julie"].principals()),
        pytest.raises(RuntimeError, match="no open transaction"),
        as_user(app, corpus.employees["elena"].principals()),
    ):
        pass


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO documents (id, title, body) VALUES ('x', 'x', 'x')",
        "UPDATE documents SET title = 'x'",
        "DELETE FROM doc_access",
        "TRUNCATE chunks",
    ],
)
def test_app_cannot_write(app: Connection, corpus: Corpus, statement: str) -> None:
    with (
        pytest.raises(psycopg.errors.InsufficientPrivilege),
        as_user(app, corpus.employees["julie"].principals()),
    ):
        app.execute(statement)


def test_access_rows_show_only_the_askers_principals(app: Connection, corpus: Corpus) -> None:
    principals = corpus.employees["julie"].principals()
    with as_user(app, principals):
        visible = _ids(app, "SELECT principal FROM doc_access")
    assert visible
    assert visible <= principals


@pytest.mark.parametrize("principal", ["group:everyone,user:elena", ""])
def test_malformed_principal_is_refused(app: Connection, principal: str) -> None:
    with (
        pytest.raises(ValueError, match="empty or contains a comma"),
        as_user(app, ["user:julie", principal]),
    ):
        pass
    assert app.info.transaction_status == TransactionStatus.IDLE


def test_loading_again_replaces_the_rows(owner: Connection, corpus: Corpus) -> None:
    ingest(owner, corpus)
    readers = sum(len(doc.readers) for doc in corpus.documents)
    chunks = sum(len(split_into_chunks(doc)) for doc in corpus.documents)
    assert _count(owner, "SELECT count(*) FROM documents") == len(corpus.documents)
    assert _count(owner, "SELECT count(*) FROM doc_access") == readers
    assert _count(owner, "SELECT count(*) FROM chunks") == chunks
