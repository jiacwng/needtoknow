# Workplace data against the real database: each employee reads only the spending, meetings and
# tasks the policies grant, the views add nothing, writes are limited to one's own rows, and the
# seed agrees with the cloud contract document.

from collections.abc import Iterator, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import TupleRow

from needtoknow.corpus import Corpus, canonical
from needtoknow.db import as_user
from needtoknow.ingest import ingest
from needtoknow.provision import load_workplace
from needtoknow.workplace import Workplace

Connection = psycopg.Connection[TupleRow]

DEPARTMENTS = {"executives", "finance", "hr", "sales", "engineering"}
WORKPLACE_TABLES = ["spending", "forecasts", "meetings", "meeting_attendees", "tasks"]
WORKPLACE_VIEWS = ["spending_by_month", "forecast_vs_actual", "open_tasks", "my_meetings"]
OCTOBER = datetime.fromisoformat("2026-10-01T00:00:00+02:00")
DECEMBER = datetime.fromisoformat("2026-12-01T00:00:00+01:00")
NEW_MEETING = (
    "INSERT INTO meetings (organizer, title, starts_at, ends_at) "
    "VALUES (%s, 'Planning', '2026-11-10 10:00+01', '2026-11-10 11:00+01') RETURNING id"
)
NEW_TASK = (
    "INSERT INTO tasks (title, description, owner, assignee, due_date, status) "
    "VALUES ('Write notes', 'Notes of the planning', %s, 'julie', '2026-11-12', %s)"
)


def _rows(
    app: Connection,
    principals: Sequence[str],
    query: str | sql.Composed,
    params: Sequence[Any] = (),
) -> list[tuple[Any, ...]]:
    with as_user(app, principals):
        return app.execute(query, params).fetchall()


def _at(local_time: str) -> datetime:
    return datetime.fromisoformat(local_time + ":00+02:00")


def _principals(corpus: Corpus, employee: str) -> list[str]:
    return sorted(corpus.employees[employee].principals())


def _seed_meeting_id(owner: Connection, title: str) -> int:
    row = owner.execute("SELECT id FROM meetings WHERE title = %s", [title]).fetchone()
    assert row is not None
    return int(row[0])


def _seed_task_id(owner: Connection, title: str) -> int:
    row = owner.execute("SELECT id FROM tasks WHERE title = %s", [title]).fetchone()
    assert row is not None
    return int(row[0])


@pytest.fixture
def restore(owner: Connection, workplace: Workplace) -> Iterator[None]:
    yield
    load_workplace(owner, workplace)


@pytest.mark.parametrize(
    "table", ["spending", "forecasts", "spending_by_month", "forecast_vs_actual"]
)
@pytest.mark.parametrize(
    ("employee", "departments"),
    [
        ("sofia", {"sales"}),
        ("julie", {"engineering"}),
        ("nadia", {"hr"}),
        ("lukas", DEPARTMENTS),
        ("elena", DEPARTMENTS),
        ("rafael", DEPARTMENTS),
    ],
)
def test_spending_is_visible_by_department(
    app: Connection, corpus: Corpus, table: str, employee: str, departments: set[str]
) -> None:
    query = sql.SQL("SELECT DISTINCT department FROM {}").format(sql.Identifier(table))
    seen = {row[0] for row in _rows(app, _principals(corpus, employee), query)}
    assert seen == departments


def test_views_add_up_to_the_visible_rows(app: Connection, corpus: Corpus) -> None:
    principals = _principals(corpus, "sofia")
    spent = _rows(app, principals, "SELECT sum(amount_eur) FROM spending")
    by_month = _rows(app, principals, "SELECT sum(amount_eur) FROM spending_by_month")
    actual = _rows(app, principals, "SELECT sum(actual_eur) FROM forecast_vs_actual")
    forecast = _rows(app, principals, "SELECT sum(amount_eur) FROM forecasts")
    planned = _rows(app, principals, "SELECT sum(forecast_eur) FROM forecast_vs_actual")
    assert spent == by_month == actual
    assert forecast == planned


def test_each_employee_sees_the_meetings_they_attend(
    app: Connection, corpus: Corpus, workplace: Workplace
) -> None:
    for employee in corpus.employees:
        expected = {
            meeting.title: sorted(meeting.attendees)
            for meeting in workplace.meetings
            if employee == meeting.organizer or employee in meeting.attendees
        }
        principals = _principals(corpus, employee)
        titles = {row[0] for row in _rows(app, principals, "SELECT title FROM meetings")}
        mine = {
            row[0]: row[1]
            for row in _rows(app, principals, "SELECT title, attendees FROM my_meetings")
        }
        attendee_rows = _rows(app, principals, "SELECT count(*) FROM meeting_attendees")
        assert titles == set(expected), employee
        assert mine == expected, employee
        assert attendee_rows == [(sum(len(a) for a in expected.values()),)], employee


def test_tasks_are_visible_to_owner_assignee_and_their_manager(
    app: Connection, corpus: Corpus, workplace: Workplace
) -> None:
    for employee in corpus.employees:
        expected = {
            task.title
            for task in workplace.tasks
            if employee in (task.owner, task.assignee)
            or corpus.employees[task.assignee].manager == employee
        }
        principals = _principals(corpus, employee)
        titles = {row[0] for row in _rows(app, principals, "SELECT title FROM tasks")}
        open_titles = {row[0] for row in _rows(app, principals, "SELECT title FROM open_tasks")}
        assert titles == open_titles == expected, employee


def test_busy_times_show_only_times_of_meetings_the_caller_cannot_read(
    app: Connection, corpus: Corpus, workplace: Workplace
) -> None:
    principals = _principals(corpus, "sofia")
    expected = sorted(
        (employee, meeting.starts_at, meeting.ends_at)
        for meeting in workplace.meetings
        for employee in (meeting.organizer, *meeting.attendees)
        if employee in ("amir", "julie")
    )
    with as_user(app, principals):
        cursor = app.execute(
            "SELECT * FROM busy_times(%s, %s, %s)", [["amir", "julie"], OCTOBER, DECEMBER]
        )
        busy = cursor.fetchall()
        columns = [column.name for column in cursor.description or []]
        readable = app.execute("SELECT count(*) FROM meetings WHERE organizer = 'amir'").fetchone()
    assert columns == ["employee", "starts_at", "ends_at"]
    assert sorted(busy) == expected
    assert readable == (0,)


def test_busy_times_keep_only_meetings_overlapping_the_range(
    app: Connection, corpus: Corpus
) -> None:
    window = [["amir"], _at("2026-10-13T11:00"), _at("2026-10-16T16:15")]
    busy = _rows(app, _principals(corpus, "ben"), "SELECT * FROM busy_times(%s, %s, %s)", window)
    assert busy == [
        ("amir", _at("2026-10-13T10:00"), _at("2026-10-13T11:30")),
        ("amir", _at("2026-10-16T16:00"), _at("2026-10-16T16:30")),
    ]


def test_busy_times_need_an_identity(app: Connection, reader: Connection) -> None:
    query = "SELECT count(*) FROM busy_times(%s, %s, %s)"
    assert app.execute(query, [["amir"], OCTOBER, DECEMBER]).fetchone() == (0,)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        reader.execute(query, [["amir"], OCTOBER, DECEMBER])


@pytest.mark.usefixtures("restore")
def test_an_employee_creates_a_meeting_and_invites_attendees(
    app: Connection, corpus: Corpus
) -> None:
    with as_user(app, _principals(corpus, "julie")):
        row = app.execute(NEW_MEETING, ["julie"]).fetchone()
        assert row is not None
        app.execute(
            "INSERT INTO meeting_attendees (meeting_id, employee) VALUES (%s, 'oskar')", [row[0]]
        )
    mine = _rows(app, _principals(corpus, "oskar"), "SELECT attendees FROM my_meetings")
    assert (["oskar"],) in mine


@pytest.mark.usefixtures("restore")
def test_a_meeting_cannot_be_created_for_someone_else(app: Connection, corpus: Corpus) -> None:
    with (
        pytest.raises(psycopg.errors.InsufficientPrivilege),
        as_user(app, _principals(corpus, "julie")),
    ):
        app.execute(NEW_MEETING, ["amir"])


@pytest.mark.usefixtures("restore")
def test_attendees_cannot_be_added_to_someone_elses_meeting(
    app: Connection, owner: Connection, corpus: Corpus
) -> None:
    meeting_id = _seed_meeting_id(owner, "Event pipeline design review")
    with (
        pytest.raises(psycopg.errors.InsufficientPrivilege),
        as_user(app, _principals(corpus, "julie")),
    ):
        app.execute(
            "INSERT INTO meeting_attendees (meeting_id, employee) VALUES (%s, 'ben')", [meeting_id]
        )


@pytest.mark.usefixtures("restore")
@pytest.mark.parametrize(
    ("task_owner", "status", "allowed"),
    [("julie", "open", True), ("amir", "open", False), ("julie", "done", False)],
)
def test_a_task_is_created_open_and_owned_by_its_creator(
    app: Connection, corpus: Corpus, task_owner: str, status: str, allowed: bool
) -> None:
    principals = _principals(corpus, "julie")
    if allowed:
        with as_user(app, principals):
            app.execute(NEW_TASK, [task_owner, status])
        assert ("Write notes",) in _rows(app, principals, "SELECT title FROM open_tasks")
    else:
        with pytest.raises(psycopg.errors.InsufficientPrivilege), as_user(app, principals):
            app.execute(NEW_TASK, [task_owner, status])


@pytest.mark.usefixtures("restore")
@pytest.mark.parametrize(("employee", "updated"), [("ben", 1), ("marco", 0), ("julie", 0)])
def test_only_the_owner_or_assignee_marks_a_task_done(
    app: Connection, owner: Connection, corpus: Corpus, employee: str, updated: int
) -> None:
    task_id = _seed_task_id(owner, "Qualify trade fair leads")
    with as_user(app, _principals(corpus, employee)):
        cursor = app.execute("UPDATE tasks SET status = 'done' WHERE id = %s", [task_id])
        assert cursor.rowcount == updated
    status = owner.execute("SELECT status FROM tasks WHERE id = %s", [task_id]).fetchone()
    assert status == ("done" if updated else "open",)


@pytest.mark.usefixtures("restore")
@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM spending",
        "DELETE FROM forecasts",
        "DELETE FROM meetings",
        "DELETE FROM meeting_attendees",
        "DELETE FROM tasks",
        "UPDATE tasks SET title = 'x'",
        "UPDATE meetings SET title = 'x'",
        "UPDATE spending SET amount_eur = 0",
        "INSERT INTO spending (department, vendor, month, amount_eur) "
        "VALUES ('sales', 'x', '2026-10-01', 1)",
    ],
)
def test_app_cannot_delete_or_change_other_columns(
    app: Connection, corpus: Corpus, statement: str
) -> None:
    with (
        pytest.raises(psycopg.errors.InsufficientPrivilege),
        as_user(app, _principals(corpus, "elena")),
    ):
        app.execute(statement)


@pytest.mark.parametrize("relation", WORKPLACE_TABLES + WORKPLACE_VIEWS)
def test_principals_without_a_user_or_department_see_nothing(
    app: Connection, relation: str
) -> None:
    query = sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(relation))
    assert _rows(app, ["group:everyone"], query) == [(0,)]


@pytest.mark.usefixtures("restore")
@pytest.mark.parametrize("principals", [[], ["group:everyone", "group:engineering"]])
def test_writes_fail_without_a_user(
    app: Connection, owner: Connection, principals: list[str]
) -> None:
    meeting_id = _seed_meeting_id(owner, "Event pipeline design review")
    statements: list[tuple[str, list[Any]]] = [
        (NEW_MEETING, ["julie"]),
        ("INSERT INTO meeting_attendees (meeting_id, employee) VALUES (%s, 'ben')", [meeting_id]),
        (NEW_TASK, ["julie", "open"]),
    ]
    for statement, params in statements:
        with pytest.raises(psycopg.errors.InsufficientPrivilege), as_user(app, principals):
            app.execute(statement, params)
    with as_user(app, principals):
        assert app.execute("UPDATE tasks SET status = 'done'").rowcount == 0


def test_two_users_in_the_principals_are_refused(app: Connection) -> None:
    with (
        pytest.raises(ValueError, match="more than one user"),
        as_user(app, ["user:julie", "user:amir"]),
    ):
        pass


def test_ingesting_again_gives_the_same_rows(
    owner: Connection, corpus: Corpus, workplace: Workplace
) -> None:
    def snapshot() -> dict[str, list[tuple[Any, ...]]]:
        queries = {
            "spending": "SELECT * FROM spending ORDER BY id",
            "forecasts": "SELECT * FROM forecasts ORDER BY department, month",
            "meetings": "SELECT id, organizer, title, starts_at, ends_at FROM meetings ORDER BY id",
            "meeting_attendees": "SELECT * FROM meeting_attendees ORDER BY 1, 2",
            "tasks": "SELECT id, title, description, owner, assignee, due_date, status "
            "FROM tasks ORDER BY id",
        }
        return {name: owner.execute(query).fetchall() for name, query in queries.items()}

    before = snapshot()
    ingest(owner, corpus, workplace)
    after = snapshot()
    assert after == before
    assert len(after["spending"]) == len(workplace.spending)
    assert len(after["forecasts"]) == len(DEPARTMENTS) * 12
    assert len(after["meetings"]) == len(workplace.meetings)
    assert len(after["tasks"]) == len(workplace.tasks)


def test_cloud_spend_shows_the_overrun_the_contract_describes(
    app: Connection, owner: Connection, corpus: Corpus
) -> None:
    contract = next(doc for doc in corpus.documents if doc.id == "fin-cloud-contract")
    assert contract.planted is not None
    commitment = Decimal(canonical(contract.planted.fact))
    assert "Since July 2026 our usage has been above the commitment" in contract.body

    rows = owner.execute(
        "SELECT month, amount_eur FROM spending WHERE vendor = 'Stratus Cloud' ORDER BY month"
    ).fetchall()
    cloud = {month.month: amount for month, amount in rows}
    before_contract = [cloud[m] for m in (1, 2, 3)]
    under_contract = [cloud[m] - commitment for m in (4, 5, 6, 7, 8, 9)]
    assert min(before_contract) > max(cloud[m] for m in (4, 5, 6))
    assert all(excess > 0 for excess in under_contract)
    assert sorted(under_contract)[-3:] == under_contract[-3:]

    query = (
        "SELECT month, difference_eur FROM forecast_vs_actual "
        "WHERE department = 'engineering' AND month >= '2026-07-01' AND month < '2026-10-01'"
    )
    differences = {row[0]: row[1] for row in _rows(app, _principals(corpus, "tomas"), query)}
    assert set(differences) == {date(2026, m, 1) for m in (7, 8, 9)}
    assert all(difference > 0 for difference in differences.values())

    denied = contract.planted.denied_user
    amounts = _rows(
        app,
        _principals(corpus, denied),
        "SELECT amount_eur FROM spending UNION ALL SELECT forecast_eur FROM forecast_vs_actual "
        "UNION ALL SELECT difference_eur FROM forecast_vs_actual",
    )
    assert amounts
    assert all(canonical(contract.planted.fact) not in str(row[0]) for row in amounts)
