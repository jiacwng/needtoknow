# Loads the company's workplace data from corpus/workplace.toml: monthly spending by department
# and vendor, monthly forecasts, meetings and tasks. It refuses a department that is not a group
# in the corpus and a person who is not an employee.

import tomllib
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from needtoknow.corpus import Corpus, CorpusError

ACTUAL_MONTHS = 9
FORECAST_MONTHS = 12


@dataclass(frozen=True)
class Spending:
    department: str
    vendor: str
    month: date
    amount_eur: Decimal


@dataclass(frozen=True)
class Forecast:
    department: str
    month: date
    amount_eur: Decimal


@dataclass(frozen=True)
class Meeting:
    organizer: str
    title: str
    starts_at: datetime
    ends_at: datetime
    attendees: tuple[str, ...]


@dataclass(frozen=True)
class Task:
    title: str
    description: str
    owner: str
    assignee: str
    due_date: date


@dataclass(frozen=True)
class Workplace:
    spending: tuple[Spending, ...]
    forecasts: tuple[Forecast, ...]
    meetings: tuple[Meeting, ...]
    tasks: tuple[Task, ...]


def read_workplace(root: Path, corpus: Corpus) -> Workplace:
    path = root / "workplace.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
    except tomllib.TOMLDecodeError as error:
        raise CorpusError(f"{path.name}: {error}") from error
    year = data["year"]
    departments = {group for employee in corpus.employees.values() for group in employee.groups}

    spending = []
    for table in data["spending"]:
        department = _department(table, departments)
        for month, amount in enumerate(_amounts(table, ACTUAL_MONTHS), start=1):
            if amount:
                spending.append(
                    Spending(department, table["vendor"], date(year, month, 1), Decimal(amount))
                )

    forecasts = []
    for table in data["forecasts"]:
        department = _department(table, departments)
        for month, amount in enumerate(_amounts(table, FORECAST_MONTHS), start=1):
            forecasts.append(Forecast(department, date(year, month, 1), Decimal(amount)))

    meetings = []
    for table in data["meetings"]:
        if table["starts_at"].tzinfo is None or table["ends_at"] <= table["starts_at"]:
            raise CorpusError(f"meeting {table['title']!r}: times need an offset and a duration")
        meetings.append(
            Meeting(
                organizer=_employee(table["organizer"], corpus),
                title=table["title"],
                starts_at=table["starts_at"],
                ends_at=table["ends_at"],
                attendees=tuple(_employee(name, corpus) for name in table["attendees"]),
            )
        )

    tasks = [
        Task(
            title=table["title"],
            description=table["description"],
            owner=_employee(table["owner"], corpus),
            assignee=_employee(table["assignee"], corpus),
            due_date=table["due_date"],
        )
        for table in data["tasks"]
    ]
    return Workplace(tuple(spending), tuple(forecasts), tuple(meetings), tuple(tasks))


def _department(table: dict[str, Any], departments: set[str]) -> str:
    department: str = table["department"]
    if department not in departments:
        raise CorpusError(f"workplace.toml: unknown department {department}")
    return department


def _amounts(table: dict[str, Any], months: int) -> list[Any]:
    amounts: list[Any] = table["eur"]
    if len(amounts) != months:
        raise CorpusError(
            f"workplace.toml: {table['department']} needs {months} monthly amounts, "
            f"got {len(amounts)}"
        )
    return amounts


def _employee(name: str, corpus: Corpus) -> str:
    if name not in corpus.employees:
        raise CorpusError(f"workplace.toml: unknown employee {name}")
    return name
