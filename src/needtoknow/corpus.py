# Loads the fictional company and its documents from corpus/, and refuses any corpus whose
# planted facts could make the evaluation report a leak that is not one, or miss one that is.

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVERYONE = "group:everyone"
FRONT_MATTER = "+++"
PLANTED_KEYS = frozenset({"fact", "question", "allowed_user", "denied_user"})


class CorpusError(ValueError):
    pass


@dataclass(frozen=True)
class Employee:
    id: str
    name: str
    title: str
    groups: tuple[str, ...]
    manager: str | None

    def principals(self) -> frozenset[str]:
        return frozenset({f"user:{self.id}", EVERYONE, *(f"group:{g}" for g in self.groups)})


@dataclass(frozen=True)
class PlantedFact:
    fact: str
    question: str
    allowed_user: str
    denied_user: str


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    readers: tuple[str, ...]
    body: str
    planted: PlantedFact | None

    @property
    def restricted(self) -> bool:
        return EVERYONE not in self.readers


@dataclass(frozen=True)
class Corpus:
    company: str
    employees: dict[str, Employee]
    documents: tuple[Document, ...]

    def can_read(self, user_id: str, document: Document) -> bool:
        return not self.employees[user_id].principals().isdisjoint(document.readers)


def load_corpus(root: Path) -> Corpus:
    company = _parse_toml(_read(root / "company.toml"), "company.toml")
    employees = [_parse_employee(table) for table in _tables(company, "employees", "company.toml")]
    documents = tuple(_parse_document(path) for path in sorted((root / "documents").glob("*.md")))
    corpus = Corpus(
        company=_text(company, "name", "company.toml"),
        employees={employee.id: employee for employee in employees},
        documents=documents,
    )

    problems = _check_employees(employees) + _check_documents(corpus)
    if problems:
        raise CorpusError("\n".join(problems))
    return corpus


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _parse_toml(text: str, where: str) -> dict[str, Any]:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise CorpusError(f"{where}: {error}") from error


def _parse_employee(table: dict[str, Any]) -> Employee:
    where = f"employee {table.get('id', '?')}"
    manager = table.get("manager")
    if manager is not None and not isinstance(manager, str):
        raise CorpusError(f"{where}: manager must be a string")
    return Employee(
        id=_text(table, "id", where),
        name=_text(table, "name", where),
        title=_text(table, "title", where),
        groups=_texts(table, "groups", where),
        manager=manager,
    )


def _parse_document(path: Path) -> Document:
    text = _read(path)
    opening = FRONT_MATTER + "\n"
    if not text.startswith(opening):
        raise CorpusError(f"{path.name}: the file must start with a {FRONT_MATTER} header")
    header, closing, body = text[len(opening) :].partition("\n" + FRONT_MATTER + "\n")
    if not closing:
        raise CorpusError(f"{path.name}: the {FRONT_MATTER} header is never closed")

    meta = _parse_toml(header, path.name)
    document_id = _text(meta, "id", path.name)
    if document_id != path.stem:
        raise CorpusError(f"{path.name}: id {document_id!r} does not match the file name")

    present = PLANTED_KEYS & meta.keys()
    if present and present != PLANTED_KEYS:
        missing = ", ".join(sorted(PLANTED_KEYS - present))
        raise CorpusError(f"{path.name}: planted fact is missing {missing}")
    planted = None
    if present:
        planted = PlantedFact(
            fact=_text(meta, "fact", path.name),
            question=_text(meta, "question", path.name),
            allowed_user=_text(meta, "allowed_user", path.name),
            denied_user=_text(meta, "denied_user", path.name),
        )

    return Document(
        id=document_id,
        title=_text(meta, "title", path.name),
        readers=_texts(meta, "readers", path.name),
        body=body,
        planted=planted,
    )


def _check_employees(employees: list[Employee]) -> list[str]:
    problems = []
    ids = [employee.id for employee in employees]
    for employee_id in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"employee {employee_id} is listed more than once")
    for employee in employees:
        if employee.manager is not None and employee.manager not in ids:
            problems.append(f"employee {employee.id}: unknown manager {employee.manager}")
    return problems


def _check_documents(corpus: Corpus) -> list[str]:
    problems = []
    known = {EVERYONE}
    for employee in corpus.employees.values():
        known |= employee.principals()

    facts = [doc.planted.fact for doc in corpus.documents if doc.planted is not None]
    for fact in sorted({f for f in facts if facts.count(f) > 1}):
        problems.append(f"fact {fact!r} is planted in more than one document")

    for doc in corpus.documents:
        for reader in doc.readers:
            if reader not in known:
                problems.append(f"{doc.id}: unknown reader {reader}")

        if doc.planted is None:
            if doc.restricted:
                problems.append(f"{doc.id}: a restricted document needs a planted fact")
            continue
        if not doc.restricted:
            problems.append(f"{doc.id}: a document readable by everyone cannot hold a planted fact")

        planted = doc.planted
        if planted.fact not in doc.body:
            problems.append(f"{doc.id}: fact {planted.fact!r} does not appear in the document")
        for other in corpus.documents:
            if other is not doc and planted.fact in other.title + "\n" + other.body:
                problems.append(f"{doc.id}: fact {planted.fact!r} also appears in {other.id}")

        for role, user_id, should_read in (
            ("allowed_user", planted.allowed_user, True),
            ("denied_user", planted.denied_user, False),
        ):
            if user_id not in corpus.employees:
                problems.append(f"{doc.id}: {role} {user_id} is not an employee")
            elif corpus.can_read(user_id, doc) != should_read:
                verb = "cannot" if should_read else "can"
                problems.append(f"{doc.id}: {role} {user_id} {verb} read the document")
    return problems


def _text(table: dict[str, Any], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise CorpusError(f"{where}: {key} must be a non-empty string")
    return value


def _texts(table: dict[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise CorpusError(f"{where}: {key} must be a non-empty list of strings")
    return tuple(value)


def _tables(table: dict[str, Any], key: str, where: str) -> list[dict[str, Any]]:
    value = table.get(key)
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise CorpusError(f"{where}: {key} must be a list of tables")
    return value
