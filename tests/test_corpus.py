# The real corpus loads, and each kind of broken corpus is refused with a clear reason.

import re
from pathlib import Path

import pytest

from needtoknow.corpus import CorpusError, load_corpus

CORPUS = Path(__file__).parent.parent / "corpus"

COMPANY = """
name = "Test Company"

[[employees]]
id = "ana"
name = "Ana"
title = "Analyst"
groups = ["finance"]

[[employees]]
id = "bo"
name = "Bo"
title = "Engineer"
groups = ["engineering"]
manager = "ana"
"""


def _document(
    doc_id: str,
    readers: str = '["group:finance"]',
    planted: str = 'fact = "ZEBRA-9"\nquestion = "What is the code?"\n'
    'allowed_user = "ana"\ndenied_user = "bo"\n',
    body: str = "The code is ZEBRA-9.",
) -> str:
    return f'+++\nid = "{doc_id}"\ntitle = "Title"\nreaders = {readers}\n{planted}+++\n{body}\n'


def _write_corpus(root: Path, documents: dict[str, str], company: str = COMPANY) -> Path:
    (root / "documents").mkdir(parents=True)
    (root / "company.toml").write_text(company, encoding="utf-8")
    for name, text in documents.items():
        (root / "documents" / f"{name}.md").write_text(text, encoding="utf-8")
    return root


PUBLIC = _document("welcome", readers='["group:everyone"]', planted="", body="Welcome.")


def test_real_corpus_loads() -> None:
    corpus = load_corpus(CORPUS)
    restricted = [doc for doc in corpus.documents if doc.restricted]

    assert corpus.company == "Calder Systems"
    assert len(corpus.employees) == 12
    assert len(corpus.documents) == 40
    assert len(restricted) == 30
    assert all(doc.planted is not None for doc in restricted)


def test_minimal_corpus_loads(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path, {"code": _document("code"), "welcome": PUBLIC})
    corpus = load_corpus(root)
    code = next(doc for doc in corpus.documents if doc.id == "code")

    assert corpus.can_read("ana", code)
    assert not corpus.can_read("bo", code)
    assert corpus.employees["bo"].principals() == {
        "user:bo",
        "group:everyone",
        "group:engineering",
    }


def test_windows_line_endings_are_accepted(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path, {"code": _document("code").replace("\n", "\r\n")})
    assert load_corpus(root).documents[0].planted is not None


@pytest.mark.parametrize(
    ("documents", "reason"),
    [
        (
            {"code": _document("code", body="Nothing to see.")},
            "does not appear in the document",
        ),
        (
            {"code": _document("code"), "welcome": PUBLIC.replace("Welcome.", "ZEBRA-9")},
            "also appears in welcome",
        ),
        (
            {"code": _document("code"), "copy": _document("copy")},
            "planted in more than one document",
        ),
        (
            {"code": _document("code", readers='["group:finance", "group:engineering"]')},
            "denied_user bo can read the document",
        ),
        (
            {"code": _document("code", readers='["user:bo"]')},
            "allowed_user ana cannot read the document",
        ),
        (
            {"code": _document("code", readers='["group:legal"]')},
            "unknown reader group:legal",
        ),
        (
            {"code": _document("code", planted="")},
            "a restricted document needs a planted fact",
        ),
        (
            {"code": _document("code", readers='["group:everyone"]')},
            "readable by everyone cannot hold a planted fact",
        ),
        (
            {"code": _document("other-name")},
            "does not match the file name",
        ),
        (
            {"code": "id = 'code'\n"},
            "must start with a +++ header",
        ),
        (
            {"code": _document("code", planted='fact = "ZEBRA-9"\n')},
            "planted fact is missing",
        ),
    ],
)
def test_broken_corpus_is_refused(tmp_path: Path, documents: dict[str, str], reason: str) -> None:
    root = _write_corpus(tmp_path, documents)
    with pytest.raises(CorpusError, match=re.escape(reason)):
        load_corpus(root)


def test_unknown_manager_is_refused(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path, {}, company=COMPANY.replace('manager = "ana"', 'manager = "x"'))
    with pytest.raises(CorpusError, match="unknown manager x"):
        load_corpus(root)
