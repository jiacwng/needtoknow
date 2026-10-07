# The real corpus loads, and each kind of broken corpus is refused with a clear reason.

import re
from pathlib import Path

import pytest

from needtoknow.corpus import CorpusError, canonical, load_corpus

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


def _planted(fact: str) -> str:
    return f'fact = "{fact}"\nquestion = "What is it?"\nallowed_user = "ana"\ndenied_user = "bo"\n'


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
    facts = {doc.planted.fact for doc in restricted if doc.planted is not None}
    assert {fact for fact in facts if fact[0].isdigit()} == {
        "78,400",
        "38,500",
        "412,760",
        "1,240,000",
    }


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Project Kestrel", "project kestrel"),
        ("14%", "14 percent"),
        ("14 %", "14 percent"),
        ("6,800", "6800"),
        ("6 800", "6800"),
        ("1,240,000 EUR", "1240000 eur"),
        ("  two\n words\t", "two words"),
        ("1,5 and 2,25", "1,5 and 2,25"),
        ("items 1, 234", "items 1, 234"),
    ],
)
def test_canonical(text: str, expected: str) -> None:
    assert canonical(text) == expected


def test_facts_are_matched_in_their_canonical_form(tmp_path: Path) -> None:
    code = _document("code", planted=_planted("78,400"), body="The bound is 78 400 EUR.")
    copy = _document("copy", planted=_planted("ZEBRA-9"), body="ZEBRA-9 and 78400.")
    root = _write_corpus(tmp_path, {"code": code, "copy": copy})
    with pytest.raises(CorpusError, match="fact '78,400' also appears in copy"):
        load_corpus(root)


@pytest.mark.parametrize("fact", ["6,800", "14 percent", "14%", "1.6x", "4.62 million", "€25M"])
def test_a_short_numeric_fact_is_refused(tmp_path: Path, fact: str) -> None:
    root = _write_corpus(tmp_path, {"code": _document("code", planted=_planted(fact), body=fact)})
    with pytest.raises(CorpusError, match="is a number with fewer than 5 digits"):
        load_corpus(root)


@pytest.mark.parametrize("fact", ["78,400", "1,240,000 EUR", "PT-26-031", "2.5 times EBITDA"])
def test_a_long_number_or_a_code_is_accepted(tmp_path: Path, fact: str) -> None:
    root = _write_corpus(tmp_path, {"code": _document("code", planted=_planted(fact), body=fact)})
    assert load_corpus(root).documents[0].planted is not None


def test_unknown_manager_is_refused(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path, {}, company=COMPANY.replace('manager = "ana"', 'manager = "x"'))
    with pytest.raises(CorpusError, match="unknown manager x"):
        load_corpus(root)
