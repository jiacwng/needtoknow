# The HTTP API against the real database and embedding model, with tokens signed by the throwaway
# key from conftest.py and a scripted chat model: no token or a bad one is refused, results are
# only documents the token's employee may read, and nothing in the body can change who is asking.

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import jwt
import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import TupleRow

from fakes import call_tool, final_reply, scripted, search_call
from needtoknow.agent import REFUSAL
from needtoknow.api import create_app
from needtoknow.auth import SigningKeyLookup
from needtoknow.config import Settings, load_settings
from needtoknow.corpus import Corpus

Sign = Callable[[dict[str, object]], str]

QUESTION = {"query": "What is the salary band for a senior engineer?"}
ASK = {"question": "What is the lower bound of the 2026 salary band for senior engineers?"}
SALARY_DOC = "hr-salary-bands-2026"


@pytest.fixture(scope="module")
def client(
    owner: psycopg.Connection[TupleRow],
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
) -> Iterator[TestClient]:
    # Any call to this model fails: these tests either never reach it or bring their own.
    model = scripted()
    with TestClient(create_app(token_settings, token_signing_key, model)) as test_client:
        yield test_client


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _julie(sign_token: Sign) -> dict[str, str]:
    return _bearer(sign_token({"preferred_username": "julie", "groups": ["engineering"]}))


def test_health_needs_no_token(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="no-header"),
        pytest.param({"Authorization": "Basic anVsaWU6bmVlZHRva25vdw=="}, id="basic-scheme"),
        pytest.param({"Authorization": "Bearer"}, id="empty-bearer"),
    ],
)
def test_search_without_a_token_is_refused(client: TestClient, headers: dict[str, str]) -> None:
    response = client.post("/search", json=QUESTION, headers=headers)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "claims",
    [
        pytest.param({"exp": 1}, id="expired"),
        pytest.param({"iss": "http://evil.test/realms/needtoknow"}, id="wrong-issuer"),
        pytest.param({"typ": "ID"}, id="id-token"),
    ],
)
def test_search_with_a_bad_token_is_refused(
    client: TestClient, sign_token: Sign, claims: dict[str, object]
) -> None:
    response = client.post("/search", json=QUESTION, headers=_bearer(sign_token(claims)))
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == 'Bearer error="invalid_token"'


def test_search_with_garbage_is_refused(client: TestClient) -> None:
    response = client.post("/search", json=QUESTION, headers=_bearer("not-a-token"))
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == 'Bearer error="invalid_token"'


def test_search_with_the_login_server_down_is_unavailable(
    owner: psycopg.Connection[TupleRow], token_settings: Settings, sign_token: Sign
) -> None:
    def unreachable(token: str) -> jwt.PyJWK:
        raise jwt.PyJWKClientConnectionError("connection refused")

    with TestClient(create_app(token_settings, unreachable, scripted())) as down_client:
        response = down_client.post("/search", json=QUESTION, headers=_julie(sign_token))

    assert response.status_code == 503
    assert response.json() == {"detail": "the login server cannot be reached"}
    assert "WWW-Authenticate" not in response.headers


def test_search_returns_only_documents_julie_can_read(
    client: TestClient, sign_token: Sign, corpus: Corpus
) -> None:
    response = client.post("/search", json=QUESTION | {"k": 20}, headers=_julie(sign_token))

    assert response.status_code == 200
    body = response.json()
    readable = {doc.id for doc in corpus.documents if corpus.can_read("julie", doc)}
    assert body["user"] == "julie"
    assert {result["doc_id"] for result in body["results"]} == readable
    for result in body["results"]:
        assert set(result) == {"doc_id", "title", "text", "distance"}


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"user": "elena"}, id="user"),
        pytest.param({"principals": ["group:executives", "group:hr"]}, id="principals"),
        pytest.param({"user_id": "elena", "groups": ["executives"]}, id="user-id-and-groups"),
    ],
)
def test_identity_in_the_body_is_rejected(
    client: TestClient, sign_token: Sign, extra: dict[str, object]
) -> None:
    response = client.post("/search", json=QUESTION | extra, headers=_julie(sign_token))
    assert response.status_code == 422
    assert {error["type"] for error in response.json()["detail"]} == {"extra_forbidden"}


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="no-query"),
        pytest.param({"query": ""}, id="empty-query"),
        pytest.param({"query": "x" * 2001}, id="long-query"),
        pytest.param({"query": 42}, id="query-not-a-string"),
        pytest.param(QUESTION | {"k": 0}, id="k-zero"),
        pytest.param(QUESTION | {"k": 21}, id="k-too-large"),
        pytest.param(QUESTION | {"k": "5"}, id="k-a-string"),
    ],
)
def test_invalid_body_is_refused(
    client: TestClient, sign_token: Sign, body: dict[str, object]
) -> None:
    response = client.post("/search", json=body, headers=_julie(sign_token))
    assert response.status_code == 422


def test_ask_without_a_token_is_refused(client: TestClient) -> None:
    response = client.post("/ask", json=ASK)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(ASK | {"user": "elena"}, id="user"),
        pytest.param(ASK | {"principals": ["group:hr"]}, id="principals"),
        pytest.param({}, id="no-question"),
        pytest.param({"question": ""}, id="empty-question"),
        pytest.param({"question": "x" * 2001}, id="long-question"),
        pytest.param({"question": 42}, id="question-not-a-string"),
    ],
)
def test_ask_with_an_invalid_body_is_refused(
    client: TestClient, sign_token: Sign, body: dict[str, object]
) -> None:
    response = client.post("/ask", json=body, headers=_julie(sign_token))
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("user", "groups", "reply", "expected"),
    [
        pytest.param(
            "nadia",
            ["hr"],
            f"The lower bound is 78,400 EUR [{SALARY_DOC}].",
            {
                "user": "nadia",
                "answer": f"The lower bound is 78,400 EUR [{SALARY_DOC}].",
                "citations": [SALARY_DOC],
                "refused": False,
            },
            id="nadia-allowed",
        ),
        pytest.param(
            "sofia",
            ["sales"],
            REFUSAL,
            {"user": "sofia", "answer": REFUSAL, "citations": [], "refused": True},
            id="sofia-denied",
        ),
    ],
)
def test_ask_answers_as_the_token_employee(
    owner: psycopg.Connection[TupleRow],
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    sign_token: Sign,
    user: str,
    groups: list[str],
    reply: str,
    expected: dict[str, object],
) -> None:
    model = scripted(search_call({"query": ASK["question"]}), final_reply(reply))
    token = sign_token({"preferred_username": user, "groups": groups})

    with TestClient(create_app(token_settings, token_signing_key, model)) as ask_client:
        response = ask_client.post("/ask", json=ASK, headers=_bearer(token))

    assert response.status_code == 200
    assert response.json() == expected
    search_result = model.prompts[-1][-1].text
    assert (f"[{SALARY_DOC}]" in search_result) == (user == "nadia")


def test_ask_after_opening_a_document_keeps_the_response_shape(
    owner: psycopg.Connection[TupleRow],
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    sign_token: Sign,
) -> None:
    reply = f"The lower bound is 78,400 EUR [{SALARY_DOC}]."
    model = scripted(call_tool("open_document", {"doc_id": SALARY_DOC}), final_reply(reply))
    token = sign_token({"preferred_username": "nadia", "groups": ["hr"]})

    with TestClient(create_app(token_settings, token_signing_key, model)) as ask_client:
        response = ask_client.post("/ask", json=ASK, headers=_bearer(token))

    assert response.status_code == 200
    assert response.json() == {
        "user": "nadia",
        "answer": reply,
        "citations": [SALARY_DOC],
        "refused": False,
    }


def test_ask_with_the_budget_spent_is_unavailable(
    owner: psycopg.Connection[TupleRow],
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
    sign_token: Sign,
) -> None:
    model = scripted(final_reply(REFUSAL))
    spent = replace(token_settings, budget_usd=0.0)

    with TestClient(create_app(spent, token_signing_key, model)) as ask_client:
        response = ask_client.post("/ask", json=ASK, headers=_julie(sign_token))

    assert response.status_code == 503
    assert model.prompts == []


def test_concurrent_searches_each_answer_as_their_own_caller(
    client: TestClient, sign_token: Sign, corpus: Corpus
) -> None:
    callers = {
        "julie": _julie(sign_token),
        "nadia": _bearer(sign_token({"preferred_username": "nadia", "groups": ["hr"]})),
    }

    def search_as(user: str) -> tuple[str, dict[str, object]]:
        response = client.post("/search", json=QUESTION | {"k": 10}, headers=callers[user])
        assert response.status_code == 200
        return user, response.json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        replies = list(pool.map(search_as, ["julie", "nadia"] * 20))

    documents = {doc.id: doc for doc in corpus.documents}
    found: dict[str, set[str]] = {"julie": set(), "nadia": set()}
    for user, body in replies:
        assert body["user"] == user
        results = body["results"]
        assert isinstance(results, list)
        for result in results:
            assert corpus.can_read(user, documents[result["doc_id"]])
            found[user].add(result["doc_id"])
    assert len(replies) == 40
    assert SALARY_DOC in found["nadia"]
    assert SALARY_DOC not in found["julie"]


def test_the_api_settings_hold_no_provisioning_password(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "NEEDTOKNOW_ADMIN_USER",
        "NEEDTOKNOW_ADMIN_PASSWORD",
        "NEEDTOKNOW_OWNER_PASSWORD",
        "NEEDTOKNOW_READER_PASSWORD",
    ):
        monkeypatch.setenv(name, "provisioning-secret")
    settings = load_settings()

    assert "provisioning-secret" not in repr(settings)
    assert not hasattr(settings, "admin_password")
    assert not hasattr(settings, "owner_password")
    assert not hasattr(settings, "reader_password")
