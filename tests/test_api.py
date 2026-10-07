# The HTTP API against the real database and model, with tokens signed by the throwaway key from
# conftest.py: no token or a bad one is refused, results are only documents the token's employee
# may read, and nothing in the request body can change who is asking.

from collections.abc import Callable, Iterator

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import TupleRow

from needtoknow.api import create_app
from needtoknow.auth import SigningKeyLookup
from needtoknow.config import Settings
from needtoknow.corpus import Corpus

Sign = Callable[[dict[str, object]], str]

QUESTION = {"query": "What is the salary band for a senior engineer?"}


@pytest.fixture(scope="module")
def client(
    owner: psycopg.Connection[TupleRow],
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
) -> Iterator[TestClient]:
    with TestClient(create_app(token_settings, token_signing_key)) as test_client:
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
