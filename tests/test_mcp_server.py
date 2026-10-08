# The MCP server over Streamable HTTP, driven in process through an ASGI transport, against the
# real database and embedding model with tokens signed by the throwaway key from conftest.py.

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx2
import psycopg
import pytest
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent
from psycopg.rows import TupleRow
from starlette.applications import Starlette

from needtoknow.auth import SigningKeyLookup
from needtoknow.config import Settings
from needtoknow.corpus import Corpus
from needtoknow.mcp_server import create_server
from needtoknow.retrieval import MAX_RESULTS

Sign = Callable[[dict[str, object]], str]

URL = "http://127.0.0.1:8001/mcp"
QUERY = "What is the salary band for a senior engineer?"
SALARY_DOC = "hr-salary-bands-2026"

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def asgi_app(token_settings: Settings, token_signing_key: SigningKeyLookup) -> Starlette:
    return create_server(token_settings, token_signing_key).streamable_http_app()


@asynccontextmanager
async def _connect(app: Starlette, token: str) -> AsyncIterator[Client]:
    headers = {"Authorization": f"Bearer {token}"}
    async with (
        app.router.lifespan_context(app),
        httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), headers=headers) as http,
        Client(streamable_http_client(URL, http_client=http)) as client,
    ):
        yield client


async def _post(app: Starlette, headers: dict[str, str]) -> httpx2.Response:
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)) as http:
        return await http.post(URL, json=message, headers=headers)


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="no-header"),
        pytest.param({"Authorization": "Basic anVsaWU6bmVlZHRva25vdw=="}, id="basic-scheme"),
        pytest.param({"Authorization": "Bearer"}, id="empty-bearer"),
    ],
)
async def test_a_request_without_a_token_is_refused(
    asgi_app: Starlette, headers: dict[str, str]
) -> None:
    response = await _post(asgi_app, headers)
    assert response.status_code == 401


@pytest.mark.parametrize(
    "claims",
    [
        pytest.param({"exp": 1}, id="expired"),
        pytest.param({"iss": "http://evil.test/realms/needtoknow"}, id="wrong-issuer"),
        pytest.param({"typ": "ID"}, id="id-token"),
    ],
)
async def test_a_request_with_a_bad_token_is_refused(
    asgi_app: Starlette, sign_token: Sign, claims: dict[str, object]
) -> None:
    response = await _post(asgi_app, {"Authorization": f"Bearer {sign_token(claims)}"})
    assert response.status_code == 401


async def test_a_request_with_garbage_is_refused(asgi_app: Starlette) -> None:
    response = await _post(asgi_app, {"Authorization": "Bearer not-a-token"})
    assert response.status_code == 401


@pytest.mark.parametrize("user", ["julie", "nadia"])
async def test_search_returns_only_documents_the_token_employee_can_read(
    owner: psycopg.Connection[TupleRow],
    asgi_app: Starlette,
    sign_token: Sign,
    corpus: Corpus,
    user: str,
) -> None:
    groups = {"julie": ["engineering"], "nadia": ["hr"]}[user]
    token = sign_token({"preferred_username": user, "groups": groups})

    async with _connect(asgi_app, token) as client:
        result = await client.call_tool("search_documents", {"query": QUERY, "k": MAX_RESULTS})

    assert not result.is_error
    assert result.structured_content is not None
    hits: list[dict[str, Any]] = result.structured_content["result"]
    found = {hit["doc_id"] for hit in hits}
    readable = {doc.id for doc in corpus.documents if corpus.can_read(user, doc)}
    assert found <= readable
    assert (SALARY_DOC in found) == (user == "nadia")
    for hit in hits:
        assert set(hit) == {"doc_id", "title", "text", "distance"}


@pytest.mark.parametrize("k", [0, MAX_RESULTS + 1])
async def test_k_out_of_range_is_a_tool_error(
    owner: psycopg.Connection[TupleRow], asgi_app: Starlette, sign_token: Sign, k: int
) -> None:
    async with _connect(asgi_app, sign_token({})) as client:
        result = await client.call_tool("search_documents", {"query": QUERY, "k": k})

    assert result.is_error
    [content] = result.content
    assert isinstance(content, TextContent)
    assert content.text.endswith(f"k must be between 1 and {MAX_RESULTS}, got {k}")
