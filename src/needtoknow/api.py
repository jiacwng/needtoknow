# The HTTP API. Every request is answered as the employee named in its bearer token, and nothing
# in the request body can change who that is.
# Run locally: docker compose up -d --wait, PYTHONPATH=src .venv/bin/python -m needtoknow.ingest,
# then .venv/bin/uvicorn --app-dir src needtoknow.api:app

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from needtoknow import db, retrieval
from needtoknow.auth import AuthError, Identity, SigningKeyLookup, authenticate, jwks_signing_key
from needtoknow.config import Settings, load_settings

_bearer = HTTPBearer(auto_error=False)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=retrieval.MAX_RESULTS)


class SearchResponse(BaseModel):
    user: str
    results: list[retrieval.Hit]


def create_app(settings: Settings, signing_key: SigningKeyLookup) -> FastAPI:
    api = FastAPI(title="needtoknow")

    def current_identity(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    ) -> Identity:
        if credentials is None:
            raise _unauthorized(None)
        try:
            return authenticate(credentials.credentials, settings, signing_key)
        except AuthError:
            raise _unauthorized("invalid_token") from None

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # A plain def runs in FastAPI's thread pool, so the blocking database call and the embedding
    # do not stall other requests. Each request opens its own connection.
    @api.post("/search")
    def search(
        request: SearchRequest, identity: Annotated[Identity, Depends(current_identity)]
    ) -> SearchResponse:
        with db.connect_app(settings) as connection:
            hits = retrieval.search(connection, identity.principals, request.query, request.k)
        return SearchResponse(user=identity.user_id, results=hits)

    return api


def _unauthorized(error: str | None) -> HTTPException:
    # RFC 6750: a request without a token gets a bare challenge, a rejected token names the error.
    challenge = "Bearer" if error is None else f'Bearer error="{error}"'
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="missing bearer token" if error is None else "invalid bearer token",
        headers={"WWW-Authenticate": challenge},
    )


_settings = load_settings()
app = create_app(_settings, jwks_signing_key(_settings))
