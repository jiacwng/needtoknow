# Shared fixtures. Database: an admin connection creates the roles and tables, the owner ingests
# the real corpus with its embeddings, and the tests read through the app role. Tokens: access
# tokens signed with a throwaway RSA key, so tests that need one run without Keycloak.

import time
from collections.abc import Callable, Iterator
from dataclasses import replace

import jwt
import psycopg
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from psycopg.rows import TupleRow

from needtoknow.auth import SigningKeyLookup
from needtoknow.config import Settings, load_settings
from needtoknow.corpus import Corpus, load_corpus
from needtoknow.db import (
    connect_admin,
    connect_app,
    connect_owner,
    create_schema,
    prepare_database,
)
from needtoknow.ingest import CORPUS, ingest

TOKEN_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return load_corpus(CORPUS)


@pytest.fixture(scope="session")
def owner(corpus: Corpus) -> Iterator[psycopg.Connection[TupleRow]]:
    settings = load_settings()
    try:
        admin = connect_admin(settings)
    except psycopg.OperationalError as error:
        pytest.fail(
            f"cannot reach PostgreSQL at {settings.db_host}:{settings.db_port}, "
            f"start it with: docker compose up -d --wait db\n{error}",
            pytrace=False,
        )
    with admin:
        prepare_database(admin, settings)
    with connect_owner(settings) as connection:
        create_schema(connection)
        ingest(connection, corpus)
        yield connection


@pytest.fixture(scope="session")
def app(owner: psycopg.Connection[TupleRow]) -> Iterator[psycopg.Connection[TupleRow]]:
    with connect_app(load_settings()) as connection:
        yield connection


@pytest.fixture(scope="session")
def token_settings() -> Settings:
    return replace(
        load_settings(), issuer="http://keycloak.test/realms/needtoknow", client_id="needtoknow-api"
    )


@pytest.fixture(scope="session")
def token_signing_key() -> SigningKeyLookup:
    public_key = jwt.PyJWK.from_dict(RSAAlgorithm.to_jwk(TOKEN_KEY.public_key(), as_dict=True))

    def signing_key(token: str) -> jwt.PyJWK:
        return public_key

    return signing_key


@pytest.fixture(scope="session")
def sign_token(token_settings: Settings) -> Callable[[dict[str, object]], str]:
    # The claims given replace the defaults below, and a claim given as None is left out.
    def sign(claims: dict[str, object]) -> str:
        now = int(time.time())
        base: dict[str, object] = {
            "iss": token_settings.issuer,
            "aud": "account",
            "azp": token_settings.client_id,
            "typ": "Bearer",
            "iat": now,
            "exp": now + 300,
            "preferred_username": "lukas",
            "groups": ["finance"],
        }
        merged = {name: value for name, value in (base | claims).items() if value is not None}
        return jwt.encode(merged, TOKEN_KEY, algorithm="RS256")

    return sign
