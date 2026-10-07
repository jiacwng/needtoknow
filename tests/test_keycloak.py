# Logs in as each employee through the running Keycloak and checks that auth.py derives the same
# principals from the real token as the corpus gives that employee.

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from needtoknow.auth import (
    AuthError,
    SigningKeyLookup,
    authenticate,
    jwks_signing_key,
)
from needtoknow.config import load_settings
from needtoknow.corpus import load_corpus

pytestmark = pytest.mark.keycloak

SETTINGS = load_settings()
CORPUS = load_corpus(Path(__file__).parent.parent / "corpus")
PASSWORD = "needtoknow"  # noqa: S105, the development password in keycloak/needtoknow-realm.json


def _open(url: str, form: dict[str, str] | None = None) -> bytes:
    data = None if form is None else urllib.parse.urlencode(form).encode()
    # The issuer comes from settings and is plain http in development.
    with urllib.request.urlopen(url, data=data, timeout=10) as response:  # noqa: S310
        body: bytes = response.read()
    return body


def _login(username: str) -> dict[str, Any]:
    form = {
        "grant_type": "password",
        "client_id": SETTINGS.client_id,
        "username": username,
        "password": PASSWORD,
        "scope": "openid",
    }
    tokens: dict[str, Any] = json.loads(
        _open(f"{SETTINGS.issuer}/protocol/openid-connect/token", form)
    )
    return tokens


@pytest.fixture(scope="module")
def signing_key() -> SigningKeyLookup:
    discovery = f"{SETTINGS.issuer}/.well-known/openid-configuration"
    try:
        _open(discovery)
    except urllib.error.URLError as error:
        pytest.fail(
            f"Keycloak does not answer at {discovery} ({error}). "
            "Start it with: docker compose up -d --wait keycloak"
        )
    return jwks_signing_key(SETTINGS)


@pytest.mark.parametrize("username", sorted(CORPUS.employees))
def test_token_gives_the_corpus_principals(username: str, signing_key: SigningKeyLookup) -> None:
    identity = authenticate(_login(username)["access_token"], SETTINGS, signing_key)

    assert identity.user_id == username
    assert identity.principals == CORPUS.employees[username].principals()


def test_id_token_is_refused(signing_key: SigningKeyLookup) -> None:
    with pytest.raises(AuthError, match="not an access token"):
        authenticate(_login("julie")["id_token"], SETTINGS, signing_key)
