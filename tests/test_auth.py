# auth.py turns a signed access token into principals and refuses every token it should.
# Tokens are signed here with a throwaway RSA key, so no Keycloak server is needed.

import time
from dataclasses import replace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from needtoknow.auth import AuthError, authenticate
from needtoknow.config import load_settings

SETTINGS = replace(
    load_settings(), issuer="http://keycloak.test/realms/needtoknow", client_id="needtoknow-api"
)
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PUBLIC_KEY = jwt.PyJWK.from_dict(RSAAlgorithm.to_jwk(KEY.public_key(), as_dict=True))


def _signing_key(token: str) -> jwt.PyJWK:
    return PUBLIC_KEY


def _token(claims: dict[str, object], key: rsa.RSAPrivateKey = KEY) -> str:
    now = int(time.time())
    base: dict[str, object] = {
        "iss": SETTINGS.issuer,
        "aud": "account",
        "azp": "needtoknow-api",
        "typ": "Bearer",
        "iat": now,
        "exp": now + 300,
        "preferred_username": "lukas",
        "groups": ["finance"],
    }
    merged = {name: value for name, value in (base | claims).items() if value is not None}
    return jwt.encode(merged, key, algorithm="RS256")


def test_valid_token_gives_user_groups_and_everyone() -> None:
    identity = authenticate(_token({}), SETTINGS, _signing_key)

    assert identity.user_id == "lukas"
    assert identity.principals == {"user:lukas", "group:everyone", "group:finance"}


def test_missing_groups_gives_user_and_everyone() -> None:
    identity = authenticate(_token({"groups": None}), SETTINGS, _signing_key)
    assert identity.principals == {"user:lukas", "group:everyone"}


@pytest.mark.parametrize(
    ("claims", "reason"),
    [
        pytest.param({"exp": int(time.time()) - 10}, "expired", id="expired"),
        pytest.param({"exp": None}, "exp", id="no-expiry"),
        pytest.param({"iss": "http://evil.test/realms/needtoknow"}, "issuer", id="wrong-issuer"),
        pytest.param({"azp": "other-client"}, "issued to 'other-client'", id="wrong-azp"),
        pytest.param({"typ": "ID"}, "not an access token", id="id-token"),
        pytest.param({"preferred_username": None}, "preferred_username", id="no-username"),
        pytest.param({"preferred_username": "a,b"}, "comma", id="comma-in-username"),
        pytest.param({"groups": ["finance,hr"]}, "comma", id="comma-in-group"),
        pytest.param({"groups": "finance"}, "groups must be a list", id="groups-not-a-list"),
    ],
)
def test_bad_claims_are_refused(claims: dict[str, object], reason: str) -> None:
    with pytest.raises(AuthError, match=reason):
        authenticate(_token(claims), SETTINGS, _signing_key)


def test_bad_signature_is_refused() -> None:
    with pytest.raises(AuthError, match="Signature verification failed"):
        authenticate(_token({}, key=OTHER_KEY), SETTINGS, _signing_key)


def test_symmetric_algorithm_is_refused() -> None:
    token = jwt.encode({"preferred_username": "lukas"}, "a-shared-secret-of-32-bytes-long", "HS256")
    with pytest.raises(AuthError, match="invalid token"):
        authenticate(token, SETTINGS, _signing_key)


def test_garbage_is_refused() -> None:
    with pytest.raises(AuthError, match="invalid token"):
        authenticate("not-a-token", SETTINGS, _signing_key)
