# auth.py turns a signed access token into principals and refuses every token it should.
# Tokens are signed with the throwaway RSA key from conftest.py, so no Keycloak server is needed.

import time
from collections.abc import Callable

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from needtoknow.auth import AuthError, SigningKeyLookup, authenticate
from needtoknow.config import Settings

Sign = Callable[[dict[str, object]], str]


def test_valid_token_gives_user_groups_and_everyone(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    identity = authenticate(sign_token({}), token_settings, token_signing_key)

    assert identity.user_id == "lukas"
    assert identity.principals == {"user:lukas", "group:everyone", "group:finance"}


def test_missing_groups_gives_user_and_everyone(
    sign_token: Sign, token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    identity = authenticate(sign_token({"groups": None}), token_settings, token_signing_key)
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
def test_bad_claims_are_refused(
    claims: dict[str, object],
    reason: str,
    sign_token: Sign,
    token_settings: Settings,
    token_signing_key: SigningKeyLookup,
) -> None:
    with pytest.raises(AuthError, match=reason):
        authenticate(sign_token(claims), token_settings, token_signing_key)


def test_bad_signature_is_refused(
    token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = jwt.encode({"preferred_username": "lukas"}, other_key, algorithm="RS256")
    with pytest.raises(AuthError, match="Signature verification failed"):
        authenticate(token, token_settings, token_signing_key)


def test_symmetric_algorithm_is_refused(
    token_settings: Settings, token_signing_key: SigningKeyLookup
) -> None:
    token = jwt.encode({"preferred_username": "lukas"}, "a-shared-secret-of-32-bytes-long", "HS256")
    with pytest.raises(AuthError, match="invalid token"):
        authenticate(token, token_settings, token_signing_key)


def test_garbage_is_refused(token_settings: Settings, token_signing_key: SigningKeyLookup) -> None:
    with pytest.raises(AuthError, match="invalid token"):
        authenticate("not-a-token", token_settings, token_signing_key)
