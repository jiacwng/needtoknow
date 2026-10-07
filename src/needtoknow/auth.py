# Validates a Keycloak access token and turns it into the principals the database filters on.

from collections.abc import Callable
from dataclasses import dataclass

import jwt

from needtoknow.config import Settings
from needtoknow.corpus import EVERYONE

SigningKeyLookup = Callable[[str], jwt.PyJWK]


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class Identity:
    user_id: str
    principals: frozenset[str]


def jwks_signing_key(settings: Settings) -> SigningKeyLookup:
    client = jwt.PyJWKClient(f"{settings.issuer}/protocol/openid-connect/certs")
    return client.get_signing_key_from_jwt


def authenticate(token: str, settings: Settings, signing_key: SigningKeyLookup) -> Identity:
    # Keycloak does not put this client in aud, so azp is the claim that names the
    # client the token was issued to. typ separates an access token from an ID token.
    try:
        claims = jwt.decode(
            token,
            signing_key(token),
            algorithms=["RS256"],
            issuer=settings.issuer,
            options={
                "require": ["exp", "iat", "iss", "azp", "typ", "preferred_username"],
                "verify_aud": False,
            },
        )
    except jwt.PyJWTError as error:
        raise AuthError(f"invalid token: {error}") from error

    if claims["typ"] != "Bearer":
        raise AuthError(f"not an access token: typ is {claims['typ']!r}")
    if claims["azp"] != settings.client_id:
        raise AuthError(f"token was issued to {claims['azp']!r}, not {settings.client_id!r}")

    username = claims["preferred_username"]
    groups = claims.get("groups", [])
    if not isinstance(username, str) or not username:
        raise AuthError("preferred_username must be a non-empty string")
    if not isinstance(groups, list) or not all(isinstance(g, str) and g for g in groups):
        raise AuthError("groups must be a list of non-empty strings")
    # The database receives the principals joined by commas.
    for name in [username, *groups]:
        if "," in name:
            raise AuthError(f"{name!r} contains a comma")

    principals = frozenset({f"user:{username}", EVERYONE, *(f"group:{g}" for g in groups)})
    return Identity(user_id=username, principals=principals)
