# The Keycloak realm and corpus/company.toml describe the same people in the same groups, so the
# two cannot drift apart.

import json
from pathlib import Path
from typing import Any

from needtoknow.corpus import load_corpus

ROOT = Path(__file__).parent.parent
REALM: dict[str, Any] = json.loads((ROOT / "keycloak" / "needtoknow-realm.json").read_text())
CORPUS = load_corpus(ROOT / "corpus")


def test_realm_users_and_their_groups_match_the_corpus() -> None:
    realm_users = {
        user["username"]: sorted(group.removeprefix("/") for group in user["groups"])
        for user in REALM["users"]
    }
    corpus_users = {employee.id: sorted(employee.groups) for employee in CORPUS.employees.values()}
    assert realm_users == corpus_users


def test_realm_groups_match_the_corpus() -> None:
    corpus_groups = {group for employee in CORPUS.employees.values() for group in employee.groups}
    assert sorted(group["name"] for group in REALM["groups"]) == sorted(corpus_groups)


def test_every_user_can_log_in() -> None:
    for user in REALM["users"]:
        assert user["enabled"]
        assert [c["type"] for c in user["credentials"]] == ["password"]


def test_client_puts_group_names_in_the_access_token() -> None:
    clients = [c for c in REALM["clients"] if c["clientId"] == "needtoknow-api"]
    assert len(clients) == 1
    client = clients[0]
    assert client["publicClient"]
    assert client["directAccessGrantsEnabled"]

    mappers = [
        m
        for m in client["protocolMappers"]
        if m["protocolMapper"] == "oidc-group-membership-mapper"
    ]
    assert len(mappers) == 1
    config = mappers[0]["config"]
    assert config["claim.name"] == "groups"
    assert config["full.path"] == "false"
    assert config["access.token.claim"] == "true"
