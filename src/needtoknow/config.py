# Settings read from NEEDTOKNOW_* environment variables. The defaults are the development
# values from compose.yaml. The API process loads only Settings; the admin and owner passwords
# live in ProvisionSettings, which only ingestion and the tests load.

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    db_host: str
    db_port: int
    db_name: str
    app_password: str
    model_cache: Path
    issuer: str
    client_id: str
    model: str
    budget_usd: float


@dataclass(frozen=True)
class ProvisionSettings:
    admin_user: str
    admin_password: str
    owner_password: str


def load_settings() -> Settings:
    env = os.environ
    return Settings(
        db_host=env.get("NEEDTOKNOW_DB_HOST", "localhost"),
        db_port=int(env.get("NEEDTOKNOW_DB_PORT", "5432")),
        db_name=env.get("NEEDTOKNOW_DB_NAME", "needtoknow"),
        app_password=env.get("NEEDTOKNOW_APP_PASSWORD", "needtoknow_app"),
        model_cache=Path(
            env.get("NEEDTOKNOW_MODEL_CACHE", "~/.cache/needtoknow/models")
        ).expanduser(),
        issuer=env.get("NEEDTOKNOW_ISSUER", "http://localhost:8080/realms/needtoknow"),
        client_id=env.get("NEEDTOKNOW_CLIENT_ID", "needtoknow-api"),
        model=env.get("NEEDTOKNOW_MODEL", "claude-haiku-5-5"),
        budget_usd=float(env.get("NEEDTOKNOW_BUDGET_USD", "2.0")),
    )


def load_provision_settings() -> ProvisionSettings:
    env = os.environ
    return ProvisionSettings(
        admin_user=env.get("NEEDTOKNOW_ADMIN_USER", "needtoknow"),
        admin_password=env.get("NEEDTOKNOW_ADMIN_PASSWORD", "needtoknow"),
        owner_password=env.get("NEEDTOKNOW_OWNER_PASSWORD", "needtoknow_owner"),
    )
