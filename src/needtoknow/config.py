# Settings read from NEEDTOKNOW_* environment variables. The defaults are the development
# values from compose.yaml.

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    db_host: str
    db_port: int
    db_name: str
    admin_user: str
    admin_password: str
    owner_password: str
    app_password: str


def load_settings() -> Settings:
    env = os.environ
    return Settings(
        db_host=env.get("NEEDTOKNOW_DB_HOST", "localhost"),
        db_port=int(env.get("NEEDTOKNOW_DB_PORT", "5432")),
        db_name=env.get("NEEDTOKNOW_DB_NAME", "needtoknow"),
        admin_user=env.get("NEEDTOKNOW_ADMIN_USER", "needtoknow"),
        admin_password=env.get("NEEDTOKNOW_ADMIN_PASSWORD", "needtoknow"),
        owner_password=env.get("NEEDTOKNOW_OWNER_PASSWORD", "needtoknow_owner"),
        app_password=env.get("NEEDTOKNOW_APP_PASSWORD", "needtoknow_app"),
    )
