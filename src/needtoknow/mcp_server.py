# An MCP server over Streamable HTTP with one tool, search_documents. Every call searches as the
# employee named in the request's bearer token, so an MCP client finds only what they may read.
# Run locally: PYTHONPATH=src .venv/bin/python -m needtoknow.mcp_server

import anyio.to_thread
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import AnyHttpUrl

from needtoknow import db, retrieval
from needtoknow.auth import AuthError, Identity, SigningKeyLookup, authenticate, jwks_signing_key
from needtoknow.config import Settings, load_settings


class EmployeeToken(AccessToken):
    identity: Identity


class EmployeeTokenVerifier:
    def __init__(self, settings: Settings, signing_key: SigningKeyLookup) -> None:
        self.settings = settings
        self.signing_key = signing_key

    # The SDK answers 401 when this returns None. The signing key lookup can block on Keycloak,
    # so it runs in a worker thread.
    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            identity = await anyio.to_thread.run_sync(
                authenticate, token, self.settings, self.signing_key
            )
        except AuthError:
            return None
        return EmployeeToken(
            token=token,
            client_id=self.settings.client_id,
            scopes=[],
            subject=identity.user_id,
            claims={"iss": self.settings.issuer},
            identity=identity,
        )


def create_server(settings: Settings, signing_key: SigningKeyLookup) -> MCPServer:
    server = MCPServer(
        name="needtoknow",
        token_verifier=EmployeeTokenVerifier(settings, signing_key),
        auth=AuthSettings(issuer_url=AnyHttpUrl(settings.issuer), resource_server_url=None),
    )

    @server.tool(description="Search the company documents the signed-in employee may read.")
    def search_documents(query: str, k: int = 5) -> list[retrieval.Hit]:
        token = get_access_token()
        if not isinstance(token, EmployeeToken):
            raise ToolError("no signed-in employee")
        try:
            with db.connect_app(settings) as connection:
                return retrieval.search(connection, token.identity.principals, query, k)
        except ValueError as error:
            raise ToolError(str(error)) from error

    return server


def main() -> None:
    settings = load_settings()
    server = create_server(settings, jwks_signing_key(settings))
    server.run("streamable-http", port=settings.mcp_port)


if __name__ == "__main__":
    main()
