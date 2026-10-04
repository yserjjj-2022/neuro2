"""Minimal stdio MCP server for transport tests (no network, hermetic).

Run as a subprocess by ``test_mcp_client.py`` via the SDK's high-level
``MCPServer``. Exposes two tools:
- ``echo(text)`` — returns ``echo:<text>``;
- ``boom()`` — raises (to exercise error isolation).
"""

from __future__ import annotations

from mcp.server import MCPServer

server = MCPServer("fake-server")


@server.tool()
async def echo(text: str) -> str:
    """Echo the given text back with a prefix."""
    return f"echo:{text}"


@server.tool()
async def boom() -> str:
    """Always raise — for error-path testing."""
    raise RuntimeError("boom")


def main() -> None:
    """Run the server over stdio."""
    server.run("stdio")


if __name__ == "__main__":
    main()
