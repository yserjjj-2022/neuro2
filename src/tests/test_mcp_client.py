"""Transport tests for MCPClient (ADR-0011 §7).

Uses a hermetic stdio subprocess (``fake_mcp_server.py``) — no network, no
external packages. Verifies the sync↔async bridge, ``tools/list``,
``tools/call`` and error isolation.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.mcp.client import MCPClient, MCPClientError

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _server_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{_REPO_ROOT}{os.pathsep}{existing}" if existing else str(_REPO_ROOT)
    )
    return env


@pytest.fixture()
def client() -> Iterator[MCPClient]:
    mcp = MCPClient()
    mcp.connect_stdio(
        sys.executable,
        ("-m", "src.tests.fake_mcp_server"),
        _server_env(),
    )
    try:
        yield mcp
    finally:
        mcp.close()


class TestMCPClient:
    def test_connected(self, client: MCPClient) -> None:
        assert client.connected

    def test_list_tools(self, client: MCPClient) -> None:
        tools = client.list_tools()
        names = {t.name for t in tools}
        assert names == {"echo", "boom"}

    def test_call_tool_ok(self, client: MCPClient) -> None:
        result = client.call_tool("echo", {"text": "hi"})
        assert result.success
        assert result.text == "echo:hi"

    def test_call_tool_error_isolated(self, client: MCPClient) -> None:
        result = client.call_tool("boom")
        assert not result.success

    def test_not_connected_raises(self) -> None:
        mcp = MCPClient()
        try:
            with pytest.raises(MCPClientError):
                mcp.list_tools()
        finally:
            mcp.close()

    def test_bad_command_raises(self) -> None:
        mcp = MCPClient()
        try:
            with pytest.raises(MCPClientError):
                mcp.connect_stdio("definitely-not-a-command-xyz")
        finally:
            mcp.close()
