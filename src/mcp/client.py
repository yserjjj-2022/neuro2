"""MCP transport client — synchronous wrapper over the async SDK (ADR-0011 §7).

The host loop is synchronous; the official ``mcp`` SDK is async. This Shell
bridges the two with a single long-lived worker task on a dedicated event-loop
thread. All connection lifecycle (enter/exit of the transport + session) runs
inside that one task, so anyio cancel scopes are entered and exited in the same
task — the reason a naive "one task per call" bridge breaks on ``close()``.

Callers submit commands from the host thread via a thread-safe queue; the worker
executes them and returns results through ``concurrent.futures.Future``.

Failures are isolated: methods raise ``MCPClientError`` (never crash the host).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Self

from mcp.client.stdio import StdioServerParameters, stdio_client

from mcp import ClientSession, types

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0


class MCPClientError(RuntimeError):
    """Ошибка транспорта MCP (сбой подключения/вызова)."""


@dataclass(frozen=True)
class ToolInfo:
    """Описание тула, полученное из ``tools/list``.

    Attributes:
        name: Имя тула.
        description: Человекочитаемое описание ("" если нет).
    """

    name: str
    description: str = ""


@dataclass(frozen=True)
class ToolResult:
    """Результат ``tools/call``.

    Attributes:
        name: Имя вызванного тула.
        success: Успешно ли выполнение (не ``is_error``).
        text: Собранный текстовый вывод ("" если нет).
    """

    name: str
    success: bool
    text: str


class _Worker:
    """Единственная задача, владеющая event loop и всеми async-ресурсами.

    Соединение и сессия открываются и закрываются внутри одной и той же
    корутины ``_serve`` — это требование anyio cancel scopes (иначе закрытие
    падает с "cancel scope in a different task").
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="mcp-client-loop", daemon=True
        )
        self._queue: asyncio.Queue[Any] | None = None
        self._ready = threading.Event()
        self.connected = False
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise MCPClientError("worker loop failed to start")

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._bootstrap())
        self._loop.run_forever()

    async def _bootstrap(self) -> None:
        self._queue = asyncio.Queue()
        self._ready.set()
        await self._serve()

    async def _serve(self) -> None:
        """Обрабатывать команды, удерживая соединение в этой же задаче."""
        stack = AsyncExitStack()
        session: ClientSession | None = None
        while True:
            assert self._queue is not None
            cmd, future = await self._queue.get()
            action = cmd[0]
            try:
                if action == "connect_stdio":
                    if session is not None:
                        await stack.aclose()
                        stack = AsyncExitStack()
                        session = None
                    params: StdioServerParameters = cmd[1]
                    read, write = await stack.enter_async_context(
                        stdio_client(params)
                    )
                    session = await stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    await session.initialize()
                    self.connected = True
                    future.set_result(None)
                elif action == "list_tools":
                    if session is None:
                        raise MCPClientError("client is not connected")
                    future.set_result(await session.list_tools())
                elif action == "call_tool":
                    if session is None:
                        raise MCPClientError("client is not connected")
                    future.set_result(await session.call_tool(cmd[1], cmd[2]))
                elif action == "close":
                    await stack.aclose()
                    self.connected = False
                    future.set_result(None)
                    return
            except Exception as exc:  # noqa: BLE001 — изоляция транспорта
                future.set_exception(exc)

    def submit(self, cmd: Any, *, timeout: float = _DEFAULT_TIMEOUT) -> Any:
        """Отправить команду воркеру и дождаться результата."""
        assert self._queue is not None
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (cmd, future))
        return future.result(timeout=timeout)

    def stop(self) -> None:
        """Остановить loop и дождаться завершения потока."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        if not self._loop.is_closed():
            self._loop.close()


def _collect_text(result: types.CallToolResult) -> str:
    """Собрать текстовое содержимое ``CallToolResult`` (чистая)."""
    parts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
    return "\n".join(parts)


class MCPClient:
    """Синхронный MCP-клиент над async SDK (ADR-0011 §7).

    Attributes:
        _worker: Воркер, владеющий event loop и соединением.
        _closed: Флаг идемпотентного закрытия.
    """

    def __init__(self) -> None:
        self._worker = _Worker()
        self._closed = False

    # --- подключение -----------------------------------------------------

    def connect_stdio(
        self,
        command: str,
        args: tuple[str, ...] = (),
        env: dict[str, str] | None = None,
    ) -> None:
        """Запустить MCP-сервер процессом и подключиться (stdio).

        Args:
            command: Исполняемый файл ("npx", "uvx", "node").
            args: Аргументы команды.
            env: Переменные окружения сервера.

        Raises:
            MCPClientError: Если подключение не удалось.
        """
        params = StdioServerParameters(command=command, args=list(args), env=env)
        try:
            self._worker.submit(("connect_stdio", params))
        except MCPClientError:
            raise
        except Exception as exc:
            raise MCPClientError(f"connect failed: {exc}") from exc

    # --- операции --------------------------------------------------------

    @property
    def connected(self) -> bool:
        """Активна ли сессия."""
        return self._worker.connected

    def list_tools(self) -> tuple[ToolInfo, ...]:
        """Получить список тулов сервера (``tools/list``).

        Returns:
            Кортеж ``ToolInfo``.

        Raises:
            MCPClientError: Если клиент не подключён или вызов не удался.
        """
        try:
            result: types.ListToolsResult = self._worker.submit(("list_tools",))
        except MCPClientError:
            raise
        except Exception as exc:
            raise MCPClientError(f"list_tools failed: {exc}") from exc
        return tuple(
            ToolInfo(name=t.name, description=t.description or "")
            for t in result.tools
        )

    def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        """Вызвать тул сервера (``tools/call``).

        Args:
            name: Имя тула.
            arguments: Аргументы вызова.

        Returns:
            ToolResult (``success=False`` при ``is_error``).

        Raises:
            MCPClientError: Если клиент не подключён или вызов не удался.
        """
        try:
            result: types.CallToolResult = self._worker.submit(
                ("call_tool", name, arguments or {})
            )
        except MCPClientError:
            raise
        except Exception as exc:
            raise MCPClientError(f"call_tool failed: {exc}") from exc
        if not isinstance(result, types.CallToolResult):
            raise MCPClientError(f"unexpected result type for tool {name!r}")
        return ToolResult(
            name=name,
            success=not bool(result.is_error),
            text=_collect_text(result),
        )

    # --- завершение ------------------------------------------------------

    def close(self) -> None:
        """Закрыть сессию и остановить воркер (идемпотентно)."""
        if self._closed:
            return
        self._closed = True
        try:
            self._worker.submit(("close",))
        except Exception as exc:  # noqa: BLE001 — закрытие best-effort
            logger.warning("MCP client close error: %s", exc)
        finally:
            self._worker.stop()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["MCPClient", "MCPClientError", "ToolInfo", "ToolResult"]
