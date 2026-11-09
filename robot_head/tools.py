"""Tools from MCP servers, and skills, for the brain to use while it talks.

An MCP server is a program, online or on this computer, that offers tools: tell the time, check the
weather, read a calendar, switch a lamp. The hub keeps every enabled server connected, on a thread
with its own event loop, and hands the brain the tools you allowed. Tools that only look things up
just run; anything that does something waits until the person has said yes in their next message,
because anyone in the room (or a TV) can talk to the robot.

A skill is a written set of instructions for one kind of job (a SKILL.md, the open Agent Skills
format). The brain sees each skill's name and description, and reads the full instructions when one
fits. Scripts that come with a skill are never run.

Searching the internet and reading web pages are built in: OpenRouter's own server tools, which it
runs on its servers in the middle of a reply, with the same key. The robot never sees those calls.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import threading
import time
import urllib.parse
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path

from .store import Store

RETRY_SECONDS = 60.0  # a server that couldn't be reached is tried again this often
MAX_RESULT_CHARS = 6000  # what a tool returns goes back to the model; more would only slow it down
MAX_SKILL_CHARS = 12000
CONFIRM_WITHIN = 300.0  # seconds a "may I?" stays open
READ_SKILL = "read_skill"
NOT_YET = (
    "Not done: this needs the user's OK first. Tell them in one short sentence what you are about to do and ask. "
    "Call it again with the same arguments only after they say yes in their next message."
)
WEB_SEARCH = "openrouter:web_search"
WEB_READ = "openrouter:web_fetch"
# exa works with every model; native is the model maker's own search, auto picks native where there is one.
SEARCH_ENGINES = ("exa", "parallel", "perplexity", "firecrawl", "native", "auto")
READ_TOKENS = 6000  # a page is cut off after about this many tokens: plenty for pulling facts out of it
READS_PER_REPLY = 3


@dataclass
class Internet:
    """Searching and reading the web, done by OpenRouter itself while the model writes its reply."""

    search: bool = True
    read: bool = True
    engine: str = "exa"
    results: int = 3  # per search; more finds more, but the answer gets slower and costs more

    def definitions(self) -> list[dict]:
        tools = []
        if self.search:
            # Short extracts per result: a spoken answer needs the facts, not whole articles.
            parameters = {"max_results": max(1, min(int(self.results), 20)), "search_context_size": "low"}
            if self.engine and self.engine != "auto":
                parameters["engine"] = self.engine
            tools.append({"type": WEB_SEARCH, "parameters": parameters})
        if self.read:
            tools.append({"type": WEB_READ, "parameters": {"max_content_tokens": READ_TOKENS, "max_uses": READS_PER_REPLY}})
        return tools


@dataclass
class _Connection:
    server: dict
    state: str = "connecting"  # connecting, sign_in, ready, error, off
    error: str = ""
    tools: list = field(default_factory=list)  # the server's mcp_types.Tool list
    client: object = None
    wake: asyncio.Event | None = None
    closing: bool = False
    task: object = None  # the concurrent.futures.Future of _keep
    sign_in_url: str = ""  # where the person signs in, while the server waits for that
    signed_in: asyncio.Future | None = None  # gets the code the sign-in page sends back


class _Tokens:
    """Keeps a server's sign-in (OAuth) tokens and registration in robot.db, for the MCP library."""

    def __init__(self, store: Store, server_id: int) -> None:
        self.store, self.server_id = store, server_id

    def _get(self, key: str):
        server = self.store.tool_server(self.server_id)
        return (server or {}).get("oauth", {}).get(key)

    def _set(self, key: str, value: dict) -> None:
        server = self.store.tool_server(self.server_id)
        if server is not None:
            self.store.update_tool_server(self.server_id, oauth={**server["oauth"], key: value})

    async def get_tokens(self):
        from mcp.shared.auth import OAuthToken

        tokens = self._get("tokens")
        return OAuthToken.model_validate(tokens) if tokens else None

    async def set_tokens(self, tokens) -> None:
        self._set("tokens", tokens.model_dump(mode="json"))

    async def get_client_info(self):
        from mcp.shared.auth import OAuthClientInformationFull

        client = self._get("client")
        return OAuthClientInformationFull.model_validate(client) if client else None

    async def set_client_info(self, client_info) -> None:
        self._set("client", client_info.model_dump(mode="json"))


LOOKS = re.compile(
    r"^(get|list|search|find|read|fetch|look|lookup|query|show|check|describe|count|calc|calculate|convert|compute|"
    r"current|now|today|weather|forecast|sun|time|date|datetime)"
)
DOES = re.compile(r"(send|delete|remove|create|update|write|post|buy|pay|book|cancel|set|add|move|publish|order|turn|switch)")


def only_looks(tool) -> tuple[bool, bool]:
    """Whether a tool only looks things up, and whether that is a guess. Servers can say so (read-only hint);
    when they say nothing, the name has to do: get_forecast looks, send_message does something."""
    hints = tool.annotations
    if hints is not None and (hints.read_only_hint is not None or hints.destructive_hint is not None):
        return bool(hints.read_only_hint), False
    name = re.sub(r"[^a-z0-9]+", "_", tool.name.lower())
    return bool(LOOKS.match(name)) and not DOES.search(name), True


def function_name(server_name: str, tool: str) -> str:
    """A name the model can call: letters, digits, _ and -, at most 64 characters."""
    short = server_name.rsplit("/", 1)[-1]
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{short}__{tool}").strip("_")
    return name[:64]


def result_text(result) -> str:
    """What a tool call returned, as text for the model."""
    parts = []
    for item in result.content or []:
        kind = getattr(item, "type", "")
        if kind == "text":
            parts.append(item.text)
        elif kind in ("image", "audio"):
            parts.append(f"(an {kind} the robot can't show)")
        elif kind == "resource_link":
            parts.append(f"(a link: {getattr(item, 'uri', '')})")
        elif kind == "resource":
            parts.append(getattr(getattr(item, "resource", None), "text", "") or "(a file)")
    text = "\n".join(part for part in parts if part)
    if not text and result.structured_content is not None:
        text = json.dumps(result.structured_content, ensure_ascii=False)
    if result.is_error:
        text = "The tool failed: " + (text or "no reason given")
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + " (cut off)"
    return text or "(nothing)"


def _reason(error: BaseException) -> str:
    """The real reason behind an error, which anyio likes to wrap in exception groups."""
    while getattr(error, "exceptions", None):  # an exception group
        error = error.exceptions[0]
    return str(error) or type(error).__name__


def find_command(command: str) -> str | None:
    """Where a launcher like npx or uvx is; uv puts uvx in ~/.local/bin, which a service's PATH lacks."""
    found = shutil.which(command)
    if found is None:
        local = Path.home() / ".local" / "bin" / command
        found = str(local) if local.exists() else None
    return found


class ToolHub:
    def __init__(
        self,
        store: Store,
        timeout: float = 30.0,
        log: Callable[[str, str], None] | None = None,
        callback: str = "http://127.0.0.1:8765/api/tools/oauth/callback",
    ) -> None:
        self.store = store
        self.timeout = timeout
        self.enabled = True  # off: the brain gets no tools and no skills
        self.internet = Internet()
        self.log = log or (lambda kind, text: None)
        self.callback = callback  # where sign-in pages send people back to, unless a server has its own
        self._connections: dict[int, _Connection] = {}
        self._signing_in: dict[str, _Connection] = {}  # by the state a sign-in page sends back
        self._lock = threading.Lock()
        self._choices = store.tool_choices()
        self._skills = [skill for skill in store.skills() if skill["enabled"]]
        self._asked: dict[tuple[str, str], tuple[int, float]] = {}  # (function, arguments) -> (turn, when)
        self._loop: asyncio.AbstractEventLoop | None = None

    # --- Connections ---

    def start(self) -> None:
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="tools").start()
        for server in self.store.tool_servers():
            self._open(server)

    def close(self) -> None:
        if self._loop is None:
            return
        for server_id in list(self._connections):
            self._shut(server_id)
        time.sleep(0.2)  # let them say goodbye
        self._loop.call_soon_threadsafe(self._loop.stop)

    def reload(self, server_id: int) -> None:
        """Connect again after a server was added, changed, switched on or off, or removed."""
        self._shut(server_id)
        server = self.store.tool_server(server_id)
        if server is not None:
            self._open(server)

    def _open(self, server: dict) -> None:
        connection = _Connection(server, state="connecting" if server["enabled"] else "off")
        with self._lock:
            self._connections[server["id"]] = connection
        if server["enabled"] and self._loop is not None:
            connection.task = asyncio.run_coroutine_threadsafe(self._keep(connection), self._loop)

    def _shut(self, server_id: int) -> None:
        with self._lock:
            connection = self._connections.pop(server_id, None)
        if connection is not None and self._loop is not None:
            connection.closing = True
            if connection.state == "ready" and connection.wake is not None:
                self._loop.call_soon_threadsafe(connection.wake.set)  # hang up politely
            elif connection.task is not None:
                connection.task.cancel()  # still connecting, or waiting for a sign-in that won't come now

    def finish_sign_in(self, code: str, state: str, iss: str | None = None) -> str:
        """The sign-in page sent the person back with a code: hand it to the waiting connection. Returns its title."""
        from mcp.shared.auth import AuthorizationCodeResult

        connection = self._signing_in.pop(state, None)
        if connection is None or connection.signed_in is None or self._loop is None:
            raise ValueError("That sign-in isn't expected (any more). Start it again from the Tools tab.")
        result = AuthorizationCodeResult(code=code, state=state, iss=iss)
        self._loop.call_soon_threadsafe(lambda: connection.signed_in.done() or connection.signed_in.set_result(result))
        return connection.server["title"]

    async def _keep(self, connection: _Connection) -> None:
        """Stay connected until closed; reconnect after the connection broke, retry later after a failure."""
        from mcp import Client

        connection.wake = asyncio.Event()
        while not connection.closing:
            connection.wake.clear()
            try:
                async with AsyncExitStack() as stack:
                    client = await stack.enter_async_context(Client(await self._transport(connection, stack)))
                    tools, cursor = [], None
                    while True:
                        listed = await client.list_tools(cursor=cursor)
                        tools += listed.tools
                        cursor = listed.next_cursor
                        if not cursor:
                            break
                    connection.client, connection.tools, connection.state, connection.error = client, tools, "ready", ""
                    connection.sign_in_url = ""
                    self.log("info", f"(tools from {connection.server['title']} are ready: {len(tools)})")
                    await connection.wake.wait()
            except Exception as error:  # noqa: BLE001 - whatever went wrong, the robot keeps going without these tools
                if connection.closing:
                    break
                connection.state, connection.error = "error", _reason(error)
                self.log("error", f"Couldn't connect to {connection.server['title']}: {connection.error}")
                try:
                    await asyncio.wait_for(connection.wake.wait(), RETRY_SECONDS)
                except TimeoutError:
                    pass
            finally:
                connection.client = None
        connection.state = "off"

    async def _transport(self, connection: _Connection, stack: AsyncExitStack):
        from mcp import StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        server = connection.server
        if server["url"]:
            headers = {key: value for key, value in server["headers"].items() if value}
            # Without a key of its own, an online server may ask to sign in; until it does, this does nothing.
            auth = None if any(key.lower() == "authorization" for key in headers) else self._sign_in(connection)
            if server["transport"] == "sse":
                return sse_client(server["url"], headers=headers, auth=auth)
            http = await stack.enter_async_context(create_mcp_http_client(headers=headers, auth=auth))
            return streamable_http_client(server["url"], http_client=http)
        command = find_command(server["command"])
        if command is None:
            raise RuntimeError(f"{server['command']} isn't installed here")
        env = {key: value for key, value in server["env"].items() if value}
        return stdio_client(StdioServerParameters(command=command, args=server["args"], env=env))

    def _sign_in(self, connection: _Connection):
        """OAuth for servers that want people to sign in: the control panel shows a Sign in button with the
        server's sign-in page, which sends the person back to the robot with a code, swapped for a token."""
        from mcp.client.auth import OAuthClientProvider
        from mcp.shared.auth import OAuthClientMetadata

        async def show_sign_in(url: str) -> None:
            state = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("state", [""])[0]
            connection.signed_in = asyncio.get_running_loop().create_future()
            connection.sign_in_url, connection.state = url, "sign_in"
            self._signing_in[state] = connection
            self.log("info", f"({connection.server['title']} wants you to sign in: see the Tools tab)")

        async def wait_for_sign_in():
            result = await connection.signed_in
            connection.sign_in_url, connection.state = "", "connecting"
            return result

        return OAuthClientProvider(
            server_url=connection.server["url"],
            client_metadata=OAuthClientMetadata(
                client_name="robot-head",
                redirect_uris=[connection.server["callback"] or self.callback],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                token_endpoint_auth_method="none",
            ),
            storage=_Tokens(self.store, connection.server["id"]),
            redirect_handler=show_sign_in,
            callback_handler=wait_for_sign_in,
        )

    # --- What the control panel shows ---

    def status(self) -> dict[int, dict]:
        """Per server: its state, why it failed, and its tools with what you chose for each."""
        with self._lock:
            connections = dict(self._connections)
        return {
            server_id: {
                "state": connection.state,
                "error": connection.error,
                "sign_in_url": connection.sign_in_url,
                "tools": [self._describe(server_id, tool) for tool in connection.tools],
            }
            for server_id, connection in connections.items()
        }

    def _describe(self, server_id: int, tool) -> dict:
        read_only, guessed = only_looks(tool)
        choice = self._choices.get((server_id, tool.name), {})
        ask_first = choice.get("ask_first")
        return {
            "name": tool.name,
            "description": (tool.description or "").strip(),
            "read_only": read_only,
            "guessed": guessed,
            "enabled": choice.get("enabled", True),
            "ask_first": (not read_only) if ask_first is None else ask_first,
        }

    def refresh(self) -> None:
        """Read your choices and skills again after the control panel changed them."""
        self._choices = self.store.tool_choices()
        self._skills = [skill for skill in self.store.skills() if skill["enabled"]]

    # --- What the brain uses ---

    def _usable(self) -> dict[str, tuple[_Connection, object, dict]]:
        """Every tool the model may call right now, by the name it calls it."""
        with self._lock:
            connections = list(self._connections.items())
        usable = {}
        for server_id, connection in connections:
            if connection.state != "ready":
                continue
            for tool in connection.tools:
                described = self._describe(server_id, tool)
                if described["enabled"]:
                    usable[function_name(connection.server["name"], tool.name)] = (connection, tool, described)
        return usable

    def definitions(self) -> list[dict]:
        """The tools, in the form OpenAI-style chat APIs take them, with OpenRouter's web tools last."""
        if not self.enabled:
            return []
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{connection.server['title']}: {(tool.description or tool.name).strip()}"[:1000],
                    "parameters": tool.input_schema or {"type": "object", "properties": {}},
                },
            }
            for name, (connection, tool, _) in self._usable().items()
        ]
        if self._skills:
            definitions.append({
                "type": "function",
                "function": {
                    "name": READ_SKILL,
                    "description": "Read the full instructions of one of your skills before doing that kind of job.",
                    "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                },
            })
        return definitions + self.internet.definitions()

    def skills_list(self) -> str:
        """The skills' names and descriptions, for the system prompt."""
        if not self.enabled:
            return ""
        return "\n".join(f"- {skill['name']}: {skill['description']}" for skill in self._skills)

    def run(self, name: str, arguments: dict, turn: int) -> str:
        """Run a tool the model called during the person's message number ``turn``; returns what to tell the model."""
        if name == READ_SKILL:
            wanted = str(arguments.get("name", "")).strip().lower()
            skill = next((skill for skill in self._skills if skill["name"].lower() == wanted), None)
            if skill is None:
                return f"There is no skill called {wanted!r}."
            self.log("info", f"(reading the skill {skill['name']})")
            return skill["instructions"][:MAX_SKILL_CHARS]
        found = self._usable().get(name)
        if found is None:
            return f"There is no tool called {name} (any more)."
        connection, tool, described = found
        label = f"{tool.name} from {connection.server['title']}"
        if described["ask_first"]:
            key = (name, json.dumps(arguments, sort_keys=True))
            now = time.monotonic()
            self._asked = {k: v for k, v in self._asked.items() if now - v[1] < CONFIRM_WITHIN}
            asked = self._asked.get(key)
            if asked is None or asked[0] >= turn:  # not asked yet, or the person hasn't answered since
                self._asked[key] = (turn, now)
                self.log("info", f"(asking before using {label})")
                return NOT_YET
            del self._asked[key]
        self.log("info", f"(using {label})")
        try:
            result = asyncio.run_coroutine_threadsafe(
                connection.client.call_tool(tool.name, arguments, read_timeout_seconds=self.timeout), self._loop
            ).result(self.timeout + 5)
        except Exception as error:  # noqa: BLE001 - the model hears what went wrong and can tell the person
            reason = _reason(error)
            self.log("error", f"{label} failed: {reason}")
            if connection.wake is not None and connection.client is not None:
                self._loop.call_soon_threadsafe(connection.wake.set)  # the connection may be broken: build a new one
            return f"The tool failed: {reason}"
        return result_text(result)


# --- Skills ---


def parse_skill(text: str) -> tuple[str, str, str]:
    """A SKILL.md's (name, description, instructions). The name and description come from its front matter."""
    match = re.match(r"\A﻿?---\s*\n(.*?)\n---\s*\n?(.*)\Z", text, re.DOTALL)
    if not match:
        raise ValueError("That isn't a SKILL.md: it has to start with a --- block holding its name and description.")
    header, body = match.groups()
    fields = {}
    lines = header.splitlines()
    for index, line in enumerate(lines):
        key_value = re.match(r"^(name|description):\s*(.*)$", line)
        if not key_value:
            continue
        key, value = key_value.groups()
        value = value.strip()
        if value in (">", "|", ">-", "|-", ">+", "|+"):  # the value continues on the indented lines below
            following = []
            for next_line in lines[index + 1 :]:
                if next_line and not next_line[0].isspace():
                    break
                following.append(next_line.strip())
            value = " ".join(part for part in following if part)
        fields[key] = value.strip().strip("\"'")
    if not fields.get("name") or not fields.get("description"):
        raise ValueError("That SKILL.md has no name or no description in its --- block.")
    return fields["name"], fields["description"], body.strip()
