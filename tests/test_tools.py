import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_control import robot  # noqa: F401  (fixture)

from robot_head import catalog
from robot_head.brain import Brain
from robot_head.control import _sites
from robot_head.store import Store
from robot_head.tools import (
    NOT_YET,
    READ_SKILL,
    READ_TOKENS,
    READS_PER_REPLY,
    WEB_READ,
    WEB_SEARCH,
    Internet,
    ToolHub,
    function_name,
    only_looks,
    parse_skill,
)

SERVER = Path(__file__).with_name("tool_server.py")


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "robot.db")
    yield store
    store.close()


def functions(hub: ToolHub) -> list[str]:
    """The names of the tools the robot runs itself (OpenRouter's web tools come after them)."""
    return [d["function"]["name"] for d in hub.definitions() if d["type"] == "function"]


def ready(hub: ToolHub, seconds: float = 30) -> dict:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = next(iter(hub.status().values()))
        if status["state"] in ("ready", "error"):
            return status
        time.sleep(0.1)
    raise AssertionError("the test server never connected")


def test_a_servers_tools_are_offered_and_used(store):
    server_id = store.add_tool_server(
        {"name": "io.example/demo", "title": "Demo", "command": sys.executable, "args": [str(SERVER)]}, 0
    )
    hub = ToolHub(store, timeout=20)
    hub.start()
    try:
        status = ready(hub)
        assert status["state"] == "ready", status["error"]
        tools = {tool["name"]: tool for tool in status["tools"]}
        assert (tools["add"]["read_only"], tools["add"]["ask_first"]) == (True, False)
        assert (tools["send_note"]["read_only"], tools["send_note"]["ask_first"]) == (False, True)
        assert functions(hub) == ["demo__add", "demo__send_note"]

        assert hub.run("demo__add", {"a": 2, "b": 3}, turn=1) == "5"
        # Doing something waits for the person's next message: asking again in the same one doesn't count.
        assert hub.run("demo__send_note", {"text": "hi"}, turn=1) == NOT_YET
        assert hub.run("demo__send_note", {"text": "hi"}, turn=1) == NOT_YET
        assert hub.run("demo__send_note", {"text": "hi"}, turn=2) == "sent: hi"
        assert hub.run("demo__send_note", {"text": "again"}, turn=3) == NOT_YET  # a different note is asked anew

        store.set_tool_choice(server_id, "add", False, None)
        store.set_tool_choice(server_id, "send_note", True, False)
        hub.refresh()
        assert functions(hub) == ["demo__send_note"]
        assert hub.run("demo__send_note", {"text": "now"}, turn=3) == "sent: now"
        hub.enabled = False
        assert hub.definitions() == []
    finally:
        hub.close()


def test_a_server_that_cannot_start_says_why(store):
    store.add_tool_server({"name": "broken", "title": "Broken", "command": "no-such-launcher-here", "args": []}, 0)
    hub = ToolHub(store)
    hub.start()
    try:
        status = ready(hub)
        assert status["state"] == "error" and "isn't installed" in status["error"]
        assert functions(hub) == []
    finally:
        hub.close()


def test_tools_that_only_look_need_no_ok():
    def tool(name, annotations=None):
        return SimpleNamespace(name=name, annotations=annotations)

    hints = SimpleNamespace(read_only_hint=False, destructive_hint=None)
    assert only_looks(tool("get_forecast")) == (True, True)
    assert only_looks(tool("datetime_now")) == (True, True)
    assert only_looks(tool("send_message")) == (False, True)
    assert only_looks(tool("get_and_delete_note")) == (False, True)
    assert only_looks(tool("get_forecast", hints)) == (False, False)  # what the server says wins


def test_tool_names_suit_the_model():
    assert function_name("io.github.someone/weather-tools", "get forecast!") == "weather-tools__get_forecast"
    assert len(function_name("x" * 80, "y")) == 64


class Toolbox:
    def __init__(self, result="14:05"):
        self.calls, self.result = [], result

    def definitions(self):
        return [{"type": "function", "function": {"name": "clock__now", "parameters": {"type": "object"}}}]

    def skills_list(self):
        return ""

    def run(self, name, arguments, turn):
        self.calls.append((name, arguments, turn))
        return self.result


def chunk(content=None, calls=None, annotations=None):
    delta = SimpleNamespace(content=content, tool_calls=calls, annotations=annotations)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def call(index, arguments, id=None, name=None):
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments))


class Stream(list):
    def close(self):
        pass


class SlowStream(Stream):
    """Stops for a moment wherever there's a None: OpenRouter searching in the middle of the reply."""

    def __iter__(self):
        for item in list.__iter__(self):
            if item is None:
                time.sleep(0.3)
            else:
                yield item


def brain_with(rounds):
    brain = Brain("key", "some/model", "none", "You are Bolt.")
    requests = []

    def create(**request):
        requests.append({**request, "messages": json.loads(json.dumps(request["messages"]))})  # as sent then
        round = rounds.pop(0)
        return round if isinstance(round, Stream) else Stream(round)

    brain._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return brain, requests


def test_the_brain_speaks_uses_a_tool_and_answers_from_it():
    brain, requests = brain_with([
        [chunk("Let me check."), chunk(calls=[call(0, '{"zone": ', "c1", "clock__now")]), chunk(calls=[call(0, '"Europe/Berlin"}')])],
        [chunk("It's five past two.")],
    ])
    brain.toolbox = toolbox = Toolbox()
    assert "You have tools." in brain.system_prompt
    assert "".join(brain.reply("What time is it?")) == "Let me check. It's five past two."
    assert toolbox.calls == [("clock__now", {"zone": "Europe/Berlin"}, 1)]
    assert "tools" in requests[0]
    assistant, tool = requests[1]["messages"][-2:]
    assert assistant["tool_calls"][0]["function"] == {"name": "clock__now", "arguments": '{"zone": "Europe/Berlin"}'}
    assert (tool["role"], tool["tool_call_id"], tool["content"]) == ("tool", "c1", "14:05")
    # The tool's part of the conversation is kept, so it can answer follow-ups about it.
    assert [m["role"] for m in brain._history] == ["user", "assistant", "tool", "assistant"]
    assert brain._history[-1]["content"] == "It's five past two."


def test_a_note_from_the_system_is_not_the_persons_answer():
    brain, _ = brain_with([[chunk(calls=[call(0, "{}", "c1", "clock__now")])], [chunk("Done.")],
                           [chunk(calls=[call(0, "{}", "c2", "clock__now")])], [chunk("Done.")]])
    brain.toolbox = toolbox = Toolbox()
    list(brain.reply("Do it."))
    list(brain.reply("(Make a remark.)", from_user=False))
    assert [turn for _, _, turn in toolbox.calls] == [1, 1]


def test_what_it_said_before_a_tool_is_spoken_while_the_tool_runs():
    heard = []

    class Watching(Toolbox):
        def run(self, name, arguments, turn):
            self.heard_then = "".join(heard)
            return super().run(name, arguments, turn)

    brain, _ = brain_with([[chunk("Let me check."), chunk(calls=[call(0, "{}", "c1", "clock__now")])], [chunk("It's two.")]])
    brain.toolbox = toolbox = Watching()
    for piece in brain.reply("What time is it?"):
        heard.append(piece)  # noqa: PERF402  (one at a time: the tool looks while the reply is still going)
    # The space ends the sentence, so the voice can say it before the tool has answered.
    assert toolbox.heard_then == "Let me check. "
    assert "".join(heard) == "Let me check. It's two."


def test_openrouter_searches_and_reads_the_web_for_it(store):
    hub = ToolHub(store)
    assert hub.definitions() == [
        {"type": WEB_SEARCH, "parameters": {"max_results": 3, "search_context_size": "low", "engine": "exa"}},
        {"type": WEB_READ, "parameters": {"max_content_tokens": READ_TOKENS, "max_uses": READS_PER_REPLY}},
    ]
    hub.internet = Internet(search=True, read=False, engine="auto", results=50)
    # auto leaves the engine to OpenRouter; results are capped.
    assert hub.definitions() == [{"type": WEB_SEARCH, "parameters": {"max_results": 20, "search_context_size": "low"}}]
    hub.enabled = False
    assert hub.definitions() == []


def test_it_speaks_its_first_sentence_while_the_web_is_searched(monkeypatch, store):
    monkeypatch.setattr("robot_head.brain.IDLE_SECONDS", 0.05)
    cited = {"type": "url_citation", "url_citation": {"url": "https://www.bbc.com/sport/f1", "title": "F1 results"}}
    brain, requests = brain_with([
        SlowStream([chunk("Let me check."), None, chunk("Norris won", annotations=[cited, cited]), chunk(" in Brazil.")]),
    ])
    brain.toolbox = ToolHub(store)
    brain.location = "Hamburg, Germany"
    thinking, sites = [], []
    brain.on_tool = lambda: thinking.append(True)
    brain.on_web = sites.append
    prompt = brain.system_prompt
    assert "You can search the internet and read web pages." in prompt
    assert time.strftime("%Y") in prompt and "you are in Hamburg, Germany" in prompt

    pieces = list(brain.reply("Who won the last race?"))
    assert pieces[:2] == ["Let me check.", " "]  # let go during the pause, so it's said while OpenRouter searches
    assert "".join(pieces) == "Let me check. Norris won in Brazil."
    assert thinking == [True]  # the head looks like it's thinking while it waits
    assert sites == [["https://www.bbc.com/sport/f1"]]
    assert [tool["type"] for tool in requests[0]["tools"]] == [WEB_SEARCH, WEB_READ]
    assert brain._history[-1]["content"] == "Let me check. Norris won in Brazil."


def test_web_settings_apply_right_away(robot, store, monkeypatch):  # noqa: F811
    keys = {"WEB_SEARCH": "false", "WEB_SEARCH_ENGINE": "parallel", "WEB_SEARCH_RESULTS": "5", "LOCATION": "Hamburg, Germany"}
    for key in keys:
        monkeypatch.setenv(key, "")  # put back afterwards: saving a setting also sets it in this process
    robot.toolbox = ToolHub(store)
    assert set(robot.update_settings(keys)["applied"]) == set(keys)
    internet = robot.toolbox.internet
    assert (internet.search, internet.read, internet.engine, internet.results) == (False, True, "parallel", 5)
    assert robot.tools()["internet"] == {"search": False, "read": True, "engine": "parallel"}
    assert robot.brain.location == "Hamburg, Germany"
    assert _sites(["https://www.bbc.com/a", "https://bbc.com/b", "https://exa.ai/x"]) == ["bbc.com", "exa.ai"]


def test_skills_are_listed_and_read_when_needed(store):
    store.save_skill("cooking-coach", "Walk someone through cooking.", "One step at a time.", "yours", 0)
    hub = ToolHub(store)
    assert hub.skills_list() == "- cooking-coach: Walk someone through cooking."
    assert functions(hub) == [READ_SKILL]
    assert hub.run(READ_SKILL, {"name": "Cooking-Coach"}, turn=1) == "One step at a time."
    brain = Brain("key", "some/model", "none", "You are Bolt.")
    brain.toolbox = hub
    assert "call read_skill" in brain.system_prompt and "- cooking-coach:" in brain.system_prompt


def test_a_skill_file_is_read_with_its_front_matter():
    text = "---\nname: day-planner\ndescription: >\n  Help sort out the day.\n  Use when overwhelmed.\nlicense: MIT\n---\n\n# Steps\nFirst this."
    assert parse_skill(text) == ("day-planner", "Help sort out the day. Use when overwhelmed.", "# Steps\nFirst this.")
    with pytest.raises(ValueError, match="---"):
        parse_skill("# Just a heading")
    assert {skill["name"] for skill in catalog.builtin_skills()} == {"cooking-coach", "day-planner", "language-buddy"}


def test_registry_entries_become_ways_to_run_them():
    server = {
        "name": "io.example/weather",
        "description": "Weather.",
        "remotes": [{
            "type": "streamable-http",
            "url": "https://{region}.weather.example/mcp",
            "variables": {"region": {"description": "eu or us", "default": "eu"}},
            "headers": [{"name": "Authorization", "value": "Bearer {key}", "variables": {"key": {"isRequired": True, "isSecret": True}}}],
        }],
        "packages": [{
            "registryType": "npm", "identifier": "@example/weather", "version": "1.2.0", "transport": {"type": "stdio"},
            "environmentVariables": [{"name": "WEATHER_KEY", "isRequired": True, "isSecret": True}],
            "packageArguments": [{"type": "named", "name": "--units", "default": "metric"}],
        }],
    }
    online, here = catalog.ways_to_run(server)
    assert [field["key"] for field in online["fields"]] == ["region", "key"]
    assert here["command"] == "npx" and here["args"] == ["-y", "@example/weather@1.2.0", "--units", "metric"]
    filled = catalog.fill(online, {"key": "secret"})
    assert filled["url"] == "https://eu.weather.example/mcp"
    assert filled["headers"] == {"Authorization": "Bearer secret"}
    with pytest.raises(ValueError, match="WEATHER_KEY"):
        catalog.fill(here, {})


def test_skill_links_lead_to_their_skill_md():
    assert catalog.skill_urls("https://github.com/someone/skills/tree/main/skills/cooking") == [
        "https://raw.githubusercontent.com/someone/skills/main/skills/cooking/SKILL.md"
    ]
    assert "https://raw.githubusercontent.com/someone/skills/main/skills/cooking/SKILL.md" in catalog.skill_urls(
        "https://skills.sh/someone/skills/cooking"
    )
    with pytest.raises(ValueError):
        catalog.skill_urls("https://example.com/nothing")


def test_keys_never_go_back_to_the_page(robot):  # noqa: F811
    server_id = robot.add_tool_server({"url": "https://mcp.example.com/s/very-secret-token/mcp", "header": "Authorization",
                                       "value": "Bearer secret-value", "title": ""})
    shown = json.dumps(robot.tools())
    assert "secret-value" not in shown and "very-secret-token" not in shown
    (server,) = robot.tools()["servers"]
    assert (server["title"], server["where"], server["keys"]) == ("mcp.example.com", "online at mcp.example.com",
                                                                  [{"name": "Authorization", "set": True}])
    robot.update_tool_server(server_id, {"keys": {"Authorization": ""}, "enabled": False})
    assert robot.store.tool_server(server_id)["headers"] == {"Authorization": "Bearer secret-value"}  # empty keeps it
    with pytest.raises(ValueError, match="https"):
        robot.add_tool_server({"url": "ftp://nope"})


def test_the_control_panel_offers_tools_and_skills(robot):  # noqa: F811
    from robot_head.web import create_app

    paths = {route.path for route in create_app(robot).routes}
    assert {"/api/tools", "/api/tools/catalog", "/api/tools/oauth/callback", "/api/skills/catalog"} <= paths
