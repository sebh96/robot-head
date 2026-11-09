import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_control import robot  # noqa: F401  (fixture)

from robot_head import catalog
from robot_head.brain import Brain
from robot_head.store import Store
from robot_head.tools import NOT_YET, READ_SKILL, ToolHub, function_name, only_looks, parse_skill

SERVER = Path(__file__).with_name("tool_server.py")


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "robot.db")
    yield store
    store.close()


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
        assert [d["function"]["name"] for d in hub.definitions()] == ["demo__add", "demo__send_note"]

        assert hub.run("demo__add", {"a": 2, "b": 3}, turn=1) == "5"
        # Doing something waits for the person's next message: asking again in the same one doesn't count.
        assert hub.run("demo__send_note", {"text": "hi"}, turn=1) == NOT_YET
        assert hub.run("demo__send_note", {"text": "hi"}, turn=1) == NOT_YET
        assert hub.run("demo__send_note", {"text": "hi"}, turn=2) == "sent: hi"
        assert hub.run("demo__send_note", {"text": "again"}, turn=3) == NOT_YET  # a different note is asked anew

        store.set_tool_choice(server_id, "add", False, None)
        store.set_tool_choice(server_id, "send_note", True, False)
        hub.refresh()
        assert [d["function"]["name"] for d in hub.definitions()] == ["demo__send_note"]
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
        assert hub.definitions() == []
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


def chunk(content=None, calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=calls))])


def call(index, arguments, id=None, name=None):
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments))


class Stream(list):
    def close(self):
        pass


def brain_with(rounds):
    brain = Brain("key", "some/model", "none", "You are Bolt.")
    requests = []

    def create(**request):
        requests.append({**request, "messages": json.loads(json.dumps(request["messages"]))})  # as sent then
        return Stream(rounds.pop(0))

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


def test_skills_are_listed_and_read_when_needed(store):
    store.save_skill("cooking-coach", "Walk someone through cooking.", "One step at a time.", "yours", 0)
    hub = ToolHub(store)
    assert hub.skills_list() == "- cooking-coach: Walk someone through cooking."
    assert [d["function"]["name"] for d in hub.definitions()] == [READ_SKILL]
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
