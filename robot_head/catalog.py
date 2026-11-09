"""Where tools and skills come from: the official MCP Registry, and SKILL.md files.

The registry (registry.modelcontextprotocol.io) describes each server with a server.json: online
addresses ("remotes") and packages to run here (npm, PyPI or Docker), plus the keys and settings
each needs. ``ways_to_run`` turns that into options the control panel offers with an Add button.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import PROJECT_DIR
from .tools import find_command, parse_skill

REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0.1/servers"
SKILLS_DIR = PROJECT_DIR / "skills"  # skills that come with the robot
ANTHROPIC_SKILLS = ("anthropics", "skills", "main", "skills")  # owner, repo, branch, folder
CACHE_SECONDS = 3600
SEARCH_CACHE_SECONDS = 600
REGISTRY_TIMEOUT = 45.0  # its searches sometimes take 15 s and more
LAUNCHERS = {"npm": ("npx", "Node.js"), "pypi": ("uvx", "uv"), "oci": ("docker", "Docker")}
PLACEHOLDER = re.compile(r"\{([^{}]+)\}")
_cache: dict[str, tuple[float, object]] = {}


def fetch(url: str, timeout: float = 15.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "robot-head", "Accept": "application/json, text/plain"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


# --- MCP servers ---


def search_servers(query: str, cursor: str = "") -> dict:
    """One page of servers from the registry that match ``query``, newest version of each.
    The registry can take many seconds to answer, so answers are kept for a while."""
    params = {"search": query, "version": "latest", "limit": "30"}
    if cursor:
        params["cursor"] = cursor
    url = f"{REGISTRY_URL}?{urllib.parse.urlencode(params)}"
    cached = _cache.get(url)
    if cached and time.monotonic() - cached[0] < SEARCH_CACHE_SECONDS:
        data = cached[1]
    else:
        data = json.loads(fetch(url, timeout=REGISTRY_TIMEOUT))
        _cache[url] = (time.monotonic(), data)
    servers = [entry(item.get("server", item)) for item in data.get("servers", [])]
    return {"servers": [server for server in servers if server["options"]], "next": data.get("metadata", {}).get("nextCursor", "")}


def entry(server: dict) -> dict:
    """A registry server.json, as the control panel shows it."""
    name = server.get("name", "")
    return {
        "name": name,
        "title": server.get("title") or name.rsplit("/", 1)[-1],
        "description": (server.get("description") or "").strip(),
        "version": server.get("version", ""),
        "website": server.get("websiteUrl") or (server.get("repository") or {}).get("url", ""),
        "options": ways_to_run(server),
    }


def ways_to_run(server: dict) -> list[dict]:
    """How this server can be used, best first: online needs nothing installed, a package needs its launcher."""
    options = []
    for remote in server.get("remotes", []):
        if remote.get("type") not in ("streamable-http", "sse"):
            continue
        fields: list[dict] = []
        url = _template(remote.get("url", ""), remote.get("variables", {}), fields)
        headers = {
            header["name"]: _input(header, header["name"], fields) for header in remote.get("headers", []) if header.get("name")
        }
        options.append({
            "kind": "online", "label": "Online", "url": url, "transport": remote["type"], "headers": headers,
            "command": "", "args": [], "env": {}, "fields": fields, "available": True, "needs": "",
        })
    for package in server.get("packages", []):
        kind = package.get("registryType") or package.get("registry_type")
        if kind not in LAUNCHERS or (package.get("transport") or {}).get("type", "stdio") != "stdio":
            continue
        launcher, needs = LAUNCHERS[kind]
        identifier, version = package.get("identifier", ""), package.get("version", "")
        fields = []
        env = {
            variable["name"]: _input(variable, variable["name"], fields)
            for variable in package.get("environmentVariables", []) if variable.get("name")
        }
        arguments = [argument for item in package.get("packageArguments", []) for argument in _argument(item, fields)]
        if kind == "npm":
            args = ["-y", f"{identifier}@{version}" if version else identifier, *arguments]
        elif kind == "pypi":
            args = [f"{identifier}@{version}" if version else identifier, *arguments]
        else:
            image = identifier if ":" in identifier.rsplit("/", 1)[-1] or not version else f"{identifier}:{version}"
            args = ["run", "-i", "--rm", *[part for name in env for part in ("-e", name)], image, *arguments]
        options.append({
            "kind": "here", "label": f"Runs here ({needs})", "url": "", "transport": "stdio", "headers": {},
            "command": launcher, "args": args, "env": env, "fields": fields,
            "available": find_command(launcher) is not None, "needs": needs,
        })
    return options


def _input(spec: dict, key: str, fields: list[dict]) -> str:
    """A header's or variable's value: fixed, a template with fields in it, or a field of its own."""
    if spec.get("value"):
        return _template(spec["value"], spec.get("variables", {}), fields)
    _field(key, spec, fields)
    return "{" + key + "}"


def _template(text: str, variables: dict, fields: list[dict]) -> str:
    for key in PLACEHOLDER.findall(text):
        _field(key, variables.get(key, {}), fields)
    return text


def _field(key: str, spec: dict, fields: list[dict]) -> None:
    if any(field["key"] == key for field in fields):
        return
    fields.append({
        "key": key,
        "description": (spec.get("description") or "").strip(),
        "required": bool(spec.get("isRequired", spec.get("is_required"))),
        "secret": bool(spec.get("isSecret", spec.get("is_secret"))),
        "default": spec.get("default", "") or "",
        "choices": spec.get("choices") or [],
    })


def _argument(item: dict, fields: list[dict]) -> list[str]:
    name = item.get("name", "")
    value = item.get("value") or item.get("default")
    if value is None:
        if not item.get("isRequired", item.get("is_required")):
            return []
        key = name.lstrip("-") or item.get("valueHint") or "argument"
        _field(key, item, fields)
        value = "{" + key + "}"
    value = _template(str(value), item.get("variables", {}), fields)
    return [name, value] if item.get("type") == "named" and name else [value]


def fill(option: dict, values: dict[str, str]) -> dict:
    """An option with the person's answers put in, ready to be saved as a server."""
    missing = [field["key"] for field in option.get("fields", []) if field["required"] and not values.get(field["key"], field["default"])]
    if missing:
        raise ValueError("Needs " + ", ".join(missing) + ".")

    def put(text: str) -> str:
        return PLACEHOLDER.sub(lambda match: str(values.get(match.group(1)) or _default(option, match.group(1))), text)

    return {
        "url": put(option.get("url", "")),
        "transport": option.get("transport", "streamable-http"),
        "headers": {key: put(value) for key, value in option.get("headers", {}).items()},
        "command": option.get("command", ""),
        "args": [put(arg) for arg in option.get("args", [])],
        "env": {key: put(value) for key, value in option.get("env", {}).items()},
    }


def _default(option: dict, key: str) -> str:
    return next((str(field["default"]) for field in option.get("fields", []) if field["key"] == key), "")


# --- Skills ---


def builtin_skills() -> list[dict]:
    """The skills in this repository's skills folder, written for a robot that talks."""
    skills = []
    for path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        name, description, _ = parse_skill(path.read_text(encoding="utf-8"))
        skills.append({"name": name, "description": description, "source": f"builtin:{path.parent.name}"})
    return skills


def anthropic_skills() -> list[dict]:
    """Anthropic's example skills (mostly about documents and code), read from GitHub, kept for an hour."""
    cached = _cache.get("anthropic")
    if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
        return cached[1]
    owner, repo, branch, folder = ANTHROPIC_SKILLS
    listing = json.loads(fetch(f"https://api.github.com/repos/{owner}/{repo}/contents/{folder}?ref={branch}"))
    urls = [
        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{folder}/{item['name']}/SKILL.md"
        for item in listing if item.get("type") == "dir"
    ]

    def read(url: str) -> dict | None:
        try:
            name, description, _ = parse_skill(fetch(url).decode("utf-8"))
        except (OSError, ValueError):
            return None
        return {"name": name, "description": description, "source": url}

    with ThreadPoolExecutor(max_workers=8) as pool:  # one download per skill: all at once, not one by one
        skills = [skill for skill in pool.map(read, urls) if skill]
    _cache["anthropic"] = (time.monotonic(), skills)
    return skills


def load_skill(source: str) -> tuple[str, str, str]:
    """(name, description, instructions) of a skill: one that comes with the robot, or one behind a link."""
    if source.startswith("builtin:"):
        path = SKILLS_DIR / Path(source.removeprefix("builtin:")).name / "SKILL.md"
        return parse_skill(path.read_text(encoding="utf-8"))
    errors = []
    for url in skill_urls(source.strip()):
        try:
            return parse_skill(fetch(url).decode("utf-8"))
        except (OSError, ValueError, UnicodeDecodeError) as error:
            errors.append(f"{url}: {error}")
    raise ValueError("Couldn't find a SKILL.md at that link." + (f" Tried {len(errors)} places." if errors else ""))


def skill_urls(link: str) -> list[str]:
    """Where the SKILL.md behind a link could be: a skills.sh page, a GitHub folder or file, or the file itself."""
    parsed = urllib.parse.urlparse(link)
    parts = [part for part in parsed.path.split("/") if part]
    host = parsed.netloc.lower().removeprefix("www.")
    if host == "raw.githubusercontent.com" or link.endswith("SKILL.md") and host != "github.com":
        return [link]
    if host == "github.com" and len(parts) >= 2:
        owner, repo = parts[:2]
        if len(parts) >= 4 and parts[2] in ("tree", "blob"):
            branch, path = parts[3], "/".join(parts[4:])
            path = path if path.endswith("SKILL.md") else f"{path}/SKILL.md".lstrip("/")
            return [f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"]
        return [f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/SKILL.md" for branch in ("main", "master")]
    if host == "skills.sh" and len(parts) >= 3:
        owner, repo, skill = parts[:3]
        places = [f"skills/{skill}", skill, f".claude/skills/{skill}", f"plugins/{skill}", ""]
        return [
            f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{place}/SKILL.md".replace("//SKILL", "/SKILL")
            for branch in ("main", "master") for place in places
        ]
    raise ValueError("Paste a skills.sh page, a GitHub folder with a SKILL.md in it, or a link to a SKILL.md.")
