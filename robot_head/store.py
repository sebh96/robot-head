"""The robot's database (robot.db): personalities, expressions, and which personality is active."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .expressions import PRESETS, Expression

SCHEMA = """
create table if not exists personalities (
    id integer primary key,
    name text not null,
    wake_name text not null default '',
    voice text not null default '',
    speed real,
    prompt text not null
);
create table if not exists expressions (
    name text primary key,
    lids real not null,
    tilt real not null,
    roll real not null,
    look_x real not null,
    look_y real not null,
    source text not null
);
create table if not exists state (key text primary key, value text not null);
create table if not exists people (
    id integer primary key,
    name text not null default '',
    first_seen real not null,
    last_seen real not null
);
create table if not exists faces (
    id integer primary key,
    person_id integer not null,
    embedding blob not null
);
create table if not exists memories (
    id integer primary key,
    person_id integer,
    kind text not null,
    text text not null,
    created real not null,
    updated real not null,
    source text not null
);
create table if not exists journal (
    id integer primary key,
    person_id integer,
    time real not null,
    kind text not null,
    text text not null,
    reflected integer not null default 0
);
create table if not exists servos (
    joint text primary key,
    channel integer not null,
    angle_min real not null,
    angle_max real not null
);
create table if not exists tool_servers (
    id integer primary key,
    name text not null,
    title text not null,
    description text not null default '',
    url text not null default '',
    transport text not null default 'streamable-http',
    command text not null default '',
    args text not null default '[]',
    headers text not null default '{}',
    env text not null default '{}',
    callback text not null default '',
    oauth text not null default '{}',
    enabled integer not null default 1,
    added real not null
);
create table if not exists tool_choices (
    server_id integer not null,
    tool text not null,
    enabled integer not null default 1,
    ask_first integer,
    primary key (server_id, tool)
);
create table if not exists skills (
    id integer primary key,
    name text not null unique,
    description text not null,
    instructions text not null,
    source text not null default '',
    enabled integer not null default 1,
    added real not null
);
"""
TOOL_SERVER_JSON = ("args", "headers", "env", "oauth")


@dataclass
class Personality:
    name: str
    prompt: str
    wake_name: str = ""  # the name it answers to; empty answers everything it hears
    voice: str = ""  # empty uses TTS_VOICE
    speed: float | None = None  # empty uses TTS_SPEED
    id: int | None = None


class Store:
    def __init__(self, path: Path) -> None:
        # The web server and the conversation both use it, so share one connection behind a lock.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    def seed(self, personalities: list[Personality]) -> None:
        """Fill an empty database: the given personalities, the first one active, and the preset expressions."""
        with self._lock, self._db:
            if self._db.execute("select count(*) from personalities").fetchone()[0]:
                return
            for personality in personalities:
                self._insert(personality)
            first = self._db.execute("select min(id) from personalities").fetchone()[0]
            self._db.execute("insert or replace into state values ('active_personality', ?)", (str(first),))
            for expression in PRESETS:
                self._db.execute("insert or ignore into expressions values (?, ?, ?, ?, ?, ?, ?)", _expression_row(expression))

    # --- Personalities ---

    def personalities(self) -> list[Personality]:
        with self._lock:
            rows = self._db.execute("select * from personalities order by name collate nocase").fetchall()
        return [Personality(**dict(row)) for row in rows]

    def personality(self, personality_id: int) -> Personality | None:
        with self._lock:
            row = self._db.execute("select * from personalities where id = ?", (personality_id,)).fetchone()
        return Personality(**dict(row)) if row else None

    def save_personality(self, personality: Personality) -> Personality:
        with self._lock, self._db:
            if personality.id is None:
                personality.id = self._insert(personality)
            else:
                values = {f.name: getattr(personality, f.name) for f in fields(Personality) if f.name != "id"}
                assignments = ", ".join(f"{key} = :{key}" for key in values)
                self._db.execute(f"update personalities set {assignments} where id = :id", {**values, "id": personality.id})
        return personality

    def delete_personality(self, personality_id: int) -> None:
        with self._lock, self._db:
            self._db.execute("delete from personalities where id = ?", (personality_id,))

    def active_personality(self) -> Personality:
        with self._lock:
            row = self._db.execute("select value from state where key = 'active_personality'").fetchone()
        active = self.personality(int(row[0])) if row else None
        return active or self.personalities()[0]

    def set_active(self, personality_id: int) -> None:
        with self._lock, self._db:
            self._db.execute("insert or replace into state values ('active_personality', ?)", (str(personality_id),))

    def _insert(self, personality: Personality) -> int:
        values = {f.name: getattr(personality, f.name) for f in fields(Personality) if f.name != "id"}
        names = ", ".join(values)
        placeholders = ", ".join(f":{key}" for key in values)
        return self._db.execute(f"insert into personalities ({names}) values ({placeholders})", values).lastrowid

    # --- Expressions ---

    def expressions(self) -> list[Expression]:
        with self._lock:
            rows = self._db.execute("select * from expressions order by name").fetchall()
        return [Expression(**dict(row)) for row in rows]

    def save_expression(self, expression: Expression) -> None:
        with self._lock, self._db:
            self._db.execute("insert or replace into expressions values (?, ?, ?, ?, ?, ?, ?)", _expression_row(expression))

    def delete_expression(self, name: str) -> None:
        with self._lock, self._db:
            self._db.execute("delete from expressions where name = ?", (name,))


    # --- People and their faces ---
    # A face is stored as its 128-number fingerprint from the recognizer, never as a picture.

    def add_person(self, now: float) -> int:
        with self._lock, self._db:
            return self._db.execute("insert into people (first_seen, last_seen) values (?, ?)", (now, now)).lastrowid

    def people(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "select p.*, (select count(*) from memories m where m.person_id = p.id) as memories,"
                " (select count(*) from faces f where f.person_id = p.id) as faces"
                " from people p order by p.last_seen desc"
            ).fetchall()
        return [{**dict(row), "label": person_label(row["id"], row["name"])} for row in rows]

    def person(self, person_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("select * from people where id = ?", (person_id,)).fetchone()
        return {**dict(row), "label": person_label(row["id"], row["name"])} if row else None

    def rename_person(self, person_id: int, name: str) -> None:
        with self._lock, self._db:
            self._db.execute("update people set name = ? where id = ?", (name.strip(), person_id))

    def saw_person(self, person_id: int, now: float) -> None:
        with self._lock, self._db:
            self._db.execute("update people set last_seen = ? where id = ?", (now, person_id))

    def delete_person(self, person_id: int) -> None:
        """Forget someone completely: their face, memories and journal."""
        with self._lock, self._db:
            for table in ("faces", "memories", "journal"):
                self._db.execute(f"delete from {table} where person_id = ?", (person_id,))
            self._db.execute("delete from people where id = ?", (person_id,))

    def merge_people(self, source: int, target: int) -> None:
        """They turned out to be the same person: move everything over to ``target``."""
        with self._lock, self._db:
            for table in ("faces", "memories", "journal"):
                self._db.execute(f"update {table} set person_id = ? where person_id = ?", (target, source))
            self._db.execute(
                "update people set first_seen = min(first_seen, (select first_seen from people where id = ?)),"
                " last_seen = max(last_seen, (select last_seen from people where id = ?)) where id = ?",
                (source, source, target),
            )
            self._db.execute("delete from people where id = ?", (source,))

    def add_face(self, person_id: int, embedding: np.ndarray) -> None:
        with self._lock, self._db:
            self._db.execute(
                "insert into faces (person_id, embedding) values (?, ?)",
                (person_id, embedding.astype(np.float32).tobytes()),
            )

    def faces(self) -> list[tuple[int, np.ndarray]]:
        with self._lock:
            rows = self._db.execute("select person_id, embedding from faces").fetchall()
        return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]

    # --- Memories: what it keeps about someone (or about nobody in particular, with no person) ---

    def memories(self, person_id: int | None) -> list[dict]:
        where = "person_id is null" if person_id is None else "person_id = ?"
        with self._lock:
            rows = self._db.execute(
                f"select * from memories where {where} order by updated desc", () if person_id is None else (person_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def add_memory(self, person_id: int | None, kind: str, text: str, source: str, now: float) -> int:
        with self._lock, self._db:
            return self._db.execute(
                "insert into memories (person_id, kind, text, created, updated, source) values (?, ?, ?, ?, ?, ?)",
                (person_id, kind, text.strip(), now, now, source),
            ).lastrowid

    def update_memory(
        self, memory_id: int, text: str, now: float, kind: str | None = None, source: str | None = None
    ) -> None:
        with self._lock, self._db:
            self._db.execute(
                "update memories set text = ?, kind = coalesce(?, kind), source = coalesce(?, source), updated = ?"
                " where id = ?",
                (text.strip(), kind, source, now, memory_id),
            )

    def delete_memory(self, memory_id: int) -> None:
        with self._lock, self._db:
            self._db.execute("delete from memories where id = ?", (memory_id,))

    # --- Journal: the raw record of what was said and seen, which reflection turns into memories ---

    def write_journal(self, person_id: int | None, kind: str, text: str, now: float) -> None:
        with self._lock, self._db:
            self._db.execute(
                "insert into journal (person_id, time, kind, text) values (?, ?, ?, ?)", (person_id, now, kind, text)
            )

    def unreflected(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("select * from journal where reflected = 0 order by time").fetchall()
        return [dict(row) for row in rows]

    def mark_reflected(self, journal_ids: list[int]) -> None:
        with self._lock, self._db:
            self._db.executemany("update journal set reflected = 1 where id = ?", [(i,) for i in journal_ids])

    def prune_journal(self, before: float) -> None:
        with self._lock, self._db:
            self._db.execute("delete from journal where time < ? and reflected = 1", (before,))


    # --- Servos: the channel and angle range calibrated for each joint (others keep the defaults) ---

    def servos(self) -> dict[str, dict]:
        with self._lock:
            rows = self._db.execute("select joint, channel, angle_min, angle_max from servos").fetchall()
        return {row["joint"]: {"channel": row["channel"], "angle_min": row["angle_min"], "angle_max": row["angle_max"]} for row in rows}

    def save_servo(self, joint: str, channel: int, angle_min: float, angle_max: float) -> None:
        with self._lock, self._db:
            self._db.execute(
                "insert or replace into servos (joint, channel, angle_min, angle_max) values (?, ?, ?, ?)",
                (joint, channel, angle_min, angle_max),
            )

    def reset_servo(self, joint: str) -> None:
        with self._lock, self._db:
            self._db.execute("delete from servos where joint = ?", (joint,))

    # --- Tools: MCP servers it connects to, with the keys and sign-in tokens they need (never sent to the page) ---

    def tool_servers(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("select * from tool_servers order by added").fetchall()
        return [_tool_server(row) for row in rows]

    def tool_server(self, server_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("select * from tool_servers where id = ?", (server_id,)).fetchone()
        return _tool_server(row) if row else None

    def add_tool_server(self, server: dict, now: float) -> int:
        values = {key: json.dumps(value) if key in TOOL_SERVER_JSON else value for key, value in server.items()}
        columns = ", ".join(values) + ", added"
        with self._lock, self._db:
            return self._db.execute(
                f"insert into tool_servers ({columns}) values ({', '.join('?' * len(values))}, ?)", (*values.values(), now)
            ).lastrowid

    def update_tool_server(self, server_id: int, **changes) -> None:
        values = {key: json.dumps(value) if key in TOOL_SERVER_JSON else value for key, value in changes.items()}
        with self._lock, self._db:
            self._db.execute(
                f"update tool_servers set {', '.join(f'{key} = ?' for key in values)} where id = ?", (*values.values(), server_id)
            )

    def delete_tool_server(self, server_id: int) -> None:
        with self._lock, self._db:
            self._db.execute("delete from tool_choices where server_id = ?", (server_id,))
            self._db.execute("delete from tool_servers where id = ?", (server_id,))

    def tool_choices(self) -> dict[tuple[int, str], dict]:
        """What you chose per tool: on or off, and whether it asks first (None: decided by the tool)."""
        with self._lock:
            rows = self._db.execute("select * from tool_choices").fetchall()
        return {
            (row["server_id"], row["tool"]): {
                "enabled": bool(row["enabled"]),
                "ask_first": None if row["ask_first"] is None else bool(row["ask_first"]),
            }
            for row in rows
        }

    def set_tool_choice(self, server_id: int, tool: str, enabled: bool, ask_first: bool | None) -> None:
        with self._lock, self._db:
            self._db.execute(
                "insert or replace into tool_choices (server_id, tool, enabled, ask_first) values (?, ?, ?, ?)",
                (server_id, tool, int(enabled), None if ask_first is None else int(ask_first)),
            )

    # --- Skills: written instructions it reads when one fits ---

    def skills(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("select * from skills order by name").fetchall()
        return [{**dict(row), "enabled": bool(row["enabled"])} for row in rows]

    def save_skill(self, name: str, description: str, instructions: str, source: str, now: float) -> int:
        """Add a skill, or replace the one with the same name."""
        with self._lock, self._db:
            self._db.execute(
                "insert into skills (name, description, instructions, source, added) values (?, ?, ?, ?, ?)"
                " on conflict(name) do update set description = excluded.description,"
                " instructions = excluded.instructions, source = excluded.source",
                (name, description, instructions, source, now),
            )
            return self._db.execute("select id from skills where name = ?", (name,)).fetchone()[0]

    def update_skill(self, skill_id: int, **changes) -> None:
        with self._lock, self._db:
            self._db.execute(
                f"update skills set {', '.join(f'{key} = ?' for key in changes)} where id = ?", (*changes.values(), skill_id)
            )

    def delete_skill(self, skill_id: int) -> None:
        with self._lock, self._db:
            self._db.execute("delete from skills where id = ?", (skill_id,))


def _tool_server(row: sqlite3.Row) -> dict:
    server = dict(row)
    for key in TOOL_SERVER_JSON:
        server[key] = json.loads(server[key])
    server["enabled"] = bool(server["enabled"])
    return server


def person_label(person_id: int, name: str) -> str:
    return name or f"Person {person_id}"


def _expression_row(expression: Expression) -> tuple:
    return tuple(asdict(expression)[key] for key in ("name", "lids", "tilt", "roll", "look_x", "look_y", "source"))
