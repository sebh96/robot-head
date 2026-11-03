"""The robot's database (robot.db): personalities, expressions, and which personality is active."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path

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
"""


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


def _expression_row(expression: Expression) -> tuple:
    return tuple(asdict(expression)[key] for key in ("name", "lids", "tilt", "roll", "look_x", "look_y", "source"))
