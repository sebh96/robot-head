"""Long-term memory: what it knows about each person, from conversations and from what it saw.

Everything said and seen goes into a journal. Once a conversation has gone quiet (and every
so often for what it saw), it thinks the journal over, with one LLM call per person: what is
worth keeping for weeks (facts, preferences, habits, plans), what to correct or forget, and a
one-line summary of the conversation. What it remembers about whoever is in front of it goes
into every reply. Without face recognition, everyone counts as the same person.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable

from .store import Store

KINDS = ("fact", "preference", "habit", "plan")
REFLECT_AFTER_QUIET = 60.0  # seconds after the last words before a conversation is thought over
CHECK_EVERY = 10.0
MAX_ADDITIONS = 6
RECENT_CONVERSATIONS = 5
JOURNAL_LABELS = {"heard": "they said", "said": "you said", "saw": "you saw them"}

REFLECT = """\
You keep the long-term memory of an animatronic head. Below is what recently happened with {who},
and what you already remember about them. Today is {today}.
The head goes by {robot}. People say that to get its attention, so it is never their own name.

Already remembered (id: kind: text):
{memories}

Recent journal, oldest first:
{journal}

Reply with JSON only, in this shape:
{{"name": "<their first name if they clearly said it's theirs, like \\"I'm Sam\\", and it is not known yet, else null>",
  "add": [{{"kind": "fact|preference|habit|plan", "text": "..."}}],
  "update": [{{"id": 0, "text": "..."}}],
  "forget": [0],
  "summary": "<one short sentence on what you talked about, or null if there was no conversation>"}}

Keep only what is worth knowing weeks from now: facts about their life, what they like and dislike,
habits (things that repeat across days, like playing guitar most evenings), plans.
Not small talk and not what you said yourself.
Each thing is remembered once. If news changes or contradicts something already remembered (a game
they quit, a plan that moved), rewrite that memory under "update" with its id and don't add it again;
forget what turned out to be wrong. Write dates as dates ("Sat 7 Nov"), never "Saturday" or
"tomorrow". At most {limit} additions, each a short sentence. Write about them by name or with
they/them, never he or she."""


def parse_json(text: str) -> dict:
    """The JSON object in a model's answer, even when it wrapped it in a code fence."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


def _id(value) -> int | None:
    """A memory id from the model's answer, which sometimes sends them as strings."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _day(timestamp: float) -> str:
    return time.strftime("%a %d %b", time.localtime(timestamp))


class Memory:
    def __init__(
        self,
        store: Store,
        ask_json: Callable[[str], str],
        people=None,
        enabled: bool = True,
        reflect_minutes: float = 30.0,
        in_prompt: int = 40,
        keep_days: float = 30.0,
        log: Callable[[str, str], None] | None = None,
    ) -> None:
        self.store = store
        self.ask_json = ask_json  # prompt -> the model's JSON answer
        self.people = people  # None without face recognition: then everyone is one person
        self.enabled = enabled
        self.reflect_minutes = reflect_minutes  # how long seen activities wait before being thought over
        self.in_prompt = in_prompt  # memories about the current person put into each reply
        self.keep_days = keep_days  # how long the raw journal is kept after being thought over
        self.log = log or (lambda kind, text: None)
        self._reflecting = threading.Lock()

    # --- Writing things down ---

    def current_person(self) -> int | None:
        return self.people.closest() if self.people else None

    def _label(self, person: int | None, fallback: str) -> str:
        record = self.store.person(person) if person is not None else None
        return record["label"] if record else fallback

    def note(self, kind: str, text: str) -> None:
        if self.enabled and text.strip():
            self.store.write_journal(self.current_person(), kind, text.strip(), time.time())

    def record_exchange(self, heard: str | None, reply: str) -> None:
        if heard:
            self.note("heard", heard)
        self.note("said", reply)

    def record_activity(self, description: str) -> None:
        self.note("saw", description)

    # --- Putting it into the conversation ---

    def context(self) -> str:
        """What the brain should know right now: who is there and what it remembers about them."""
        if not self.enabled:
            return ""
        lines = []
        subject = None
        if self.people is not None:
            present = self.people.present()
            if present:
                subject = present[0]
                names = [self._label(p, "someone") for p in present]
                lines.append(f"In front of you right now: {names[0]} (closest)" + "".join(f", {n}" for n in names[1:]) + ".")
                person = self.store.person(subject)
                if person and not person["name"]:
                    lines.append(
                        f"You don't know {names[0]}'s name yet. If it fits, ask. When someone tells you their name, "
                        "add the cue [name=TheirName] once."
                    )
            else:
                lines.append("You don't recognize anyone in front of you right now.")
        memories = self.store.memories(subject)
        who = self._label(subject, "the person you talk to")
        lasting = [m for m in memories if m["kind"] != "conversation"][: self.in_prompt]
        talks = [m for m in memories if m["kind"] == "conversation"][:RECENT_CONVERSATIONS]
        if lasting:
            lines.append(f"What you remember about {who}:")
            lines += [f"- ({m['kind']}) {m['text']}" for m in lasting]
        if talks:
            lines.append(f"Your last conversations with {who}:")
            lines += [f"- {_day(m['created'])}: {m['text']}" for m in reversed(talks)]
        return "\n".join(lines)

    # --- Thinking it over ---

    def run(self, stop: threading.Event) -> None:
        last_prune = 0.0
        while not stop.wait(CHECK_EVERY):
            if not self.enabled:
                continue
            now = time.time()
            try:
                if self._due(self.store.unreflected(), now):
                    self.reflect()
                if now - last_prune > 24 * 3600:
                    self.store.prune_journal(now - self.keep_days * 24 * 3600)
                    last_prune = now
            except Exception as error:  # thinking it over can wait; it must not stop the robot
                self.log("error", f"Couldn't think over what happened: {error}")

    def _due(self, entries: list[dict], now: float) -> bool:
        talk = [e for e in entries if e["kind"] != "saw"]
        if talk and now - talk[-1]["time"] >= REFLECT_AFTER_QUIET:
            return True
        seen = [e for e in entries if e["kind"] == "saw"]
        return bool(seen) and now - seen[0]["time"] >= self.reflect_minutes * 60

    def reflect(self) -> int:
        """Think over everything not yet thought over, per person. Returns how many memories changed."""
        with self._reflecting:
            entries = self.store.unreflected()
            by_person: dict[int | None, list[dict]] = {}
            for entry in entries:
                by_person.setdefault(entry["person_id"], []).append(entry)
            changes = 0
            for person, items in by_person.items():
                try:
                    changes += self._reflect_on(person, items, time.time())
                finally:
                    self.store.mark_reflected([item["id"] for item in items])  # even if it failed, so it's not retried forever
            return changes

    def _reflect_on(self, person: int | None, items: list[dict], now: float) -> int:
        known = self.store.memories(person)
        who = self._label(person, "the person")
        remembered = "\n".join(f"{m['id']}: {m['kind']}: {m['text']}" for m in known if m["kind"] != "conversation")
        journal = "\n".join(
            f"- {time.strftime('%a %d %b %H:%M', time.localtime(e['time']))}, {JOURNAL_LABELS.get(e['kind'], e['kind'])}: {e['text']}"
            for e in items
        )
        robot_names = self.store.robot_names()
        prompt = REFLECT.format(
            who=who,
            today=time.strftime("%A %d %B %Y", time.localtime(now)),  # with the year, or it guesses one
            robot=" or ".join(sorted(name.title() for name in robot_names)) or "no name",
            memories=remembered or "(nothing yet)",
            journal=journal,
            limit=MAX_ADDITIONS,
        )
        answer = parse_json(self.ask_json(prompt))
        mine = {m["id"]: m for m in known if m["source"] == "llm"}  # never overwrite what you wrote yourself
        existing = {m["text"].strip().lower() for m in known}
        changes = 0

        name = answer.get("name")
        if person is not None and isinstance(name, str) and name.strip():
            record = self.store.person(person)
            if name.strip().lower() in robot_names:
                self.log("info", f"(didn't take {name.strip()} as their name: it's one of the robot's own)")
            elif record and not record["name"]:
                self.store.rename_person(person, name.strip())
                self.log("info", f"Learned a name: {name.strip()}")
                changes += 1
        for item in (answer.get("add") or [])[:MAX_ADDITIONS]:
            text = str(item.get("text", "")).strip() if isinstance(item, dict) else ""
            if text and text.lower() not in existing:
                kind = item.get("kind") if item.get("kind") in KINDS else "fact"
                self.store.add_memory(person, kind, text, "llm", now)
                existing.add(text.lower())
                changes += 1
        for item in answer.get("update") or []:
            if isinstance(item, dict) and _id(item.get("id")) in mine and str(item.get("text", "")).strip():
                self.store.update_memory(_id(item["id"]), str(item["text"]), now)
                changes += 1
        for memory_id in answer.get("forget") or []:
            if _id(memory_id) in mine:
                self.store.delete_memory(_id(memory_id))
                changes += 1
        summary = answer.get("summary")
        if isinstance(summary, str) and summary.strip():
            self.store.add_memory(person, "conversation", summary.strip(), "llm", now)
            changes += 1
        if changes:
            self.log("info", f"(thought over the time with {who}: {changes} memories changed)")
        return changes
