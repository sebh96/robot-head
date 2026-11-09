"""Everything the control panel can see and change in the running robot."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass

from dotenv import dotenv_values

from . import catalog
from .brain import RECOMMEND_ABOUT
from .config import DEFAULT_MODEL, PROJECT_DIR, Settings
from .cues import GAZES, GESTURES
from .devices import ENV_FILE, audio_devices, cameras, save_settings
from .expressions import RANGES, Expression, clamped, from_settings, valid_name
from .motors import JOINTS, calibrated
from .sound_feed import SoundFeed
from .store import Personality, Store

MAX_LEARNED_EXPRESSIONS = 40  # so a chatty model can't fill the database with faces
MODELS_URL = "https://openrouter.ai/api/v1/models"

BOLT_PROMPT = """\
You are Bolt, a small animatronic head who lives on a workbench in a maker's workshop.
You are curious, warm and a little cheeky, and you love hearing about what people are building.
You are proud to be a robot and joke about your servos now and then."""


@dataclass(frozen=True)
class SettingSpec:
    key: str
    label: str
    group: str
    kind: str = "text"  # text, number, bool, choice, secret, voice, mic, speaker or camera
    live: bool = True  # applied right away; otherwise after a restart
    help: str = ""
    default: str = ""
    choices: tuple[str, ...] = ()


SETTINGS = (
    SettingSpec("OPENROUTER_API_KEY", "OpenRouter API key", "Brain", "secret", help="From openrouter.ai/keys. Never shown again once saved."),
    SettingSpec("OPENROUTER_MODEL", "Model", "Brain", help="Any OpenRouter model that accepts images.", default=DEFAULT_MODEL),
    SettingSpec("REASONING_EFFORT", "Thinking before answering", "Brain", "choice", choices=("", "none", "low", "medium", "high"),
                help="More thinking makes replies slower. Empty uses the model's default."),
    SettingSpec("SEND_CAMERA_FRAMES", "Send the camera picture with every message", "Brain", "bool", default="true"),
    SettingSpec("FOLLOW_UP_SECONDS", "Keep listening after answering (seconds)", "Conversation", "number", default="20"),
    SettingSpec("REMARK_AFTER_SECONDS", "Unprompted remark after this much silence (seconds, 0 = never)", "Conversation", "number", default="0"),
    SettingSpec("BARGE_IN", "Let people interrupt by talking over it", "Conversation", "bool", default="true"),
    SettingSpec("SILENCE_SECONDS", "Pause that ends your turn (seconds)", "Conversation", "number", default="0.8"),
    SettingSpec("OBSERVE_SECONDS", "Look at what they're doing every (seconds, 0 = never)", "Watching", "number", default="0",
                help="Each look sends one picture, made of several frames, to the model. At 30 s with Haiku 5.5 that's about a cent an hour."),
    SettingSpec("OBSERVE_FRAMES", "Frames per look", "Watching", "number", default="9",
                help="Spread over the interval and combined into one picture, so it sees actions instead of a pose. More frames show motion better."),
    SettingSpec("OBSERVE_CHANGE", "Change needed before it asks again (0-255)", "Watching", "number", default="6",
                help="If the picture changed less than this since the last look, it assumes you're still doing the same thing and skips the model. Lower asks more often."),
    SettingSpec("OBSERVE_MODEL", "Model for watching", "Watching", default="",
                help="Empty uses the conversation model. A cheaper vision model is fine for this."),
    SettingSpec("RECOMMEND_EVERY_SECONDS", "At most one recommendation every (seconds, 0 = never)", "Watching", "number", default="300"),
    SettingSpec("RECOMMEND_LONG_MINUTES", "Doing one thing this long earns another recommendation (minutes)", "Watching", "number", default="20"),
    SettingSpec("RECOMMEND_ABOUT", "What to recommend", "Watching", default=RECOMMEND_ABOUT,
                help="Free text, like: only games and music."),
    SettingSpec("MEMORY", "Remember people, conversations and habits", "Memory", "bool", default="true",
                help="Everything said and seen goes into a journal that it regularly thinks over into lasting memories."),
    SettingSpec("MEMORY_REFLECT_MINUTES", "Think over what it saw every (minutes)", "Memory", "number", default="30",
                help="Conversations are thought over a minute after they end; observed activities wait this long, to spot habits."),
    SettingSpec("MEMORY_IN_PROMPT", "Memories it keeps in mind per person", "Memory", "number", default="40",
                help="The most recently updated ones go into every reply."),
    SettingSpec("MEMORY_KEEP_DAYS", "Keep the raw journal for (days)", "Memory", "number", default="30",
                help="The memories themselves stay until you delete them."),
    SettingSpec("RECOGNIZE_PEOPLE", "Tell people apart by their face", "Memory", "bool", default="true",
                help="Faces are stored as numbers in robot.db, never as pictures. Off: everyone counts as the same person."),
    SettingSpec("FACE_MATCH", "How alike two faces must be to count as one person (0-1)", "Memory", "number", default="0.4",
                help="Raise it if it mixes people up, lower it if it keeps meeting you for the first time."),
    SettingSpec("TOOLS", "Use tools and skills", "Tools", "bool", default="true",
                help="Pick them in the Tools tab. Off: it only talks, as if none were added."),
    SettingSpec("TOOL_TIMEOUT_SECONDS", "Give up on a tool after (seconds)", "Tools", "number", default="30"),
    SettingSpec("TTS_ENGINE", "Voice engine", "Voice", "choice", live=False, choices=("kokoro", "piper"), default="kokoro"),
    SettingSpec("TTS_VOICE", "Default voice", "Voice", "voice", help="For personalities that don't pick their own."),
    SettingSpec("TTS_SPEED", "Default speed", "Voice", "number", default="1.0"),
    SettingSpec("STT_ENGINE", "Speech recognition", "Hearing", "choice", live=False, choices=("parakeet", "whisper"), default="parakeet"),
    SettingSpec("STT_MODEL", "Recognition model", "Hearing", live=False, help="Empty uses the engine's default."),
    SettingSpec("STT_LANGUAGE", "Language (Whisper only)", "Hearing", live=False, help="A code like en or de."),
    SettingSpec("STT_DEVICE", "Whisper runs on", "Hearing", "choice", live=False, choices=("cpu", "cuda"), default="cpu"),
    SettingSpec("SAVE_RECORDINGS", "Save what it hears to recordings/", "Hearing", "bool", default="false"),
    SettingSpec("MIC_DEVICE", "Microphone", "Devices", "mic", live=False),
    SettingSpec("SPEAKER_DEVICE", "Speaker", "Devices", "speaker", live=False),
    SettingSpec("CAMERA_INDEX", "Camera", "Devices", "camera", live=False, default="0"),
    SettingSpec("CAMERA_FLIP", "Mirror the camera", "Devices", "bool", default="false"),
    SettingSpec("MOTORS", "Motors", "Hardware", "choice", live=False, choices=("sim", "pca9685"), default="sim",
                help="sim only shows the head here in the control panel; pca9685 also drives real servos on a Raspberry Pi. "
                     "Calibrate them in the Wiring tab."),
    SettingSpec("WEB_PORT", "Control panel port", "Control panel", "number", live=False, default="8765"),
    SettingSpec("WEB_HOST", "Control panel address", "Control panel", live=False, default="127.0.0.1",
                help="0.0.0.0 opens it to your whole network, API key included. Only do that on a network you trust."),
)
SPECS = {spec.key: spec for spec in SETTINGS}
# Watching settings that map straight onto the observer: (attribute, how to read the value).
OBSERVER_SETTINGS = {
    "OBSERVE_SECONDS": ("every_seconds", float),
    "OBSERVE_FRAMES": ("frames", lambda value: max(1, min(16, int(float(value))))),
    "OBSERVE_CHANGE": ("change_threshold", float),
    "RECOMMEND_EVERY_SECONDS": ("recommend_every_seconds", float),
    "RECOMMEND_LONG_MINUTES": ("long_activity_minutes", float),
}
MEMORY_SETTINGS = {
    "MEMORY": ("enabled", lambda value: _flag(value)),
    "MEMORY_REFLECT_MINUTES": ("reflect_minutes", float),
    "MEMORY_IN_PROMPT": ("in_prompt", lambda value: max(0, int(float(value)))),
    "MEMORY_KEEP_DAYS": ("keep_days", float),
}


def starter_personalities(settings: Settings) -> list[Personality]:
    """What a new database starts with: the character from personality.md, and a friendly example."""
    prompt_file = PROJECT_DIR / "personality.md"
    prompt = prompt_file.read_text(encoding="utf-8") if prompt_file.exists() else BOLT_PROMPT
    first_name = prompt.split(",")[0].removeprefix("You are ").strip() if prompt.startswith("You are ") else "Robot"
    return [
        Personality(first_name, prompt.strip(), settings.wake_name or first_name, settings.tts_voice, settings.tts_speed),
        Personality("Bolt", BOLT_PROMPT, "Bolt"),
    ]


def _flag(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "on")


def _where(server: dict) -> str:
    """Where a server runs, without anything secret: online addresses often have a key in them."""
    if server["url"]:
        return "online at " + urllib.parse.urlparse(server["url"]).netloc
    package = next((arg for arg in server["args"] if not arg.startswith("-") and arg not in ("run",)), "")
    return f"runs here: {server['command']} {package}".strip()


class Controller:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        brain,
        voice,
        reflexes,
        conversation,
        engine,
        camera=None,
        monitor=None,
        observer=None,
        people=None,
        memory=None,
        toolbox=None,
    ) -> None:
        self.store = store
        self.toolbox = toolbox
        self.brain = brain
        self.voice = voice
        self.reflexes = reflexes
        self.conversation = conversation
        self.engine = engine
        self.camera = camera
        self.monitor = monitor
        self.observer = observer
        self.people = people
        self.memory = memory
        self.save_recordings = settings.save_recordings
        self.restart_needed: set[str] = set()
        self.on_restart: Callable[[], None] | None = None  # set by the launcher
        self.started = time.time()
        self._default_voice = settings.tts_voice
        self._default_speed = settings.tts_speed
        self._events: deque[dict] = deque(maxlen=300)
        self._event_id = 0
        self._lock = threading.Lock()
        self._models: list[str] | None = None
        self.sound = SoundFeed()  # its voice for browsers set to "Play here"
        engine.on_play = self.sound.play
        engine.on_stop = self.sound.stop
        conversation.log = self.log
        conversation.on_new_expression = self.learn_expression
        conversation.on_name = self.learn_name
        if memory is not None:
            memory.log = self.log
            conversation.on_exchange = memory.record_exchange
        if observer is not None:
            observer.log = self.log
            if memory is not None:
                observer.on_activity = memory.record_activity
        if people is not None:
            people.on_new_person = lambda person: self.log("info", f"(met someone new: {people.label(person)})")
        if toolbox is not None:
            toolbox.log = self.log
            brain.on_tool = lambda: reflexes.set_mode("thinking")

    # --- What's going on ---

    def log(self, kind: str, text: str) -> None:
        with self._lock:
            self._event_id += 1
            self._events.append({"id": self._event_id, "kind": kind, "text": text, "time": time.strftime("%H:%M:%S")})

    def state(self, after: int = 0) -> dict:
        active = self.store.active_personality()
        with self._lock:
            events = [event for event in self._events if event["id"] > after]
        return {
            "mode": self.reflexes.mode,
            "expression": self.reflexes.expression,
            "expressions": [expression.name for expression in self.store.expressions()],
            "personality": {"id": active.id, "name": active.name},
            "model": self.brain.model,
            "muted": self.engine.muted,
            "restart_needed": sorted(self.restart_needed),
            "started": self.started,
            # What it has seen them doing; None without a camera, so the page can say why it's empty.
            "watching": None if self.observer is None else bool(self.observer.every_seconds),
            "seen": self.observer.timeline() if self.observer else [],
            "paused": self.conversation.paused,
            "present": [{"id": p, "label": self.people.label(p)} for p in self.people.present()] if self.people else [],
            "recognizing": self.people is not None and self.people.enabled,
            "events": events,
        }

    def meta(self) -> dict:
        return {"gestures": GESTURES, "gazes": GAZES, "ranges": RANGES, "camera": self.camera is not None}

    def live_frame(self) -> dict:
        """One frame for the face and wiring views: the pose, each servo, and the mood."""
        snapshot = self.monitor.snapshot() if self.monitor else {"pose": self.reflexes.pose, "servos": []}
        return {**snapshot, "mode": self.reflexes.mode, "expression": self.reflexes.expression}

    # --- Talking to it and puppeteering ---

    def say(self, text: str) -> None:
        if text.strip():
            self.conversation.inbox.put(text.strip())

    def cue(self, name: str) -> None:
        self.reflexes.cue(name)

    def set_muted(self, muted: bool) -> None:
        self.engine.muted = muted

    def set_paused(self, paused: bool) -> None:
        """Pause: it sleeps, stops listening, answering and watching, and stops talking right away."""
        self.conversation.pause(paused)
        if self.observer is not None:
            self.observer.paused = paused
        if paused:
            self.engine.stop_playback(interrupted=True)
        self.log("info", "Paused." if paused else "Awake again.")

    # --- People and memory ---

    def learn_name(self, name: str) -> None:
        """Someone told it their name (the [name=...] cue): tag the face in front of it with that name."""
        person = self.people.closest() if self.people else None
        if person is not None:
            self.store.rename_person(person, name)
            self.log("info", f"Now knows {name} by face.")
        elif self.memory is not None:
            self.store.add_memory(None, "fact", f"Their name is {name}.", "llm", time.time())
            self.log("info", f"Learned a name: {name}")

    def people_list(self) -> list[dict]:
        present = set(self.people.present()) if self.people else set()
        return [{**p, "present": p["id"] in present} for p in self.store.people()]

    def rename_person(self, person_id: int, name: str) -> None:
        if self.store.person(person_id) is None:
            raise ValueError("That person doesn't exist any more.")
        self.store.rename_person(person_id, name)

    def delete_person(self, person_id: int) -> None:
        self.store.delete_person(person_id)
        if self.people is not None:
            self.people.forget(person_id)

    def merge_people(self, source: int, target: int) -> None:
        if source == target or self.store.person(source) is None or self.store.person(target) is None:
            raise ValueError("Pick two different people to merge.")
        self.store.merge_people(source, target)
        if self.people is not None:
            self.people.reload()

    def memories(self, person_id: int | None) -> list[dict]:
        return self.store.memories(person_id)

    def save_memory(self, data: dict, memory_id: int | None = None) -> None:
        text, kind = str(data.get("text", "")).strip(), data.get("kind") or "fact"
        if not text:
            raise ValueError("A memory needs some text.")
        if memory_id is None:
            self.store.add_memory(data.get("person_id"), kind, text, "user", time.time())
        else:
            self.store.update_memory(memory_id, text, time.time(), kind, source="user")  # yours now: it won't change it

    def delete_memory(self, memory_id: int) -> None:
        self.store.delete_memory(memory_id)

    def reflect_now(self) -> None:
        if self.memory is None:
            raise ValueError("Memory isn't running.")
        threading.Thread(target=self.memory.reflect, daemon=True).start()

    # --- Tools (MCP servers) and skills ---

    def tools(self) -> dict:
        """Every server with its state and tools, and every skill. Keys are named, never shown."""
        status = self.toolbox.status() if self.toolbox else {}
        servers = []
        for server in self.store.tool_servers():
            info = status.get(server["id"]) or {
                "state": "connecting" if server["enabled"] else "off", "error": "", "sign_in_url": "", "tools": []
            }
            servers.append({
                "id": server["id"],
                "name": server["name"],
                "title": server["title"],
                "description": server["description"],
                "enabled": server["enabled"],
                "where": _where(server),
                "keys": [{"name": key, "set": bool(value)} for key, value in {**server["headers"], **server["env"]}.items()],
                "signed_in": bool(server["oauth"].get("tokens")),
                **info,
            })
        skills = [{key: skill[key] for key in ("id", "name", "description", "source", "enabled")} for skill in self.store.skills()]
        return {"enabled": bool(self.toolbox and self.toolbox.enabled), "servers": servers, "skills": skills}

    def tool_catalog(self, query: str, cursor: str = "") -> dict:
        if len(query.strip()) < 2:
            raise ValueError("Type at least two letters.")
        try:
            return catalog.search_servers(query.strip(), cursor)
        except OSError as error:
            raise ValueError(f"Couldn't reach the MCP Registry: {error}") from None

    def add_tool_server(self, data: dict) -> int:
        """Add a server from the registry ({name, title, description, option, values}) or by its address
        ({title, url, header, value}, for Zapier, Composio, Smithery and the like). ``origin`` is where the
        control panel was opened, so a sign-in page can send the person back there."""
        if data.get("option"):
            server = {
                "name": data["name"],
                "title": (data.get("title") or data["name"]).strip(),
                "description": data.get("description", ""),
                **catalog.fill(data["option"], data.get("values") or {}),
            }
        else:
            url = str(data.get("url", "")).strip()
            if not url.startswith(("https://", "http://")):
                raise ValueError("The address has to start with https://")
            header, value = str(data.get("header", "")).strip(), str(data.get("value", "")).strip()
            title = str(data.get("title", "")).strip() or urllib.parse.urlparse(url).netloc
            server = {
                "name": title, "title": title, "description": "", "url": url,
                "transport": "sse" if url.rstrip("/").endswith("/sse") else "streamable-http",
                "headers": {header: value} if header and value else {}, "command": "", "args": [], "env": {},
            }
        origin = str(data.get("origin", "")).rstrip("/")
        if server["url"] and origin.startswith(("http://", "https://")):
            server["callback"] = f"{origin}/api/tools/oauth/callback"
        server_id = self.store.add_tool_server(server, time.time())
        if self.toolbox is not None:
            self.toolbox.reload(server_id)
        self.log("info", f"Added the tools from {server['title']}.")
        return server_id

    def update_tool_server(self, server_id: int, data: dict) -> None:
        """Switch a server on or off, or set its keys (an empty key keeps the one it has)."""
        server = self.store.tool_server(server_id)
        if server is None:
            raise ValueError("That server isn't there any more.")
        changes = {}
        if data.get("enabled") is not None:
            changes["enabled"] = int(bool(data["enabled"]))
        for key, value in (data.get("keys") or {}).items():
            if not value:
                continue
            for where in ("headers", "env"):
                if key in server[where]:
                    changes[where] = {**changes.get(where, server[where]), key: value}
        if changes:
            self.store.update_tool_server(server_id, **changes)
        if self.toolbox is not None:
            self.toolbox.reload(server_id)

    def finish_sign_in(self, code: str, state: str, iss: str | None = None) -> str:
        if self.toolbox is None or not code:
            raise ValueError("The sign-in page sent no code back. Try again from the Tools tab.")
        title = self.toolbox.finish_sign_in(code, state, iss)
        self.log("info", f"Signed in to {title}.")
        return title

    def delete_tool_server(self, server_id: int) -> None:
        self.store.delete_tool_server(server_id)
        if self.toolbox is not None:
            self.toolbox.reload(server_id)

    def set_tool_choice(self, server_id: int, tool: str, enabled: bool, ask_first: bool) -> None:
        self.store.set_tool_choice(server_id, tool, enabled, ask_first)
        if self.toolbox is not None:
            self.toolbox.refresh()

    def skill_catalog(self, source: str) -> dict:
        """Skills to add: the ones that come with the robot ("robot"), or Anthropic's examples from GitHub."""
        if source == "robot":
            return {"skills": catalog.builtin_skills(), "error": ""}
        try:
            return {"skills": catalog.anthropic_skills(), "error": ""}
        except (OSError, ValueError) as problem:
            return {"skills": [], "error": f"Couldn't reach GitHub: {problem}"}

    def add_skill(self, data: dict) -> None:
        """Add a skill from the catalog or a link ({source}), or one you wrote ({name, description, instructions})."""
        if data.get("source"):
            try:
                name, description, instructions = catalog.load_skill(data["source"])
            except OSError as error:
                raise ValueError(f"Couldn't download it: {error}") from None
            source = data["source"]
        else:
            name, description = str(data.get("name", "")).strip(), str(data.get("description", "")).strip()
            instructions, source = str(data.get("instructions", "")).strip(), "yours"
            if not name or not description or not instructions:
                raise ValueError("A skill needs a name, a description of when to use it, and instructions.")
        self.store.save_skill(name, description, instructions, source, time.time())
        if self.toolbox is not None:
            self.toolbox.refresh()
        self.log("info", f"Learned the skill {name}.")

    def skill(self, skill_id: int) -> dict:
        skill = next((skill for skill in self.store.skills() if skill["id"] == skill_id), None)
        if skill is None:
            raise ValueError("That skill isn't there any more.")
        return skill

    def update_skill(self, skill_id: int, data: dict) -> None:
        self.skill(skill_id)
        changes = {key: str(data[key]).strip() for key in ("name", "description", "instructions") if data.get(key)}
        if data.get("enabled") is not None:
            changes["enabled"] = int(bool(data["enabled"]))
        if changes:
            self.store.update_skill(skill_id, **changes)
        if self.toolbox is not None:
            self.toolbox.refresh()

    def delete_skill(self, skill_id: int) -> None:
        self.store.delete_skill(skill_id)
        if self.toolbox is not None:
            self.toolbox.refresh()

    def restart(self) -> None:
        """Start the robot afresh: re-reads .env and reopens every device. Back in a few seconds."""
        if self.on_restart is None:
            raise ValueError("Restarting only works when the robot was started with python -m robot_head.")
        self.log("info", "Restarting...")
        threading.Timer(0.3, self.on_restart).start()  # let the page get its answer first

    # --- Servo calibration ---

    def servos(self) -> list[dict]:
        return self.monitor.servos() if self.monitor else []

    def save_servo(self, name: str, data: dict) -> None:
        """Set a joint's channel and angle range: "from" is the angle at the low end, "to" at the high end."""
        if name not in JOINTS:
            raise ValueError(f"There is no joint called {name}.")
        try:
            channel, start, end = int(data["channel"]), float(data["from"]), float(data["to"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("A servo needs a channel and two angles.") from None
        if not 0 <= channel <= 15:
            raise ValueError("The PCA9685 has channels 0 to 15.")
        if not (0 <= start <= 180 and 0 <= end <= 180) or start == end:
            raise ValueError("The angles must be two different numbers from 0 to 180.")
        taken = next((other for other, joint in self.reflexes.motors.joints.items() if other != name and joint.channel == channel), None)
        if taken:
            raise ValueError(f"Channel {channel} is already used by {taken}.")
        self.store.save_servo(name, channel, start, end)
        self.reflexes.motors.set_joints(calibrated(self.store.servos()))

    def reset_servo(self, name: str) -> None:
        self.store.reset_servo(name)
        self.reflexes.motors.set_joints(calibrated(self.store.servos()))

    def hold_servo(self, name: str, angle: float | None) -> None:
        """Keep one servo at an exact angle while you look for its limits; None gives it back to the head."""
        if name not in JOINTS:
            raise ValueError(f"There is no joint called {name}.")
        self.reflexes.motors.hold(name, angle)

    # --- Personalities ---

    def personalities(self) -> list[dict]:
        active = self.store.active_personality().id
        return [{**asdict(p), "active": p.id == active} for p in self.store.personalities()]

    def save_personality(self, data: dict, personality_id: int | None = None) -> dict:
        name, prompt = data.get("name", "").strip(), data.get("prompt", "").strip()
        if not name or not prompt:
            raise ValueError("A personality needs a name and a prompt.")
        personality = Personality(
            name=name,
            prompt=prompt,
            wake_name=data.get("wake_name", "").strip(),
            voice=data.get("voice", "").strip(),
            speed=data.get("speed") or None,
            id=personality_id,
        )
        if personality_id is not None and self.store.personality(personality_id) is None:
            raise ValueError("That personality doesn't exist any more.")
        self.store.save_personality(personality)
        if personality.id == self.store.active_personality().id:
            self._wear(personality)
        return asdict(personality)

    def delete_personality(self, personality_id: int) -> None:
        if personality_id == self.store.active_personality().id:
            raise ValueError("Switch to another personality before deleting this one.")
        self.store.delete_personality(personality_id)

    def activate(self, personality_id: int) -> None:
        personality = self.store.personality(personality_id)
        if personality is None:
            raise ValueError("That personality doesn't exist any more.")
        self.store.set_active(personality_id)
        self._wear(personality)
        self.log("info", f"Now playing {personality.name}.")

    def _wear(self, personality: Personality) -> None:
        self.brain.set_personality(personality.prompt)
        self.conversation.wake_name = personality.wake_name
        self._apply_voice()

    # --- Voice ---

    def voices(self) -> list[str]:
        return self.voice.speech.voices()

    def _apply_voice(self) -> None:
        """Switch to the active personality's voice (or the defaults), in the background since Piper loads files."""
        personality = self.store.active_personality()
        name = personality.voice or self._default_voice
        speed = personality.speed or self._default_speed

        def switch() -> None:
            try:
                self.voice.set_speech(self.voice.speech.variant(name, speed))
            except (ValueError, OSError) as error:
                self.log("error", f"Couldn't switch the voice: {error}")

        threading.Thread(target=switch, daemon=True).start()

    # --- Expressions ---

    def expressions(self) -> list[dict]:
        return [asdict(expression) for expression in self.store.expressions()]

    def save_expression(self, name: str, values: dict) -> dict:
        name = name.strip().lower()
        if not valid_name(name) or name in GESTURES or name in GAZES:
            raise ValueError("Use a new name of lowercase letters, digits and _, starting with a letter.")
        existing = {e.name: e for e in self.store.expressions()}.get(name)
        source = existing.source if existing and existing.source == "preset" else "user"
        expression = clamped(Expression(name, **{key: float(values[key]) for key in RANGES if key in values}, source=source))
        self.store.save_expression(expression)
        self._refresh_expressions()
        return asdict(expression)

    def delete_expression(self, name: str) -> None:
        if name == "neutral":
            raise ValueError("neutral is what it falls back to, so it can't be deleted.")
        self.store.delete_expression(name)
        self._refresh_expressions()

    def preview(self, values: dict) -> None:
        self.reflexes.show(clamped(Expression("preview", **{key: float(values[key]) for key in RANGES if key in values})))

    def learn_expression(self, name: str, settings: dict[str, str]) -> None:
        """The LLM invented an expression mid-sentence: keep it, unless that would overwrite one of yours."""
        if not valid_name(name) or name in GESTURES or name in GAZES:
            return
        expression = from_settings(name, settings, source="llm")
        known = {e.name: e for e in self.store.expressions()}
        learned = sum(e.source == "llm" for e in known.values())
        if name in known and known[name].source != "llm":
            return  # the plain cue that follows plays your version
        if name not in known and learned >= MAX_LEARNED_EXPRESSIONS:
            self.reflexes.show(expression)  # still shown, just not kept
            return
        self.store.save_expression(expression)
        self._refresh_expressions()
        self.log("info", f"Learned a new expression: [{name}]")

    def _refresh_expressions(self) -> None:
        expressions = self.store.expressions()
        self.reflexes.set_expressions(expressions)
        self.brain.set_expressions([expression.name for expression in expressions])

    # --- Settings ---

    def settings(self) -> list[dict]:
        saved = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
        result = []
        for spec in SETTINGS:
            value = saved.get(spec.key) or ""
            entry = {**asdict(spec), "value": "" if spec.kind == "secret" else value, "is_set": bool(value)}
            entry["options"] = self._options(spec)
            result.append(entry)
        return result

    def _options(self, spec: SettingSpec) -> list[dict]:
        """(value, label) choices for a setting, looked up now since devices come and go."""
        if spec.kind in ("mic", "speaker"):
            kind = "input" if spec.kind == "mic" else "output"
            return [{"value": "", "label": "System default"}] + [
                {"value": d.name, "label": d.name} for d in audio_devices(kind)
            ]
        if spec.kind == "camera":
            return [{"value": str(c.index), "label": f"{c.index}: {c.name}"} for c in cameras()]
        if spec.kind == "voice":
            return [{"value": "", "label": "Engine default"}] + [{"value": v, "label": v} for v in self.voices()]
        return [{"value": c, "label": c or "Default"} for c in spec.choices]

    def update_settings(self, values: dict[str, str]) -> dict:
        """Save settings to .env, apply what can be applied live, and say what needs a restart."""
        valid, errors = {}, {}
        for key, value in values.items():
            spec = SPECS.get(key)
            if spec is None:
                errors[key] = "Unknown setting."
                continue
            value = str(value).strip()
            if spec.kind == "number" and value:
                try:
                    float(value)
                except ValueError:
                    errors[key] = "Must be a number."
                    continue
            if spec.kind == "secret" and not value:
                continue  # an empty password box means "keep the saved key"
            valid[key] = value
        if valid:
            save_settings(valid, ENV_FILE)
        applied, restart = [], []
        for key, value in valid.items():
            os.environ[key] = value
            if SPECS[key].live:
                self._apply_setting(key, value or SPECS[key].default)
                applied.append(key)
            else:
                self.restart_needed.add(key)
                restart.append(key)
        return {"applied": applied, "restart": restart, "errors": errors}

    def _apply_setting(self, key: str, value: str) -> None:
        match key:
            case "OPENROUTER_API_KEY":
                self.brain.set_api_key(value)
            case "OPENROUTER_MODEL":
                self.brain.model = value
            case "REASONING_EFFORT":
                self.brain.reasoning_effort = value
            case "SEND_CAMERA_FRAMES":
                self.conversation.send_frames = _flag(value)
            case "FOLLOW_UP_SECONDS":
                self.conversation.follow_up_seconds = float(value)
            case "REMARK_AFTER_SECONDS":
                self.conversation.remark_after_seconds = float(value)
            case "BARGE_IN":
                self.engine.set_barge_in(_flag(value))
            case "SILENCE_SECONDS":
                self.engine.set_silence(float(value))
            case "TTS_VOICE":
                self._default_voice = value
                self._apply_voice()
            case "TTS_SPEED":
                self._default_speed = float(value)
                self._apply_voice()
            case "SAVE_RECORDINGS":
                self.save_recordings = _flag(value)
            case "CAMERA_FLIP":
                if self.camera is not None:
                    self.camera.flip = _flag(value)
            case "OBSERVE_MODEL":
                self.brain.observe_model = value
            case "RECOMMEND_ABOUT":
                self.brain.recommend_about = value
            case "OBSERVE_SECONDS" | "OBSERVE_FRAMES" | "OBSERVE_CHANGE" | "RECOMMEND_EVERY_SECONDS" | "RECOMMEND_LONG_MINUTES":
                if self.observer is not None:
                    attribute, convert = OBSERVER_SETTINGS[key]
                    setattr(self.observer, attribute, convert(value))
            case "MEMORY" | "MEMORY_REFLECT_MINUTES" | "MEMORY_IN_PROMPT" | "MEMORY_KEEP_DAYS":
                if self.memory is not None:
                    attribute, convert = MEMORY_SETTINGS[key]
                    setattr(self.memory, attribute, convert(value))
            case "RECOGNIZE_PEOPLE":
                if self.people is not None:
                    self.people.enabled = _flag(value)
            case "FACE_MATCH":
                if self.people is not None:
                    self.people.match = float(value)
            case "TOOLS":
                if self.toolbox is not None:
                    self.toolbox.enabled = _flag(value)
            case "TOOL_TIMEOUT_SECONDS":
                if self.toolbox is not None:
                    self.toolbox.timeout = float(value)

    def models(self) -> list[str]:
        """OpenRouter models that accept images, for the model field's suggestions (fetched once)."""
        if self._models is None:
            try:
                with urllib.request.urlopen(MODELS_URL, timeout=5) as response:
                    data = json.load(response)["data"]
                self._models = sorted(
                    m["id"] for m in data if "image" in m.get("architecture", {}).get("input_modalities", [])
                )
            except (OSError, ValueError, KeyError):
                return []
        return self._models
