"""Everything the control panel can see and change in the running robot."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass

from dotenv import dotenv_values

from .config import DEFAULT_MODEL, PROJECT_DIR, Settings
from .cues import GAZES, GESTURES
from .devices import ENV_FILE, audio_devices, cameras, save_settings
from .expressions import RANGES, Expression, clamped, from_settings, valid_name
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
                help="sim only shows the head here in the control panel; pca9685 also drives real servos on a Raspberry Pi."),
    SettingSpec("WEB_PORT", "Control panel port", "Control panel", "number", live=False, default="8765"),
    SettingSpec("WEB_HOST", "Control panel address", "Control panel", live=False, default="127.0.0.1",
                help="0.0.0.0 opens it to your whole network, API key included. Only do that on a network you trust."),
)
SPECS = {spec.key: spec for spec in SETTINGS}


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


class Controller:
    def __init__(
        self, settings: Settings, store: Store, brain, voice, reflexes, conversation, engine, camera=None, monitor=None
    ) -> None:
        self.store = store
        self.brain = brain
        self.voice = voice
        self.reflexes = reflexes
        self.conversation = conversation
        self.engine = engine
        self.camera = camera
        self.monitor = monitor
        self.save_recordings = settings.save_recordings
        self.restart_needed: set[str] = set()
        self._default_voice = settings.tts_voice
        self._default_speed = settings.tts_speed
        self._events: deque[dict] = deque(maxlen=300)
        self._event_id = 0
        self._lock = threading.Lock()
        self._models: list[str] | None = None
        conversation.log = self.log
        conversation.on_new_expression = self.learn_expression

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
