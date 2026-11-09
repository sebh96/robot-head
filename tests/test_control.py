import base64

import numpy as np
import pytest

from robot_head import config, control
from robot_head.brain import Brain
from robot_head.control import Controller
from robot_head.conversation import Conversation
from robot_head.motors import SimulatedMotors
from robot_head.reflexes import Reflexes
from robot_head.store import Personality, Store


class FakeSpeech:
    def __init__(self, voice="am_michael", speed=1.0):
        self.voice, self.speed = voice, speed

    def voices(self):
        return ["am_michael", "am_puck"]

    def variant(self, voice, speed):
        return FakeSpeech(voice, speed)


class FakeVoice:
    def __init__(self):
        self.speech = FakeSpeech()

    def set_speech(self, speech):
        self.speech = speech


class FakeEngine:
    muted = False

    def set_barge_in(self, enabled):
        self.barge_in = enabled

    def set_silence(self, seconds):
        self.silence = seconds


@pytest.fixture
def robot(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)  # no real .env
    monkeypatch.setattr(control, "ENV_FILE", tmp_path / ".env")
    settings = config.load_settings()
    store = Store(tmp_path / "robot.db")
    store.seed([Personality("Daniel", "You are Daniel.", "Daniel"), Personality("Bolt", "You are Bolt.", "Bolt")])
    reflexes = Reflexes(SimulatedMotors())
    reflexes.set_expressions(store.expressions())
    brain = Brain("key", "some/model", "none", store.active_personality().prompt)
    voice = FakeVoice()
    conversation = Conversation(brain, voice, reflexes, wake_name="Daniel")
    controller = Controller(settings, store, brain, voice, reflexes, conversation, FakeEngine())
    yield controller
    store.close()


def test_presets_from_an_update_are_added_once(robot):
    active = robot.state()["personality"]["name"]
    robot.store.seed([Personality("Daniel", "You are Daniel.", "Daniel"), Personality("Bolt", "You are Bolt.", "Bolt")])
    assert [p["name"] for p in robot.personalities()] == ["Bolt", "Daniel"]  # nothing twice
    # A new preset is added, but it doesn't take over.
    robot.store.seed([Personality("Daniel", "You are Daniel."), Personality("Archie", "You are Archie.", "Archie")])
    assert [p["name"] for p in robot.personalities()] == ["Archie", "Bolt", "Daniel"]
    assert robot.state()["personality"]["name"] == active
    # One you deleted stays deleted.
    archie = next(p for p in robot.personalities() if p["name"] == "Archie")
    robot.delete_personality(archie["id"])
    robot.store.seed([Personality("Archie", "You are Archie.", "Archie")])
    assert [p["name"] for p in robot.personalities()] == ["Bolt", "Daniel"]
    assert "smug" in [e["name"] for e in robot.expressions()]


def test_a_new_robot_starts_as_the_character_in_personality_md():
    presets = control.starter_personalities()
    assert [p.name for p in presets] == ["Archie", "Bolt", "Daniel"]  # personality.md first, then personalities/
    assert all(p.wake_name == p.name and p.prompt.startswith(f"You are {p.name},") for p in presets)


def test_switching_personality_changes_prompt_and_name(robot):
    bolt = next(p for p in robot.personalities() if p["name"] == "Bolt")
    robot.activate(bolt["id"])
    assert robot.brain.system_prompt.startswith("You are Bolt.")
    assert robot.conversation.wake_name == "Bolt"
    assert robot.state()["personality"]["name"] == "Bolt"
    with pytest.raises(ValueError, match="Switch to another"):
        robot.delete_personality(bolt["id"])


def test_editing_the_active_personality_applies_right_away(robot):
    daniel = next(p for p in robot.personalities() if p["active"])
    robot.save_personality({**daniel, "prompt": "You are Daniel, but nicer."}, daniel["id"])
    assert robot.brain.system_prompt.startswith("You are Daniel, but nicer.")


def test_an_invented_expression_is_kept_and_usable(robot):
    robot.learn_expression("disgusted", {"lids": "0.3", "roll": "-0.2"})
    saved = {e["name"]: e for e in robot.expressions()}["disgusted"]
    assert (saved["lids"], saved["roll"], saved["source"]) == (0.3, -0.2, "llm")
    assert "[disgusted]" in robot.brain.system_prompt
    robot.reflexes.cue("disgusted")
    assert robot.reflexes.expression == "disgusted"


def test_the_llm_cannot_overwrite_your_expressions(robot):
    robot.save_expression("grumpy", {"lids": 0.4})
    robot.learn_expression("grumpy", {"lids": "1.0"})
    robot.learn_expression("smug", {"lids": "1.0"})
    faces = {e["name"]: e for e in robot.expressions()}
    assert (faces["grumpy"]["lids"], faces["grumpy"]["source"]) == (0.4, "user")
    assert (faces["smug"]["lids"], faces["smug"]["source"]) == (0.5, "preset")


def test_settings_apply_live_or_wait_for_a_restart(robot):
    result = robot.update_settings({"FOLLOW_UP_SECONDS": "5", "BARGE_IN": "false", "TTS_ENGINE": "piper", "SILENCE_SECONDS": "soon"})
    assert result == {
        "applied": ["FOLLOW_UP_SECONDS", "BARGE_IN"],
        "restart": ["TTS_ENGINE"],
        "errors": {"SILENCE_SECONDS": "Must be a number."},
    }
    assert robot.conversation.follow_up_seconds == 5.0
    assert robot.engine.barge_in is False
    saved = {s["key"]: s["value"] for s in robot.settings()}
    assert (saved["FOLLOW_UP_SECONDS"], saved["TTS_ENGINE"], saved["SILENCE_SECONDS"]) == ("5", "piper", "")


def test_the_api_key_is_never_sent_back(robot):
    robot.update_settings({"OPENROUTER_API_KEY": "sk-or-secret"})
    key = next(s for s in robot.settings() if s["key"] == "OPENROUTER_API_KEY")
    assert (key["value"], key["is_set"]) == ("", True)


def test_it_starts_without_an_api_key():
    brain = Brain("", "some/model", "none", "You are Bolt.")  # the key gets pasted in the control panel later
    brain.set_api_key("sk-or-later")


def test_its_voice_reaches_pages_set_to_play_here(robot):
    listener = robot.sound.subscribe()
    robot.engine.on_play(np.full(4800, 0.5, dtype=np.float32))  # a tenth of a second at 48 kHz
    event = listener.get_nowait()
    pcm = np.frombuffer(base64.b64decode(event["pcm"]), dtype="<i2")
    assert event["rate"] == 24000 and len(pcm) == 2400 and abs(int(pcm[1200]) - 16383) < 300
    robot.engine.on_stop()  # someone talked over it
    assert listener.get_nowait() == {"stop": True}
    robot.sound.unsubscribe(listener)
    robot.engine.on_play(np.ones(480, dtype=np.float32))
    assert listener.empty()
