import threading

import numpy as np

from robot_head import conversation as conversation_module
from robot_head.brain import Brain
from robot_head.conversation import Conversation
from robot_head.motors import SimulatedMotors
from robot_head.observer import Observer, grid
from robot_head.reflexes import Reflexes
from test_conversation import FakeBrain, FakeVoice


def frames(brightness: int, count: int = 3) -> list[np.ndarray]:
    return [np.full((480, 640, 3), brightness, np.uint8)] * count


class Answers:
    """Stands in for the model: hands out canned answers and counts the calls."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def __call__(self, picture, count, previous):
        self.calls += 1
        return self.answers.pop(0)


def test_frames_are_combined_into_one_numbered_grid():
    assert grid(frames(0, 9)).shape == (720, 960, 3)  # 3 x 3
    assert grid(frames(0, 4)).shape == (720, 960, 3)  # 2 x 2
    assert grid(frames(0, 5)).shape == (480, 960, 3)  # 3 + 2, with an empty tile


def test_the_timeline_follows_what_they_do():
    answers = Answers("Typing on a laptop.", "SAME", "Eating chips while scrolling on a phone", "NOBODY")
    tips = []
    observer = Observer(None, answers, recommend_every_seconds=300, on_recommend=tips.append)
    observer.look(frames(10), True, 1000)
    observer.look(frames(10), True, 1030)  # nothing changed: no need to ask the model
    observer.look(frames(60), True, 1060)  # SAME
    observer.look(frames(120), True, 1090)  # something new
    observer.look(frames(180), False, 1400)  # NOBODY
    assert answers.calls == 4
    assert [(a["description"], a["minutes"]) for a in observer.timeline()] == [
        ("Typing on a laptop", 1),
        ("Eating chips while scrolling on a phone", 0),
    ]
    assert "Eating chips" in observer.summary(1100)
    assert len(tips) == 1  # the second new activity came too soon after the first tip


def test_a_long_activity_earns_one_more_recommendation():
    tips = []
    observer = Observer(None, Answers("Reading a book"), recommend_every_seconds=300, long_activity_minutes=20, on_recommend=tips.append)
    for now in (0, 400, 800, 1250, 1600, 2000):
        observer.look(frames(10), True, now)
    assert len(tips) == 2  # when it started, and once it passed 20 minutes


def test_the_brain_knows_what_it_has_seen():
    brain = Brain("key", "some/model", "none", "You are Bolt.")
    brain.seen = lambda: "- 14:02, for 18 min: typing on a laptop"
    assert "typing on a laptop" in brain.system_prompt
    brain.recommend_about = "only games and music"
    assert "only games and music" in brain.recommendation_prompt("- 14:02: typing")


def test_a_recommendation_is_given_once_it_is_quiet(monkeypatch):
    monkeypatch.setattr(conversation_module, "QUIET_BEFORE_SUGGESTING", 0.0)
    brain, voice = FakeBrain(["Take a break, bro."]), FakeVoice()
    conversation = Conversation(brain, voice, Reflexes(SimulatedMotors()))
    conversation.suggest("(recommend something)")
    stop, calls = threading.Event(), []

    def listen(timeout):
        calls.append(timeout)
        if len(calls) > 2:
            stop.set()

    conversation.run(stop, listen)
    assert brain.prompts == ["(recommend something)"]
    assert voice.played == ["Take a break, bro."]
