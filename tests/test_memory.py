import json
import threading
import time

import numpy as np
import pytest
from test_control import robot  # noqa: F401  (fixture)
from test_conversation import FakeBrain, FakeVoice

from robot_head import people as people_module
from robot_head.conversation import Conversation
from robot_head.cues import parse_segment
from robot_head.memory import Memory, parse_json
from robot_head.motors import SimulatedMotors
from robot_head.people import People
from robot_head.reflexes import SLEEPING_LIDS, Reflexes
from robot_head.store import Personality, Store


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "robot.db")
    yield store
    store.close()


def face(seed: int, noise: float = 0.0, base: int | None = None) -> np.ndarray:
    """A made-up fingerprint: the same seed is the same person, noise is a new angle or light."""
    vector = np.random.default_rng(seed if base is None else base).normal(size=128)
    if noise:
        vector = vector + np.random.default_rng(seed).normal(size=128) * noise
    return people_module.normalized(vector)


class Answers:
    """Stands in for the LLM: hands out canned JSON and keeps the prompts it was given."""

    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(self.answers.pop(0)) if self.answers else "{}"


def test_a_new_face_must_show_up_a_few_times_before_it_counts(store):
    people = People(store, recognizer=object())
    assert people.identify(face(1), now=1) is None
    assert people.identify(face(3, base=1, noise=0.3), now=2) is None
    sebastian = people.identify(face(4, base=1, noise=0.2), now=3)
    assert sebastian is not None
    assert len(store.faces()) == 3
    assert people.identify(face(5, base=1, noise=0.25), now=4) == sebastian
    # A different face is someone else, and one glimpse of a stranger creates nobody.
    assert people.identify(face(2), now=5) is None
    assert [p["id"] for p in store.people()] == [sebastian]


def test_two_people_are_told_apart(store):
    people = People(store, recognizer=object())
    first = [people.identify(face(20 + i, base=1, noise=0.2 + i / 10), now=i) for i in range(3)][-1]
    second = [people.identify(face(30 + i, base=2, noise=0.2 + i / 10), now=10 + i) for i in range(3)][-1]
    assert first != second and None not in (first, second)
    assert people.identify(face(11, base=1, noise=0.3), now=20) == first
    assert people.identify(face(12, base=2, noise=0.3), now=21) == second
    # Merging them (the same person after all) moves the fingerprints over.
    store.merge_people(second, first)
    people.reload()
    assert people.identify(face(13, base=2, noise=0.3), now=22) == first


def test_merging_and_forgetting_people(store):
    a, b = store.add_person(100), store.add_person(200)
    store.add_memory(b, "habit", "Plays guitar in the evening.", "llm", 200)
    store.rename_person(a, "Sebastian")
    store.merge_people(b, a)
    (person,) = store.people()
    assert (person["label"], person["memories"], person["first_seen"], person["last_seen"]) == ("Sebastian", 1, 100, 200)
    store.delete_person(a)
    assert store.people() == [] and store.memories(a) == []


def test_reflection_keeps_what_is_worth_remembering(store):
    person = store.add_person(0)
    old = store.add_memory(person, "preference", "Likes Valorant.", "llm", 0)
    mine = store.add_memory(person, "fact", "Works night shifts.", "user", 0)
    for kind, text in [("heard", "I'm Sebastian, I quit Valorant, CS2 now."), ("said", "Bro, downgrade.")]:
        store.write_journal(person, kind, text, time.time())
    answers = Answers({
        "name": "Sebastian",
        "add": [{"kind": "plan", "text": "Has a LAN party on Saturday."}, {"kind": "weird", "text": "Likes Valorant."}],
        "update": [{"id": old, "text": "Plays CS2 now, quit Valorant."}, {"id": mine, "text": "Works days."}],
        "forget": [mine],
        "summary": "Talked about switching from Valorant to CS2.",
    })
    memory = Memory(store, answers)
    memory.reflect()
    assert "I'm Sebastian" in answers.prompts[0] and "Likes Valorant." in answers.prompts[0]
    texts = {m["text"]: (m["kind"], m["source"]) for m in store.memories(person)}
    assert texts == {
        "Plays CS2 now, quit Valorant.": ("preference", "llm"),
        "Works night shifts.": ("fact", "user"),  # what you wrote yourself is never changed by it
        "Has a LAN party on Saturday.": ("plan", "llm"),
        "Talked about switching from Valorant to CS2.": ("conversation", "llm"),
    }
    assert store.person(person)["name"] == "Sebastian"
    assert store.unreflected() == []
    memory.reflect()
    assert len(answers.prompts) == 1  # nothing new, nothing to think over


def test_a_failed_reflection_is_not_retried_forever(store):
    store.write_journal(None, "heard", "Hello.", 0)

    def broken(prompt):
        raise RuntimeError("no internet")

    with pytest.raises(RuntimeError):
        Memory(store, broken).reflect()
    assert store.unreflected() == []


def test_a_conversation_is_thought_over_once_it_has_gone_quiet(store):
    memory = Memory(store, Answers(), reflect_minutes=30)
    store.write_journal(None, "heard", "Hi.", 1000)
    assert not memory._due(store.unreflected(), 1030)
    assert memory._due(store.unreflected(), 1061)
    store.mark_reflected([e["id"] for e in store.unreflected()])
    store.write_journal(None, "saw", "playing a game", 1000)
    assert not memory._due(store.unreflected(), 1000 + 29 * 60)
    assert memory._due(store.unreflected(), 1000 + 30 * 60)


def test_the_brain_hears_what_it_remembers_about_whoever_is_there(store):
    class Present:
        def __init__(self, ids):
            self.ids = ids

        def present(self):
            return self.ids

        def closest(self):
            return self.ids[0] if self.ids else None

        def label(self, person):
            return store.person(person)["label"]

    sebastian, stranger = store.add_person(0), store.add_person(0)
    store.rename_person(sebastian, "Sebastian")
    store.add_memory(sebastian, "habit", "Plays guitar most evenings.", "llm", 0)
    store.add_memory(sebastian, "conversation", "Talked about CS2.", "llm", 0)
    store.add_memory(stranger, "fact", "Has a cat.", "llm", 0)
    memory = Memory(store, Answers(), Present([sebastian, stranger]))
    context = memory.context()
    assert "In front of you right now: Sebastian (closest), Person 2." in context
    assert "Plays guitar most evenings." in context and "Talked about CS2." in context
    assert "Has a cat." not in context and "[name=" not in context
    memory.people = Present([stranger])
    assert "[name=TheirName]" in memory.context() and "Has a cat." in memory.context()
    memory.record_exchange("I got a dog too.", "Cool story.")
    assert [(e["person_id"], e["kind"]) for e in store.unreflected()] == [(stranger, "heard"), (stranger, "said")]
    memory.enabled = False
    assert memory.context() == ""


def test_the_robots_own_names_are_never_taken_for_a_persons(store, robot):  # noqa: F811
    store.seed([Personality("Archie", "You are Archie.", "Archie"), Personality("Daniel", "You are Daniel.", "Daniel")])
    person = store.add_person(0)
    store.write_journal(person, "heard", "Daniel, what do you know about Daniel Rif?", time.time())
    answers = Answers({"name": "Daniel", "add": [], "summary": "They asked about Daniel Rif."})
    Memory(store, answers).reflect()
    assert "The head goes by Archie or Bolt or Daniel." in answers.prompts[0]  # every personality, not just this one
    assert store.person(person)["name"] == ""  # saying the robot's name doesn't make it theirs
    robot.learn_name("Bolt")  # the [name=...] cue
    assert any("didn't take Bolt" in event["text"] for event in robot.state()["events"])


def test_the_model_learns_names_with_a_cue():
    segment = parse_segment("[name=Sebastian] [smug] Sebastian, huh. Mid name.")
    assert segment.learned_name == "Sebastian"
    assert "name" not in segment.cues and segment.cues == ["smug"]
    assert parse_json('```json\n{"name": null}\n```') == {"name": None}


def test_pausing_puts_it_to_sleep_and_drops_what_it_is_told(robot):  # noqa: F811
    robot.engine.stop_playback = lambda interrupted: setattr(robot.engine, "stopped", interrupted)
    brain, voice = FakeBrain(["Yo."]), FakeVoice()
    reflexes = Reflexes(SimulatedMotors())
    conversation = Conversation(brain, voice, reflexes)
    robot.conversation = conversation
    robot.set_paused(True)
    assert robot.engine.stopped and robot.state()["paused"]
    conversation.suggest("(recommend something)")
    conversation.inbox.put("hello?")
    stop = threading.Event()
    threading.Timer(0.3, stop.set).start()
    conversation.run(stop)
    assert brain.prompts == [] and voice.played == []
    assert reflexes.mode == "paused" and reflexes.step(time.monotonic(), 1 / 30)["lids"] == pytest.approx(SLEEPING_LIDS)

    robot.set_paused(False)
    conversation.inbox.put("hello?")
    stop.clear()
    threading.Timer(0.5, stop.set).start()
    conversation.run(stop)
    assert brain.prompts == ["hello?"] and voice.played == ["Yo."]


def test_memories_you_edit_stay_yours(robot):  # noqa: F811
    robot.save_memory({"text": "Likes pineapple pizza.", "kind": "preference", "person_id": None})
    (saved,) = robot.memories(None)
    assert saved["source"] == "user"
    with pytest.raises(ValueError):
        robot.save_memory({"text": "  "})
    with pytest.raises(ValueError, match="different people"):
        robot.merge_people(1, 1)
