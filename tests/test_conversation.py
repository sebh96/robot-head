from concurrent.futures import Future

import numpy as np
import pytest

from robot_head.audio import BLOCK_SECONDS, FRAME, FRAMES_PER_BLOCK, SpeechDetector
from robot_head.conversation import Conversation, is_echo, mentions_name
from robot_head.motors import SimulatedMotors
from robot_head.reflexes import Reflexes

BLOCK = FRAME * FRAMES_PER_BLOCK


class FakeBrain:
    def __init__(self, chunks, fail_after=None):
        self.chunks, self.fail_after, self.closed, self.prompts = chunks, fail_after, False, []

    def reply(self, text, image=None):
        self.prompts.append(text)
        try:
            for i, chunk in enumerate(self.chunks):
                if i == self.fail_after:
                    raise RuntimeError("connection dropped")
                yield chunk
        finally:
            self.closed = True


class FakeVoice:
    def __init__(self, interrupt_on=None):
        self.played, self.interrupt_on = [], interrupt_on

    def prepare(self, text):
        audio = Future()
        audio.set_result(text)
        return audio

    def play(self, audio):
        self.played.append(audio.result())
        return audio.result() != self.interrupt_on


def talk(brain, voice):
    reflexes = Reflexes(SimulatedMotors())
    cues = []
    original = reflexes.cue
    reflexes.cue = lambda name: (cues.append(name), original(name))
    Conversation(brain, voice, reflexes)._speak_reply("hi", None)
    return cues


def test_a_reply_is_spoken_in_order_with_its_cues():
    voice = FakeVoice()
    cues = talk(FakeBrain(["[smug] Bro. ", "Nice setup. [roll_eyes] Not."]), voice)
    assert voice.played == ["Bro.", "Nice setup.", "Not."]
    assert cues == ["smug", "roll_eyes"]


def test_being_interrupted_stops_the_rest_of_the_reply():
    brain, voice = FakeBrain(["One. ", "Two. ", "Three. ", "Four."]), FakeVoice(interrupt_on="Two.")
    talk(brain, voice)
    assert voice.played == ["One.", "Two."]


def test_a_failing_reply_is_reported_after_what_was_said():
    voice = FakeVoice()
    with pytest.raises(RuntimeError, match="connection dropped"):
        talk(FakeBrain(["One. ", "Two. ", "Three."], fail_after=2), voice)
    assert voice.played == ["One.", "Two."]


def test_the_name_is_recognised_even_when_slightly_misheard():
    assert mentions_name("Hey Daniel, what do you think?", "Daniel")
    assert mentions_name("yo daniels look at this", "Daniel")
    assert mentions_name("Danielle, check this out", "Daniel")
    assert not mentions_name("he's in denial about it", "Daniel")
    assert not mentions_name("I'm making a sandwich", "Daniel")


def test_own_words_coming_back_are_recognised_as_echo():
    spoken = ["Bro, your cable management is a crime.", "Touch grass."]
    assert is_echo("your cable management is a crime", spoken)
    assert is_echo("bro your cable management is a time", spoken)
    assert not is_echo("Shut up and look at my keyboard.", spoken)


def blocks(level: float, seconds: float) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.normal(0, level, BLOCK).astype(np.float32) for _ in range(round(seconds / BLOCK_SECONDS))]


def run(detector: SpeechDetector, pieces) -> tuple[list[float], list[np.ndarray]]:
    """Feed (level, seconds, voice_playing) pieces; return interrupt times and utterances."""
    interrupts, utterances, t = [], [], 0.0
    for level, seconds, voice_playing in pieces:
        for block in blocks(level, seconds):
            interrupt, utterance = detector.feed(block, voice_playing)
            if interrupt:
                interrupts.append(round(t, 2))
            if utterance is not None:
                utterances.append(utterance)
            t += BLOCK_SECONDS
    return interrupts, utterances


def test_speech_between_pauses_becomes_one_utterance():
    interrupts, utterances = run(SpeechDetector(), [(0.001, 1, False), (0.05, 1.5, False), (0.001, 1.5, False)])
    assert interrupts == []
    assert len(utterances) == 1
    assert 1.5 <= len(utterances[0]) / 48000 <= 2.7  # speech plus pre-roll plus the closing pause


def test_short_echo_bursts_while_talking_do_not_interrupt():
    detector = SpeechDetector()
    pieces = [(0.001, 0.3, True), (0.05, 0.12, True)] * 8 + [(0.001, 1.5, False)]
    assert run(detector, pieces) == ([], [])


def test_talking_over_the_head_interrupts_it():
    interrupts, utterances = run(SpeechDetector(), [(0.001, 1, True), (0.05, 2, True), (0.001, 1.5, False)])
    assert len(interrupts) == 1 and 1.2 <= interrupts[0] <= 1.4  # about 300 ms after they start
    assert len(utterances) == 1


def test_barge_in_can_be_switched_off():
    interrupts, utterances = run(SpeechDetector(barge_in=False), [(0.001, 1, True), (0.05, 2, True), (0.001, 1.5, False)])
    assert interrupts == [] and utterances == []
