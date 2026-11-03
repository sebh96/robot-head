"""Turn-taking: when to answer, when to butt in unprompted, and stopping when interrupted."""

from __future__ import annotations

import difflib
import queue
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future

import openai

from .brain import Brain
from .cues import Segment, SentenceStream
from .reflexes import Reflexes
from .voice import Voice

REMARK_PROMPT = "(Nobody has said anything for a while. Make one short unprompted remark about what you see them doing.)"
ECHO_SIMILARITY = 0.6
QUIET_BEFORE_SUGGESTING = 10.0  # seconds nobody talked before it offers an unprompted recommendation


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def mentions_name(text: str, name: str) -> bool:
    """Whether the name was said. Speech recognition bends names a little ("Daniels"), so match loosely."""
    words, size = _words(text), len(_words(name))
    target = "".join(_words(name))
    candidates = ("".join(words[i : i + size]) for i in range(len(words) - size + 1))
    return any(difflib.SequenceMatcher(None, candidate, target).ratio() >= 0.8 for candidate in candidates)


def is_echo(heard: str, spoken: list[str]) -> bool:
    """Whether what was heard is just the head's own words coming back through the microphone."""
    heard_text = " ".join(_words(heard))
    for sentence in spoken:
        said = " ".join(_words(sentence))
        if len(heard_text) >= 8 and heard_text in said:
            return True
        if difflib.SequenceMatcher(None, heard_text, said).ratio() >= ECHO_SIMILARITY:
            return True
    return False


class Conversation:
    def __init__(
        self,
        brain: Brain,
        voice: Voice,
        reflexes: Reflexes,
        camera=None,
        wake_name: str = "",
        follow_up_seconds: float = 20.0,
        remark_after_seconds: float = 0.0,
        send_frames: bool = True,
        log: Callable[[str, str], None] | None = None,
        on_new_expression: Callable[[str, dict[str, str]], None] | None = None,
    ) -> None:
        self.brain = brain
        self.voice = voice
        self.reflexes = reflexes
        self.camera = camera
        self.wake_name = wake_name
        self.follow_up_seconds = follow_up_seconds
        self.remark_after_seconds = remark_after_seconds
        self.send_frames = send_frames
        self.inbox: queue.Queue[str] = queue.Queue()  # typed messages, from the terminal or the control panel
        self.log = log or (lambda kind, text: None)  # also shows lines in the control panel
        self.on_new_expression = on_new_expression  # called when the LLM invents an expression
        self.on_exchange: Callable[[str | None, str], None] | None = None  # (what they said or None, the reply)
        self.on_name: Callable[[str], None] | None = None  # someone told it their name
        self.paused = False
        self._engaged_until = 0.0
        self._typing_only = False
        self._spoken: list[str] = []
        self._suggestion: str | None = None  # an unprompted prompt waiting for a quiet moment
        self._last_talk = time.monotonic()  # when someone last spoke or it last answered

    def suggest(self, prompt: str) -> None:
        """Say something unprompted once nobody has talked for a moment; a newer one replaces an older one."""
        if not self.paused:
            self._suggestion = prompt

    def pause(self, paused: bool) -> None:
        """While paused it sleeps: it hears nothing, answers nothing and says nothing unprompted."""
        self.paused = paused
        self._suggestion = None

    def run(self, stop: threading.Event, listen: Callable[[float], str | None] | None = None) -> None:
        """``listen(timeout)`` returns what was said, "" for sounds without words, or None if nobody spoke.
        Without ``listen`` it only answers typed messages."""
        self._typing_only = listen is None
        quiet_since = time.monotonic()
        ignored_remarks = 0
        while not stop.is_set():
            if self.paused:
                self.reflexes.set_mode("paused")
                while (typed := self._typed(0.0)) is not None:
                    self._note("ignored", f"(paused, not answering: {typed})")
                if listen:
                    listen(0.5)  # keeps the microphone drained; nothing it hears is used
                else:
                    stop.wait(0.5)
                quiet_since = self._last_talk = time.monotonic()
                continue
            self.reflexes.set_mode("listening" if self._engaged() else "idle")
            typed = self._typed(timeout=0.0 if listen else 1.0)
            heard = typed if typed is not None or listen is None else listen(0.5)
            if self.paused:
                continue  # paused while it was listening
            if heard is None:
                if self._suggestion and time.monotonic() - self._last_talk >= QUIET_BEFORE_SUGGESTING:
                    prompt, self._suggestion = self._suggestion, None
                    self._note("info", "(recommending something)")
                    self._answer(prompt, from_user=False)
                    continue
                # Being ignored makes it wait twice as long before the next remark.
                delay = self.remark_after_seconds * 2**ignored_remarks
                if delay and time.monotonic() - quiet_since >= delay and self._sees_someone():
                    self._answer(REMARK_PROMPT, from_user=False)
                    quiet_since = time.monotonic()
                    ignored_remarks += 1
                continue
            if not heard:
                continue
            if typed is None:  # typed messages are always meant for it, and never an echo
                if is_echo(heard, self._spoken):
                    self._note("ignored", f"(my own voice came back: {heard})")
                    continue
                if not self._engaged() and not mentions_name(heard, self.wake_name):
                    self._note("ignored", f"(not talking to me: {heard})")
                    continue
            self._note("you", f"you: {heard}")
            quiet_since = self._last_talk = time.monotonic()
            ignored_remarks = 0
            self._answer(heard)

    def _typed(self, timeout: float) -> str | None:
        try:
            return self.inbox.get(timeout=timeout) if timeout else self.inbox.get_nowait()
        except queue.Empty:
            return None

    def _note(self, kind: str, text: str) -> None:
        print(text)
        self.log(kind, text)

    def _engaged(self) -> bool:
        """Whether it listens without hearing its name: no name is set, or a conversation is going on."""
        return self._typing_only or not self.wake_name or time.monotonic() < self._engaged_until

    def _sees_someone(self) -> bool:
        return self.camera is not None and self.camera.face() is not None

    def _answer(self, text: str, from_user: bool = True) -> None:
        self.reflexes.set_mode("thinking")
        image = self.camera.snapshot_jpeg() if self.camera and self.send_frames else None
        try:
            self._speak_reply(text, image)
        except openai.AuthenticationError:
            self._note("error", "OpenRouter rejected the API key; fix OPENROUTER_API_KEY in the control panel or .env.")
            self.reflexes.cue("sad")
        except openai.APIError as error:
            self._note("error", f"OpenRouter error: {error}")
            self.reflexes.cue("sad")
        if self.on_exchange and self._spoken:
            self.on_exchange(text if from_user else None, " ".join(self._spoken))
        self._last_talk = time.monotonic()
        self._engaged_until = self._last_talk + self.follow_up_seconds

    def _speak_reply(self, text: str, image: bytes | None) -> None:
        """Speak the reply sentence by sentence while the rest streams in; stop if interrupted.

        A helper thread reads the reply and starts making each sentence's audio as soon as
        its text is complete, so the next sentence is ready when the current one ends.
        """
        self._spoken = []
        ready: queue.Queue[tuple[Segment, Future | None] | None] = queue.Queue()
        stop_reading = threading.Event()
        failures: list[Exception] = []

        def read_reply() -> None:
            sentences = SentenceStream()
            reply = self.brain.reply(text, image)
            try:
                for chunk in reply:
                    for segment in sentences.feed(chunk):
                        ready.put((segment, self._prepare(segment)))
                    if stop_reading.is_set():
                        return
                for segment in sentences.flush():
                    ready.put((segment, self._prepare(segment)))
            except Exception as error:  # handed to the main thread, which reports it
                failures.append(error)
            finally:
                reply.close()
                ready.put(None)

        threading.Thread(target=read_reply, daemon=True).start()
        try:
            while (item := ready.get()) is not None:
                if not self._perform(*item):
                    return
        finally:
            stop_reading.set()
        if failures:
            raise failures[0]

    def _prepare(self, segment: Segment) -> Future | None:
        return self.voice.prepare(segment.text) if segment.text else None

    def _perform(self, segment: Segment, audio: Future | None) -> bool:
        if segment.learned_name and self.on_name:
            self.on_name(segment.learned_name)
        for name, settings in segment.new_expressions.items():
            if self.on_new_expression:
                self.on_new_expression(name, settings)
        for cue in segment.cues:
            self.reflexes.cue(cue)
        if audio is None:
            return True
        self.reflexes.set_mode("speaking")
        self._note("robot", f"robot: {segment.text}")
        self._spoken.append(segment.text)
        if self.voice.play(audio):
            return True
        self._note("info", "(interrupted)")
        return False
