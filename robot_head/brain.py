"""The slow layer: an LLM on OpenRouter decides what to say and how to move."""

from __future__ import annotations

import base64
import time
from collections.abc import Callable, Iterator

from openai import OpenAI

from .cues import GAZES, GESTURES
from .expressions import PRESETS, RANGES

OPENROUTER_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = 2048  # spoken replies are short; the headroom is for reasoning tokens

RULES = """\
You are the mind of a physical animatronic head with a camera, a speaker and motors.
Everything you write is spoken aloud by a text-to-speech voice, so:
- Answer in one to three short spoken sentences, in the language the user speaks.
- No lists, markdown, emoji or stage directions.
- The user's words come from speech recognition and may contain mistakes; go with what they most likely meant.
- Each user message comes with a photo from the camera in your face. That is what you see right now.
  How much you comment on it is up to your character.
- Text in parentheses is a note from the system, not something the user said.
- If a reply of yours ends with "—", the user cut you off there.

You move by writing cues in square brackets. A cue is performed when the sentence it is in starts
playing, and it is never spoken. Use one or two per reply, where they add something.
Expressions: {expressions}
Gestures: {gestures}
Gaze: {gazes}
Example: [surprised] Oh, is that a cat? [tilt_left] What's its name?

When no expression fits, invent one: give it a new name plus settings, like
[disgusted lids=0.3 tilt=0.1 roll=-0.2 look=0,-0.3]. It plays right away, and from then on [disgusted] works too.
Settings: lids {lids} (shut to wide open), tilt {tilt} (chin down to chin up), roll {roll} (lean to your left
or right), look x,y each {look} (where your eyes drift). Leave out the ones that should stay neutral.
"""

SEEN = """
What the camera has shown you the person doing lately, oldest first. Use it when it's relevant:
{timeline}
"""

OBSERVE = """\
These are {count} camera frames taken a few seconds apart, numbered in reading order.
In one short sentence (under 20 words), say what the person in front of the camera is doing,
including what they are holding, using, eating or looking at. Describe the action, not their looks.
Earlier you saw: "{previous}". If that is still what they are doing, answer with exactly: SAME
If nobody is there, answer with exactly: NOBODY"""

RECOMMEND = """\
(You have been watching what they do:
{timeline}
Give them one short, specific recommendation that fits what they are doing right now: {about}.
Say it in character, without being asked.)"""
RECOMMEND_ABOUT = "something to try, play, watch, listen to, eat, fix or change"


def _cue_list(names) -> str:
    return " ".join(f"[{name}]" for name in names)


def _span(key: str) -> str:
    low, high = RANGES[key]
    return f"{low:g} to {high:g}"


class Brain:
    def __init__(
        self,
        api_key: str,
        model: str,
        reasoning_effort: str,
        personality: str,
        expressions: list[str] | None = None,
        max_turns: int = 12,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._personality = personality.strip()
        self._expressions = expressions or [expression.name for expression in PRESETS]
        self._history: list[dict] = []
        self._max_messages = max_turns * 2
        self.seen: Callable[[], str] | None = None  # the observer's timeline, when the camera watches
        self.remembered: Callable[[], str] | None = None  # who is there and what it remembers about them
        self.observe_model = ""  # a separate model for watching; empty uses the conversation's
        self.recommend_about = RECOMMEND_ABOUT  # what kind of things to recommend
        self.set_api_key(api_key)

    def set_api_key(self, api_key: str) -> None:
        self._client = OpenAI(base_url=OPENROUTER_URL, api_key=api_key, default_headers={"X-Title": "robot-head"})

    def set_personality(self, prompt: str) -> None:
        """Become someone else. The conversation so far belonged to the old character, so it's dropped."""
        self._personality = prompt.strip()
        self._history.clear()

    def set_expressions(self, names: list[str]) -> None:
        self._expressions = list(names)

    @property
    def system_prompt(self) -> str:
        rules = RULES.format(
            expressions=_cue_list(self._expressions),
            gestures=_cue_list(GESTURES),
            gazes=_cue_list(GAZES),
            lids=_span("lids"),
            tilt=_span("tilt"),
            roll=_span("roll"),
            look=_span("look_x"),
        )
        timeline = self.seen() if self.seen else ""
        remembered = self.remembered() if self.remembered else ""
        return (
            f"{self._personality}\n\n{rules}"
            + (SEEN.format(timeline=timeline) if timeline else "")
            + (f"\n{remembered}\n" if remembered else "")
        )

    def ask_json(self, prompt: str) -> str:
        """A one-off question whose answer must be a JSON object (used to think over memories)."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            extra_body={"reasoning": {"effort": self.reasoning_effort}} if self.reasoning_effort else None,
        )
        return response.choices[0].message.content or "{}"

    def observe(self, picture_jpeg: bytes, count: int, previous: str) -> str:
        """One short line on what the person is doing in a picture of several frames (or SAME / NOBODY)."""
        url = "data:image/jpeg;base64," + base64.b64encode(picture_jpeg).decode("ascii")
        prompt = OBSERVE.format(count=count, previous=previous)
        response = self._client.chat.completions.create(
            model=self.observe_model or self.model,
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": prompt}]}],
            max_tokens=MAX_TOKENS,
            extra_body={"reasoning": {"effort": self.reasoning_effort}} if self.reasoning_effort else None,
        )
        return (response.choices[0].message.content or "").strip()

    def recommendation_prompt(self, timeline: str) -> str:
        return RECOMMEND.format(timeline=timeline, about=self.recommend_about or RECOMMEND_ABOUT)

    def reply(self, text: str, image_jpeg: bytes | None = None) -> Iterator[str]:
        """Stream the reply to what the user said, as raw text chunks with cues still in them."""
        self._forget_old_images()
        content: list[dict] = [{"type": "text", "text": text}]
        if image_jpeg:
            url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")
            content.insert(0, {"type": "image_url", "image_url": {"url": url}})
        user = {"role": "user", "content": content}

        started = time.monotonic()
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.system_prompt}, *self._history, user],
            max_tokens=MAX_TOKENS,
            stream=True,
            extra_body={"reasoning": {"effort": self.reasoning_effort}} if self.reasoning_effort else None,
        )
        parts: list[str] = []
        finished = False
        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    if not parts:
                        print(f"[brain] first words after {time.monotonic() - started:.2f}s")
                    parts.append(delta)
                    yield delta
            finished = True
        finally:
            stream.close()
            # Remember the exchange even when it was cut off, marked with "—" so the model knows.
            if parts:
                reply = "".join(parts) + ("" if finished else " —")
                self._history += [user, {"role": "assistant", "content": reply}]
                del self._history[: -self._max_messages]

    def _forget_old_images(self) -> None:
        # Only the newest frame matters; older ones would just cost tokens on every call.
        for message in self._history:
            if isinstance(message["content"], list):
                message["content"] = " ".join(p["text"] for p in message["content"] if p["type"] == "text")
