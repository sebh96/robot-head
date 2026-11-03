"""Turn the LLM's streamed reply into speakable sentences plus movement cues.

The model moves the head by writing cues like ``[nod]`` or ``[smug]`` inline. Each cue fires
when the sentence it sits in starts playing, so a gesture lands on the words it belongs to,
and it all comes from one LLM call. A cue with settings, like ``[disgusted lids=0.3 roll=-0.2]``,
invents a new expression.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

GESTURES = ("nod", "shake", "tilt_left", "tilt_right", "roll_eyes")
GAZES = ("look_left", "look_right", "look_up", "look_down", "look_at_you")

_CUE = re.compile(r"\[([^\[\]\n]{1,160})\]")
_STAGE_DIRECTION = re.compile(r"\*[^*\n]+\*")  # "*smiles*": models do this despite being told not to
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.!?…])")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")
FIRST_CLAUSE_WORDS = 4  # a long enough opening clause is spoken before its sentence ends


@dataclass
class Segment:
    text: str
    cues: list[str] = field(default_factory=list)
    # Expressions invented in this sentence: name -> settings, from cues like [name lids=0.3].
    new_expressions: dict[str, dict[str, str]] = field(default_factory=dict)
    learned_name: str | None = None  # from [name=Sebastian]: the person told it their name


def parse_cue(content: str) -> tuple[str, dict[str, str]]:
    """Split "disgusted lids=0.3 look=0, -0.2" into a name and its settings; plain cues have none.
    Names and keys are lowercased; values keep their case (for [name=Sebastian])."""
    content = re.sub(r"\s*=\s*", "=", re.sub(r",\s+", ",", content.strip()))
    words = content.split()
    settings = {key.lower(): value for key, value in (word.split("=", 1) for word in words if "=" in word)}
    return "_".join(word.lower() for word in words if "=" not in word), settings


def parse_segment(raw: str) -> Segment:
    """Split one sentence into the words to speak and the cues in it."""
    segment = Segment("")
    for match in _CUE.finditer(raw):
        name, settings = parse_cue(match.group(1))
        if not name and settings.get("name"):
            segment.learned_name = settings["name"].replace("_", " ").strip()
            continue
        if not name:
            continue
        if settings:
            segment.new_expressions[name] = settings
        segment.cues.append(name)
    # All bracketed text is dropped, so unknown cues like "[laughs]" are never read out.
    text = _STAGE_DIRECTION.sub(" ", _CUE.sub(" ", raw))
    segment.text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", " ".join(text.split()))
    return segment


class SentenceStream:
    """Buffers streamed text and releases it a sentence at a time."""

    def __init__(self) -> None:
        self._buffer = ""
        self._started = False

    def feed(self, chunk: str) -> list[Segment]:
        self._buffer += chunk
        *complete, self._buffer = _SENTENCE_END.split(self._buffer)
        if not complete and not self._started:
            # Get the first words out sooner: a long opening clause doesn't wait for its sentence.
            for match in _CLAUSE_END.finditer(self._buffer):
                if len(parse_segment(self._buffer[: match.start()]).text.split()) >= FIRST_CLAUSE_WORDS:
                    complete, self._buffer = [self._buffer[: match.start()]], self._buffer[match.end() :]
                    break
        segments = [s for s in map(parse_segment, complete) if s.text or s.cues]
        self._started = self._started or bool(segments)
        return segments

    def flush(self) -> list[Segment]:
        rest, self._buffer = self._buffer, ""
        segment = parse_segment(rest)
        return [segment] if segment.text or segment.cues else []
