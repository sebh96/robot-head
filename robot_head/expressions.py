"""Facial expressions: how open the eyelids are, how the head is held, and where the eyes drift.

Expressions are data, kept in the database: you can edit them in the control panel, and the
LLM can invent new ones mid-sentence, like [disgusted lids=0.3 tilt=0.1 roll=-0.2 look=0,-0.3].
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Expression:
    name: str
    lids: float = 0.7  # 0 shut .. 1 wide open
    tilt: float = 0.0  # chin down -0.5 .. 0.5 chin up
    roll: float = 0.0  # lean toward its left shoulder -0.6 .. 0.6 its right
    look_x: float = 0.0  # eyes drift to its left -0.6 .. 0.6 its right
    look_y: float = 0.0  # eyes drift down -0.6 .. 0.6 up
    source: str = "preset"  # preset, user or llm


# The range of every adjustable setting, (low, high).
RANGES = {"lids": (0.0, 1.0), "tilt": (-0.5, 0.5), "roll": (-0.6, 0.6), "look_x": (-0.6, 0.6), "look_y": (-0.6, 0.6)}

PRESETS = (
    Expression("neutral"),
    Expression("happy", lids=0.55),
    Expression("surprised", lids=1.0),
    Expression("sleepy", lids=0.3),
    Expression("angry", lids=0.45),
    Expression("sad", lids=0.5, look_y=-0.3),
    Expression("curious", lids=0.85, roll=0.3),
    Expression("smug", lids=0.5, tilt=0.12, roll=-0.15),
)
NEUTRAL = PRESETS[0]

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,23}$")


def valid_name(name: str) -> bool:
    return bool(_NAME.match(name))


def clamped(expression: Expression) -> Expression:
    return replace(expression, **{key: min(high, max(low, getattr(expression, key))) for key, (low, high) in RANGES.items()})


def from_settings(name: str, settings: dict[str, str], source: str = "llm") -> Expression:
    """An expression from cue settings like {"lids": "0.3", "look": "0,-0.2"}. Unusable values are skipped."""
    values: dict[str, float] = {}
    for key, raw in settings.items():
        try:
            if key == "look":
                values["look_x"], values["look_y"] = (float(part) for part in raw.split(","))
            elif key in RANGES:
                values[key] = float(raw)
        except ValueError:
            continue
    return clamped(Expression(name, source=source, **values))
