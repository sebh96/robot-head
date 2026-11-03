"""The fast layer: everything that keeps the head looking alive between LLM calls.

Runs at about 30 Hz on its own thread. Face tracking, saccades, blinking,
breathing and the jaw all happen here; the brain only nudges this layer with
a mode, an expression, gestures and gaze cues.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Iterable

from .cues import GESTURES
from .expressions import NEUTRAL, PRESETS, Expression
from .motors import JOINTS, Pose, rest_pose

IDLE_LIDS = 0.75  # droopier lids while nobody is talking to it
GAZE_TARGETS = {"look_left": (-0.8, 0.0), "look_right": (0.8, 0.0), "look_up": (0.0, 0.7), "look_down": (0.0, -0.7)}
THINKING_GAZE = (-0.35, 0.5)  # people glance up and away while they think
EYE_SHARE = 0.4  # part of a gaze shift the eyes cover before the neck catches up
FACE_GAIN = 0.7  # a face at the edge of the camera image -> this much of the full gaze range
LOOK_SECONDS = 2.5
EXPRESSION_SECONDS = 8.0
GESTURE_SECONDS = {"nod": 0.9, "shake": 1.0, "tilt_left": 1.6, "tilt_right": 1.6, "roll_eyes": 1.0}
BLINK_SECONDS = 0.18


def approach(current: float, target: float, dt: float, tau: float) -> float:
    """Exponential smoothing that behaves the same at any frame rate."""
    return target + (current - target) * math.exp(-dt / tau)


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


class Reflexes:
    def __init__(self, motors, camera=None, monitor=None, fps: int = 30) -> None:
        self.motors = motors
        self.camera = camera
        self.monitor = monitor  # gets every pose, for the control panel's face and wiring diagram
        self.fps = fps
        self.pose: Pose = rest_pose()

        # Set from the conversation thread, read by the reflex thread.
        self._lock = threading.Lock()
        self._mode = "idle"
        self._expressions = {expression.name: expression for expression in PRESETS}
        self._expression = NEUTRAL
        self._expression_until = 0.0
        self._gesture: tuple[str, float] | None = None
        self._look: tuple[float, float, float] | None = None  # x, y, until
        self._jaw_target = 0.0

        # Only touched by the reflex thread.
        self._wander = (0.0, 0.0)
        self._next_saccade = 0.0
        self._had_face = False
        self._blink_start = -1.0
        self._next_blink = 0.0
        self._pan = self._tilt = self._roll = 0.0
        self._eyes = (0.0, 0.0)
        self._jaw = 0.0

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self._mode = mode

    def set_jaw(self, level: float) -> None:
        with self._lock:
            self._jaw_target = level

    def set_expressions(self, expressions: Iterable[Expression]) -> None:
        """Replace the expressions cues can name, e.g. after one was edited or invented."""
        with self._lock:
            self._expressions = {expression.name: expression for expression in expressions}
            self._expression = self._expressions.get(self._expression.name, self._expression)

    def cue(self, name: str) -> None:
        """Play a named expression, gesture or gaze cue; unknown names are ignored."""
        now = time.monotonic()
        with self._lock:
            if name in self._expressions:
                self._expression = self._expressions[name]
                self._expression_until = now + EXPRESSION_SECONDS
            elif name in GESTURES:
                self._gesture = (name, now)
            elif name == "look_at_you":
                self._look = None
            elif name in GAZE_TARGETS:
                self._look = (*GAZE_TARGETS[name], now + LOOK_SECONDS)

    def show(self, expression: Expression) -> None:
        """Wear an expression that has no name to cue it by, like a preview from the control panel."""
        with self._lock:
            self._expression = expression
            self._expression_until = time.monotonic() + EXPRESSION_SECONDS

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def expression(self) -> str:
        return self._expression.name

    @property
    def status(self) -> str:
        with self._lock:
            return f"{self._mode} | {self._expression.name}"

    def run(self, stop: threading.Event) -> None:
        last = time.monotonic()
        while not stop.is_set():
            now = time.monotonic()
            face = self.camera.face() if self.camera else None
            self.pose = self.step(now, now - last, face)
            last = now
            self.motors.write(self.pose)
            if self.monitor is not None:
                self.monitor.update(self.pose, now)
            time.sleep(max(0.0, 1 / self.fps - (time.monotonic() - now)))

    def step(self, now: float, dt: float, face: tuple[float, float] | None = None) -> Pose:
        """Advance every motion by ``dt`` seconds and return the new pose."""
        with self._lock:
            if self._expression.name != "neutral" and now > self._expression_until:
                self._expression = self._expressions.get("neutral", NEUTRAL)
            mode, expression, gesture, look = self._mode, self._expression, self._gesture, self._look
            jaw_target = self._jaw_target
        gx, gy = self._gaze_target(now, mode, look, face)
        if mode != "thinking":
            gx, gy = gx + expression.look_x, gy + expression.look_y

        # The neck drifts toward the target slowly; the eyes jump there and then
        # re-center as the neck catches up, the way a person turns to look.
        breathing = 0.04 * math.sin(2 * math.pi * 0.2 * now)
        self._pan = approach(self._pan, gx * (1 - EYE_SHARE), dt, 0.45)
        self._tilt = approach(self._tilt, gy * (1 - EYE_SHARE) + breathing + expression.tilt, dt, 0.45)
        self._roll = approach(self._roll, expression.roll, dt, 0.6)
        eyes_x = _clamp((gx - self._pan) / EYE_SHARE)
        eyes_y = _clamp((gy - self._tilt) / EYE_SHARE)
        self._eyes = (approach(self._eyes[0], eyes_x, dt, 0.04), approach(self._eyes[1], eyes_y, dt, 0.04))

        pan, tilt, roll = self._pan, self._tilt, self._roll
        eyes_x, eyes_y = self._eyes
        if gesture is not None:
            name, start = gesture
            t, length = now - start, GESTURE_SECONDS[name]
            fade = 1 - t / length
            if t >= length:
                with self._lock:
                    if self._gesture == gesture:
                        self._gesture = None
            elif name == "nod":
                tilt -= 0.35 * math.sin(2 * math.pi * t / 0.45) * fade
            elif name == "shake":
                pan += 0.35 * math.sin(2 * math.pi * t / 0.5) * fade
            elif name == "roll_eyes":
                # Sweep from one side, over the top, to the other side.
                eyes_x = -0.8 * math.cos(math.pi * t / length)
                eyes_y = 0.9 * math.sin(math.pi * t / length)
            else:
                roll += (0.6 if name == "tilt_right" else -0.6) * math.sin(math.pi * t / length)

        # Fast attack, slower release, so the jaw snaps open on syllables.
        self._jaw = approach(self._jaw, jaw_target, dt, 0.03 if jaw_target > self._jaw else 0.08)

        pose = {
            "neck_pan": pan,
            "neck_tilt": tilt,
            "neck_roll": roll,
            "eyes_x": eyes_x,
            "eyes_y": eyes_y,
            "lids": expression.lids * (IDLE_LIDS if mode == "idle" else 1.0) * (1 - self._blink(now)),
            "jaw": self._jaw,
        }
        return {name: JOINTS[name].clamp(value) for name, value in pose.items()}

    def _gaze_target(self, now, mode, look, face) -> tuple[float, float]:
        if mode == "thinking":
            return THINKING_GAZE
        if look is not None and now < look[2]:
            return look[0], look[1]

        # Small jumps between points of a face while tracking it, bigger ones
        # around the room when nobody is there, so the head never freezes.
        if (face is not None) != self._had_face:
            self._had_face = face is not None
            self._next_saccade = now
        if now >= self._next_saccade:
            if face is not None:
                self._wander = (random.uniform(-0.06, 0.06), random.uniform(-0.04, 0.04))
                self._next_saccade = now + random.uniform(0.4, 1.5)
            else:
                self._wander = (random.uniform(-0.4, 0.4), random.uniform(-0.2, 0.2))
                self._next_saccade = now + random.uniform(1.0, 4.0)
        if face is None:
            return self._wander
        return face[0] * FACE_GAIN + self._wander[0], face[1] * FACE_GAIN + self._wander[1]

    def _blink(self, now: float) -> float:
        """How far the lids are shut by blinking right now (0 open .. 1 shut)."""
        if now >= self._next_blink:
            self._blink_start = now
            self._next_blink = now + random.uniform(2.0, 6.0)
        t = now - self._blink_start
        return math.sin(math.pi * t / BLINK_SECONDS) if 0 <= t < BLINK_SECONDS else 0.0
