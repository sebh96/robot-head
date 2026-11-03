"""Watching what the person does, so the head can talk about it and recommend things.

Every look takes a few frames spread over the interval and puts them side by side in one
picture, so the model sees an action happening rather than a single pose. The answers
become a short timeline of activities, which goes into the brain's system prompt. When a
new activity starts, or one has gone on for a long time, the head is asked for a tip.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

GRID_WIDTH = 960  # the combined picture's width; each frame gets an equal share
TIMELINE_LENGTH = 20
SAME, NOBODY = "SAME", "NOBODY"


@dataclass
class Activity:
    description: str
    start: float  # time.time()
    last_seen: float
    tipped: bool = False  # got a "you've been at this a while" recommendation


def grid(frames: list[np.ndarray]) -> np.ndarray:
    """The frames in reading order, numbered, as one picture: 2x2 for four, 3x3 for nine."""
    columns = math.ceil(math.sqrt(len(frames)))
    rows = math.ceil(len(frames) / columns)
    size = (GRID_WIDTH // columns, GRID_WIDTH // columns * 3 // 4)
    tiles = []
    for number, frame in enumerate(frames, 1):
        tile = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        cv2.putText(tile, str(number), (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    tiles += [np.zeros_like(tiles[0])] * (rows * columns - len(tiles))
    return np.vstack([np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)])


def difference(a: np.ndarray, b: np.ndarray) -> float:
    """How much two frames differ, as the mean change per pixel (0-255)."""
    small = [cv2.cvtColor(cv2.resize(f, (80, 60), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for f in (a, b)]
    return float(np.mean(cv2.absdiff(small[0], small[1])))


class Observer:
    """Every setting is a plain attribute, so the control panel can change it while it runs."""

    def __init__(
        self,
        camera,
        describe: Callable[[bytes, int, str], str],
        every_seconds: float = 30.0,
        frames: int = 9,
        change_threshold: float = 6.0,
        recommend_every_seconds: float = 300.0,
        long_activity_minutes: float = 20.0,
        on_recommend: Callable[[str], None] | None = None,
        log: Callable[[str, str], None] | None = None,
    ) -> None:
        self.camera = camera
        self.describe = describe  # (picture as JPEG, frames in it, previous activity) -> answer
        self.every_seconds = every_seconds  # 0 turns watching off
        self.frames = frames  # per look, spread over every_seconds
        self.change_threshold = change_threshold  # below this mean pixel change (0-255) nothing new happened
        self.recommend_every_seconds = recommend_every_seconds  # 0 turns tips off
        self.long_activity_minutes = long_activity_minutes  # doing one thing this long earns another tip
        self.on_recommend = on_recommend
        self.on_activity: Callable[[str], None] | None = None  # a new activity was seen (for the memory)
        self.log = log or (lambda kind, text: None)
        self.paused = False
        self._lock = threading.Lock()
        self._timeline: deque[Activity] = deque(maxlen=TIMELINE_LENGTH)
        self._last_frame: np.ndarray | None = None
        self._last_tip = float("-inf")

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            if not self.every_seconds or self.paused:
                stop.wait(1.0)
                continue
            frames, someone, count = [], False, max(1, int(self.frames))
            for _ in range(count):
                if stop.wait(self.every_seconds / count):
                    return
                frame = self.camera.latest_frame()
                if frame is not None:
                    frames.append(frame)
                    someone = someone or self.camera.face() is not None
            if frames:
                try:
                    self.look(frames, someone, time.time())
                except Exception as error:  # a failed look must not stop the watching
                    self.log("error", f"Couldn't make sense of the camera: {error}")

    def look(self, frames: list[np.ndarray], someone: bool, now: float) -> None:
        """Work out what they're doing from frames taken over the last interval."""
        current = self.current()
        still = self._last_frame is not None and difference(frames[-1], self._last_frame) < self.change_threshold
        self._last_frame = frames[-1]
        if still:
            if current is not None and someone:
                current.last_seen = now  # same scene, so the same activity: no need to ask
                self._maybe_recommend(now)
            return
        if not someone and current is None:
            return  # nobody there, and nothing to compare with
        ok, jpeg = cv2.imencode(".jpg", grid(frames), [cv2.IMWRITE_JPEG_QUALITY, 70])
        answer = self.describe(jpeg.tobytes(), len(frames), current.description if current else "nothing yet").strip()
        if answer.upper().startswith(NOBODY) or not answer:
            return
        if answer.upper().startswith(SAME):
            if current is None:
                return
            current.last_seen = now
        else:
            with self._lock:
                self._timeline.append(Activity(answer.rstrip("."), now, now))
            self.log("info", f"(sees: {answer})")
            if self.on_activity:
                self.on_activity(answer.rstrip("."))
        self._maybe_recommend(now)

    def current(self) -> Activity | None:
        with self._lock:
            return self._timeline[-1] if self._timeline else None

    def timeline(self) -> list[dict]:
        with self._lock:
            return [
                {"description": a.description, "start": a.start, "minutes": round((a.last_seen - a.start) / 60)}
                for a in self._timeline
            ]

    def summary(self, now: float | None = None) -> str:
        """The timeline in words, oldest first, for the brain."""
        now = now or time.time()
        lines = []
        for activity in list(self._timeline):
            minutes = round((activity.last_seen - activity.start) / 60)
            when = time.strftime("%H:%M", time.localtime(activity.start))
            lines.append(f"- {when}, for {minutes} min: {activity.description}" if minutes else f"- {when}: {activity.description}")
        current = self.current()
        if current is not None and now - current.last_seen > 3 * max(self.every_seconds, 1):
            lines.append(f"- not seen since {time.strftime('%H:%M', time.localtime(current.last_seen))}")
        return "\n".join(lines)

    def _maybe_recommend(self, now: float) -> None:
        current = self.current()
        if not (current and self.on_recommend and self.recommend_every_seconds):
            return
        if now - self._last_tip < self.recommend_every_seconds:
            return
        new = current.last_seen == current.start
        long = now - current.start >= self.long_activity_minutes * 60 and not current.tipped
        if new or long:
            current.tipped = current.tipped or long
            self._last_tip = now
            self.on_recommend(self.summary(now))
