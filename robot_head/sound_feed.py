"""The robot's voice for browsers: every sentence as it starts playing, so the control panel can play it too.

Handy when the robot has no speaker of its own, or to hear it from another room.
"""

from __future__ import annotations

import base64
import queue
import threading

import numpy as np
import soxr

from .audio import RATE

FEED_RATE = 24000  # plenty for a voice, and half the data of the engine's 48 kHz


class SoundFeed:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._listeners: list[queue.Queue[dict]] = []

    def subscribe(self) -> queue.Queue[dict]:
        listener: queue.Queue[dict] = queue.Queue(maxsize=50)  # a stalled page drops sentences rather than piling up
        with self._lock:
            self._listeners.append(listener)
        return listener

    def unsubscribe(self, listener: queue.Queue[dict]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def play(self, audio: np.ndarray) -> None:
        """A sentence starts playing (48 kHz). Sent as 16-bit samples, base64 encoded."""
        if not self._listeners:
            return  # nobody listens: don't spend time encoding
        samples = soxr.resample(audio, RATE, FEED_RATE)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
        self._send({"rate": FEED_RATE, "pcm": base64.b64encode(pcm.tobytes()).decode("ascii")})

    def stop(self) -> None:
        """Someone talked over it, or it was paused: browsers cut the sentence off too."""
        self._send({"stop": True})

    def _send(self, event: dict) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener.put_nowait(event)
            except queue.Full:
                pass
