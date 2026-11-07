"""Full-duplex audio: plays the voice, records the microphone, and cancels the voice's own
echo out of the recording, so you can talk over the head and interrupt it."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np
import sounddevice as sd
import soxr
from livekit import rtc

RATE = 48000
FRAME = RATE // 100  # the echo canceller works on 10 ms frames
FRAMES_PER_BLOCK = 3  # loudness is judged per 30 ms block
BLOCK_SECONDS = FRAMES_PER_BLOCK * FRAME / RATE
WHISPER_RATE = 16000
START_BLOCKS = 3  # loud blocks in a row that start an utterance
# While the head talks, leftover echo comes in short bursts but real speech stays loud,
# so interrupting takes 8 loud blocks out of the last 10 (300 ms) above a higher floor.
BARGE_IN_WINDOW = 10
BARGE_IN_LOUD = 8
BARGE_IN_FLOOR = 0.006
ECHO_TAIL_BLOCKS = 10  # the room keeps echoing for a moment after the voice stops
MIN_THRESHOLD = 0.005
PREROLL_SECONDS = 0.3  # keep a little audio from before the trigger so first syllables aren't clipped
MAX_SECONDS = 15


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))))


class EchoCanceller:
    """WebRTC's echo canceller (the one in video calls): removes what the speaker played from the mic."""

    def __init__(self) -> None:
        self._apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=True, high_pass_filter=True)
        self.delay_ms = 0

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        """Clean one 10 ms microphone frame (``near``) given the frame played at the same time (``far``)."""
        self._apm.process_reverse_stream(self._frame(far))
        self._apm.set_stream_delay_ms(self.delay_ms)
        frame = self._frame(near)
        self._apm.process_stream(frame)
        return np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32767

    @staticmethod
    def _frame(audio: np.ndarray) -> rtc.AudioFrame:
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        return rtc.AudioFrame(pcm.tobytes(), RATE, 1, len(pcm))


class SpeechDetector:
    """Finds utterances in the echo-free microphone signal, 30 ms at a time."""

    def __init__(self, silence_seconds: float = 0.8, barge_in: bool = True) -> None:
        self.threshold = MIN_THRESHOLD
        self.levels: deque[float] = deque(maxlen=round(1 / BLOCK_SECONDS))
        self.barge_in = barge_in
        self.silence_blocks = round(silence_seconds / BLOCK_SECONDS)
        self._recent: deque[bool] = deque(maxlen=BARGE_IN_WINDOW)
        self._preroll: deque[np.ndarray] = deque(maxlen=round(PREROLL_SECONDS / BLOCK_SECONDS))
        self._recorded: list[np.ndarray] = []
        self._quiet = 0
        self._since_voice = ECHO_TAIL_BLOCKS

    @property
    def recording(self) -> bool:
        return bool(self._recorded)

    def feed(self, block: np.ndarray, voice_playing: bool) -> tuple[bool, np.ndarray | None]:
        """Returns (whether to interrupt the voice, a finished utterance at 48 kHz or None)."""
        level = _rms(block)
        self.levels.append(level)
        self._since_voice = 0 if voice_playing else self._since_voice + 1
        head_talking = self._since_voice < ECHO_TAIL_BLOCKS
        loud = level > (max(self.threshold, BARGE_IN_FLOOR) if head_talking else self.threshold)
        self._recent.append(loud)

        if not self._recorded:
            self._preroll.append(block)
            if head_talking:
                if not self.barge_in or sum(self._recent) < BARGE_IN_LOUD:
                    return False, None
            elif list(self._recent)[-START_BLOCKS:] != [True] * START_BLOCKS:
                return False, None
            self._recorded = list(self._preroll)
            self._quiet = 0
            return head_talking, None

        self._recorded.append(block)
        self._quiet = 0 if loud else self._quiet + 1
        if self._quiet < self.silence_blocks and len(self._recorded) * BLOCK_SECONDS < MAX_SECONDS:
            return False, None
        utterance = np.concatenate(self._recorded)
        self._recorded = []
        self._recent.clear()
        self._preroll.clear()
        return False, utterance


class AudioEngine:
    def __init__(
        self,
        mic: int | None = None,
        speaker: int | None = None,
        listen: bool = True,
        barge_in: bool = True,
        silence_seconds: float = 0.8,
    ) -> None:
        self.muted = False  # still talks (timing, jaw), just without sound
        self.on_play: Callable[[np.ndarray], None] | None = None  # every sentence as it starts, e.g. for browsers
        self.on_stop: Callable[[], None] | None = None  # a sentence was cut off
        self._listening = listen
        self._utterances: queue.Queue[np.ndarray] = queue.Queue()
        self._echo = EchoCanceller()
        self._detector = SpeechDetector(silence_seconds, barge_in)
        self._raw: queue.Queue[tuple[np.ndarray, np.ndarray, bool]] = queue.Queue()
        self._running = False

        # Playback, shared with the audio callback.
        self._lock = threading.Lock()
        self._playback = np.zeros(0, dtype=np.float32)
        self._played = 0
        self._interrupted = threading.Event()

        if listen:
            self._stream = sd.Stream(
                samplerate=RATE, blocksize=FRAME, channels=1, dtype="float32",
                device=(mic, speaker), callback=self._on_duplex,
            )
        else:
            self._stream = sd.OutputStream(
                samplerate=RATE, blocksize=FRAME, channels=1, dtype="float32", device=speaker, callback=self._on_output
            )
        self._worker = threading.Thread(target=self._process_loop, daemon=True)

    def start(self) -> None:
        self._running = True
        self._stream.start()
        if self._listening:
            # Echo shows up in the recording one output plus one input latency after it was played.
            self._echo.delay_ms = round(sum(self._stream.latency) * 1000)
            self._worker.start()

    def close(self) -> None:
        self._running = False
        self._stream.stop()
        self._stream.close()

    def set_barge_in(self, enabled: bool) -> None:
        self._detector.barge_in = enabled

    def set_silence(self, seconds: float) -> None:
        self._detector.silence_blocks = round(seconds / BLOCK_SECONDS)

    def calibrate(self, seconds: float = 1.0) -> None:
        """Measure the room's background noise so speech detection sits above it."""
        time.sleep(seconds + 0.1)
        if self._detector.levels:
            self._detector.threshold = max(3 * float(np.median(self._detector.levels)), MIN_THRESHOLD)

    def next_utterance(self, timeout: float) -> np.ndarray | None:
        """The next thing someone said, as 16 kHz audio for Whisper, or None if nobody spoke in time."""
        try:
            return self._utterances.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def user_talking(self) -> bool:
        return self._detector.recording

    # --- Playback ---

    def play(self, audio: np.ndarray) -> None:
        """Start playing 48 kHz audio, replacing whatever was playing."""
        with self._lock:
            self._playback = audio.astype(np.float32)
            self._played = 0
        self._interrupted.clear()
        if self.on_play:
            self.on_play(audio)

    def stop_playback(self, interrupted: bool = False) -> None:
        with self._lock:
            was_playing = self._played < len(self._playback)
            self._playback = np.zeros(0, dtype=np.float32)
            self._played = 0
        if interrupted:
            self._interrupted.set()
        if was_playing and self.on_stop:
            self.on_stop()

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._played < len(self._playback)

    @property
    def interrupted(self) -> bool:
        return self._interrupted.is_set()

    @property
    def position(self) -> int:
        """Samples of the current sound that have come out of the speaker so far."""
        latency = self._stream.latency[1] if self._listening else self._stream.latency
        with self._lock:
            return max(0, self._played - round(latency * RATE))

    def _next_output(self, frames: int) -> tuple[np.ndarray, bool]:
        """The next samples to play, and whether the voice is playing (pauses between words count)."""
        with self._lock:
            chunk = self._playback[self._played : self._played + frames]
            self._played += len(chunk)
        voice_playing = len(chunk) > 0
        if self.muted:
            chunk = np.zeros(frames, dtype=np.float32)
        elif len(chunk) < frames:
            chunk = np.concatenate([chunk, np.zeros(frames - len(chunk), dtype=np.float32)])
        return chunk, voice_playing

    def _on_output(self, outdata, frames, time_info, status) -> None:
        outdata[:, 0] = self._next_output(frames)[0]

    def _on_duplex(self, indata, outdata, frames, time_info, status) -> None:
        # Keep the callback quick: echo cancelling runs on the processing thread.
        far, voice_playing = self._next_output(frames)
        outdata[:, 0] = far
        self._raw.put((indata[:, 0].copy(), far, voice_playing))

    # --- Listening ---

    def _process_loop(self) -> None:
        frames: list[np.ndarray] = []
        voice_playing = False
        while self._running:
            try:
                near, far, playing = self._raw.get(timeout=0.1)
            except queue.Empty:
                continue
            frames.append(self._echo.process(near, far))
            voice_playing |= playing
            if len(frames) < FRAMES_PER_BLOCK:
                continue
            interrupt, utterance = self._detector.feed(np.concatenate(frames), voice_playing)
            frames, voice_playing = [], False
            if interrupt:
                self.stop_playback(interrupted=True)
            if utterance is not None:
                self._utterances.put(soxr.resample(utterance, RATE, WHISPER_RATE).astype(np.float32))
