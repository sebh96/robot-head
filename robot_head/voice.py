"""Text-to-speech, and a jaw that follows the audio.

Two engines: Kokoro (the default) sounds far more natural; Piper is lighter, which matters
on a Raspberry Pi, and has many more languages, German included.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import soxr

from .audio import RATE, AudioEngine
from .downloads import MODELS_DIR, fetch

JAW_FPS = 50
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_DIR = MODELS_DIR / "kokoro"
# Kokoro voice names start with a language letter: am_michael is American English, bm_george British.
KOKORO_LANGUAGES = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br"}
PIPER_DIR = MODELS_DIR / "voices"
DEFAULT_VOICES = {"kokoro": "am_michael", "piper": "en_US-bryce-medium"}


def voice_or_default(engine: str, voice: str) -> str:
    """The voice if it is one of the engine's (Kokoro: am_michael, Piper: en_US-bryce-medium), else its default.
    So a personality made for one engine still talks after switching to the other, say on a Raspberry Pi."""
    return voice if voice and ("-" in voice) == (engine == "piper") else DEFAULT_VOICES[engine]


def envelope(audio: np.ndarray, rate: int, fps: int = JAW_FPS) -> np.ndarray:
    """Loudness of the audio in ``fps`` steps per second, scaled 0..1 for the jaw."""
    hop = max(1, rate // fps)
    count = len(audio) // hop
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    rms = np.sqrt(np.mean(np.square(audio[: count * hop].reshape(count, hop)), axis=1))
    loud = np.percentile(rms, 90)
    if loud <= 0:
        return np.zeros(count, dtype=np.float32)
    # Below a tenth of a typical loud frame the mouth stays shut, so pauses read as pauses.
    return np.clip((rms / loud - 0.1) / 0.9, 0.0, 1.0).astype(np.float32)


class KokoroSpeech:
    def __init__(self, voice: str, speed: float = 1.0) -> None:
        from kokoro_onnx import Kokoro

        model = fetch(KOKORO_URL + "kokoro-v1.0.onnx", KOKORO_DIR / "kokoro-v1.0.onnx", "the Kokoro voice model (326 MB)")
        voices = fetch(KOKORO_URL + "voices-v1.0.bin", KOKORO_DIR / "voices-v1.0.bin", "the Kokoro voices (28 MB)")
        self._kokoro = Kokoro(str(model), str(voices))
        self._use(voice, speed)

    def voices(self) -> list[str]:
        return sorted(self._kokoro.get_voices())

    def variant(self, voice: str, speed: float) -> KokoroSpeech:
        """The same model speaking with another voice or speed; switching is instant."""
        other = copy.copy(self)
        other._use(voice_or_default("kokoro", voice), speed)
        return other

    def _use(self, voice: str, speed: float) -> None:
        if voice not in self._kokoro.get_voices():
            raise ValueError(f"Kokoro has no voice {voice!r}; see --list-voices.")
        self._voice = voice
        self._speed = speed
        self._language = KOKORO_LANGUAGES.get(voice[0], "en-us")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        return self._kokoro.create(text, voice=self._voice, speed=self._speed, lang=self._language)


class PiperSpeech:
    def __init__(self, voice: str, speed: float = 1.0) -> None:
        from piper import PiperVoice, SynthesisConfig
        from piper.download_voices import download_voice

        model = PIPER_DIR / f"{voice}.onnx"
        if not model.exists():
            print(f"Downloading the {voice} voice (about 60 MB, first run only)...")
            PIPER_DIR.mkdir(parents=True, exist_ok=True)
            download_voice(voice, PIPER_DIR)
        self._piper = PiperVoice.load(model)
        self._config = SynthesisConfig(length_scale=1 / speed)
        self._voice = voice

    def voices(self) -> list[str]:
        """The voices already downloaded; any other name from --list-voices downloads when chosen."""
        return sorted(path.stem for path in PIPER_DIR.glob("*.onnx"))

    def variant(self, voice: str, speed: float) -> PiperSpeech:
        voice = voice_or_default("piper", voice)
        if voice != self._voice:
            return PiperSpeech(voice, speed)
        from piper import SynthesisConfig

        other = copy.copy(self)
        other._config = SynthesisConfig(length_scale=1 / speed)
        return other

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        chunks = list(self._piper.synthesize(text, self._config))
        if not chunks:
            return np.zeros(0, dtype=np.float32), self._piper.config.sample_rate
        return np.concatenate([chunk.audio_float_array for chunk in chunks]), chunks[0].sample_rate


def make_speech(engine: str, voice: str = "", speed: float = 1.0) -> KokoroSpeech | PiperSpeech:
    if engine not in DEFAULT_VOICES:
        raise ValueError(f"TTS_ENGINE must be kokoro or piper, not {engine!r}.")
    speech = KokoroSpeech if engine == "kokoro" else PiperSpeech
    return speech(voice_or_default(engine, voice), speed)


def print_voices(engine: str) -> None:
    if engine == "kokoro":
        print("\n".join(KokoroSpeech(DEFAULT_VOICES["kokoro"]).voices()))
    else:
        from piper.download_voices import list_voices

        list_voices()


class Voice:
    def __init__(self, engine: AudioEngine, on_level: Callable[[float], None], speech: KokoroSpeech | PiperSpeech) -> None:
        self._engine = engine
        self._on_level = on_level
        self._speech = speech
        self._renderer = ThreadPoolExecutor(max_workers=1)  # one sentence at a time, in order

    @property
    def speech(self) -> KokoroSpeech | PiperSpeech:
        return self._speech

    def set_speech(self, speech: KokoroSpeech | PiperSpeech) -> None:
        """Switch voice; sentences already being prepared still finish in the old one."""
        self._renderer.submit(setattr, self, "_speech", speech)

    def prepare(self, text: str) -> Future[np.ndarray]:
        """Start making the audio for ``text`` in the background, so it's ready when its turn comes."""
        return self._renderer.submit(self._render, text)

    def play(self, audio: Future[np.ndarray]) -> bool:
        """Play prepared audio, moving the jaw in step with it. Returns False if someone talked over it."""
        samples = audio.result()
        if self._engine.user_talking:  # they started talking while this sentence was being made
            return False
        levels = envelope(samples, RATE)
        hop = RATE // JAW_FPS
        self._engine.play(samples)
        while self._engine.playing and len(levels):
            self._on_level(float(levels[min(self._engine.position // hop, len(levels) - 1)]))
            time.sleep(1 / JAW_FPS)
        self._on_level(0.0)
        return not self._engine.interrupted

    def _render(self, text: str) -> np.ndarray:
        audio, rate = self._speech.synthesize(text)
        return soxr.resample(audio, rate, RATE).astype(np.float32)
