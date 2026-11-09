"""Speech recognition, locally: turns a recorded utterance into text.

Two engines: NVIDIA Parakeet (the default) is accurate and fast on a CPU, and understands
25 European languages on its own. Whisper covers far more languages but is much slower,
because it always processes a full 30-second window.
"""

from __future__ import annotations

import time

import numpy as np
import soundfile as sf

from .config import PROJECT_DIR

RATE = 16000
RECORDINGS_DIR = PROJECT_DIR / "recordings"
DEFAULT_MODELS = {"parakeet": "nemo-parakeet-tdt-0.6b-v3", "whisper": "small"}
MIN_SPEECH_SECONDS = 0.15  # less speech than this in a sound, and it was a tap, a cough or a chair
SPEECH_LIKELY = 0.5  # how sure the voice detector must be that a moment is speech


class SpeechCheck:
    """Silero VAD, a small voice detector: tells speech from the other sounds that make the microphone jump.

    Speech recognition turns any sound into its best guess at words ("Mm.", "Yeah."), so a sound
    without speech in it is better never recognised at all.
    """

    def __init__(self, vad=None) -> None:
        if vad is None:
            import onnx_asr

            vad = onnx_asr.load_vad("silero")
        self._vad = vad

    def speech_seconds(self, audio: np.ndarray) -> float:
        """How much of 16 kHz mono audio is speech."""
        if len(audio) == 0:
            return 0.0
        (segments,) = self._vad.segment_batch(
            audio[np.newaxis, :].astype(np.float32), np.array([len(audio)]), RATE,
            threshold=SPEECH_LIKELY, min_speech_duration_ms=0, speech_pad_ms=0,
        )
        return sum(end - start for start, end in segments) / RATE

    def is_speech(self, audio: np.ndarray) -> bool:
        return self.speech_seconds(audio) >= MIN_SPEECH_SECONDS


class ParakeetEars:
    def __init__(self, model: str) -> None:
        import onnx_asr

        self._model = onnx_asr.load_model(model, quantization="int8")

    def transcribe(self, audio: np.ndarray) -> str:
        """Text for 16 kHz mono audio ("" if there were no words in it)."""
        return self._model.recognize(audio, sample_rate=RATE).strip()


class WhisperEars:
    def __init__(self, model: str, compute_device: str = "cpu", language: str | None = None) -> None:
        from faster_whisper import WhisperModel

        compute_type = "int8" if compute_device == "cpu" else "float16"
        self._model = WhisperModel(model, device=compute_device, compute_type=compute_type)
        self._language = language

    def transcribe(self, audio: np.ndarray) -> str:
        """Text for 16 kHz mono audio ("" if there were no words in it)."""
        segments, _ = self._model.transcribe(
            audio, language=self._language, beam_size=1, vad_filter=True, condition_on_previous_text=False
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


def make_ears(
    engine: str, model: str = "", compute_device: str = "cpu", language: str | None = None
) -> ParakeetEars | WhisperEars:
    if engine not in DEFAULT_MODELS:
        raise ValueError(f"STT_ENGINE must be parakeet or whisper, not {engine!r}.")
    model = model or DEFAULT_MODELS[engine]
    return ParakeetEars(model) if engine == "parakeet" else WhisperEars(model, compute_device, language)


def save_recording(audio: np.ndarray, text: str) -> None:
    """Keep what the microphone heard next to what was understood, to track down recognition problems."""
    RECORDINGS_DIR.mkdir(exist_ok=True)
    name = time.strftime("%Y%m%d-%H%M%S")
    sf.write(RECORDINGS_DIR / f"{name}.wav", audio, RATE)
    with open(RECORDINGS_DIR / "transcripts.txt", "a", encoding="utf-8") as log:
        log.write(f"{name}.wav\t{text}\n")
