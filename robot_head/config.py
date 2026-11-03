"""Settings, read from environment variables or the .env file next to pyproject.toml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "anthropic/claude-haiku-5.5"


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _flag(name: str, default: bool) -> bool:
    value = _get(name)
    return value.lower() in ("1", "true", "yes", "on") if value else default


def _audio_device(value: str) -> int | str | None:
    """Audio devices can be given by index or by part of their name."""
    if not value:
        return None
    return int(value) if value.isdigit() else value


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    model: str
    reasoning_effort: str
    camera_index: int
    camera_flip: bool
    send_camera_frames: bool
    stt_engine: str
    stt_model: str
    stt_language: str | None
    stt_device: str
    save_recordings: bool
    mic_device: int | str | None
    speaker_device: int | str | None
    silence_seconds: float
    tts_engine: str
    tts_voice: str
    tts_speed: float
    wake_name: str
    follow_up_seconds: float
    remark_after_seconds: float
    barge_in: bool
    motors: str
    web_ui: bool
    web_host: str
    web_port: int


def load_settings() -> Settings:
    load_dotenv(PROJECT_DIR / ".env")
    return Settings(
        openrouter_api_key=_get("OPENROUTER_API_KEY"),
        model=_get("OPENROUTER_MODEL", DEFAULT_MODEL),
        reasoning_effort=_get("REASONING_EFFORT"),
        camera_index=int(_get("CAMERA_INDEX", "0")),
        camera_flip=_flag("CAMERA_FLIP", False),
        send_camera_frames=_flag("SEND_CAMERA_FRAMES", True),
        stt_engine=_get("STT_ENGINE", "parakeet").lower(),
        stt_model=_get("STT_MODEL"),  # empty picks the engine's default model
        stt_language=_get("STT_LANGUAGE") or None,
        stt_device=_get("STT_DEVICE", "cpu"),
        save_recordings=_flag("SAVE_RECORDINGS", False),
        mic_device=_audio_device(_get("MIC_DEVICE")),
        speaker_device=_audio_device(_get("SPEAKER_DEVICE")),
        silence_seconds=float(_get("SILENCE_SECONDS", "0.8")),
        tts_engine=_get("TTS_ENGINE", "kokoro").lower(),
        tts_voice=_get("TTS_VOICE"),  # empty picks the engine's default voice
        tts_speed=float(_get("TTS_SPEED", "1.0")),
        wake_name=_get("WAKE_NAME"),
        follow_up_seconds=float(_get("FOLLOW_UP_SECONDS", "20")),
        remark_after_seconds=float(_get("REMARK_AFTER_SECONDS", "0")),
        barge_in=_flag("BARGE_IN", True),
        motors=_get("MOTORS", "sim"),
        web_ui=_flag("WEB_UI", True),
        web_host=_get("WEB_HOST", "127.0.0.1"),
        web_port=int(_get("WEB_PORT", "8765")),
    )
