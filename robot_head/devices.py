"""Find, choose and remember the microphone, speaker and camera."""

from __future__ import annotations

import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
from cv2_enumerate_cameras import enumerate_cameras, supported_backends

from .camera import CAMERA_BACKEND, open_capture
from .config import PROJECT_DIR, load_settings

ENV_FILE = PROJECT_DIR / ".env"
KIND_NAMES = {"input": "microphone", "output": "speaker"}
MIC_RATE = 16000  # the rate the ears record at, so the test proves the device can do it
PREVIEW_WINDOW = "camera check  (any key closes)"


@dataclass(frozen=True)
class Device:
    index: int
    name: str


def audio_devices(kind: str) -> list[Device]:
    """Microphones (kind "input") or speakers (kind "output").

    Windows lists every device once per audio API. Only the default API is
    used (MME on Windows), because it accepts any sample rate.
    """
    found = []
    for index in sd.query_hostapis(sd.default.hostapi)["devices"]:
        info = sd.query_devices(index)
        # "Sound Mapper" is Windows' alias for the default device, which is offered separately.
        if info[f"max_{kind}_channels"] > 0 and "Sound Mapper" not in info["name"]:
            found.append(Device(index, info["name"]))
    return found


def cameras() -> list[Device]:
    if CAMERA_BACKEND not in supported_backends:
        return []
    return [Device(camera.index, camera.name) for camera in enumerate_cameras(CAMERA_BACKEND)]


def find_audio_device(setting: int | str | None, kind: str) -> int | None:
    """Turn a MIC_DEVICE or SPEAKER_DEVICE setting (index, part of a name, or empty) into a device index."""
    if setting is None or isinstance(setting, int):
        return setting
    for device in audio_devices(kind):
        if setting.lower() in device.name.lower():
            return device.index
    raise ValueError(f"No {KIND_NAMES[kind]} matches {setting!r}. Run with --choose-devices to pick one.")


def describe_audio(index: int | None, kind: str) -> str:
    if index is not None:
        return sd.query_devices(index)["name"]
    default = sd.default.device[0 if kind == "input" else 1]
    return f"system default ({sd.query_devices(default)['name']})" if default >= 0 else "system default"


def describe_camera(index: int) -> str:
    return next((camera.name for camera in cameras() if camera.index == index), f"camera {index}")


def list_devices() -> None:
    for kind, setting in (("input", "MIC_DEVICE"), ("output", "SPEAKER_DEVICE")):
        default = sd.default.device[0 if kind == "input" else 1]
        print(f"{KIND_NAMES[kind].capitalize()}s (set {setting} to a name or part of one; empty = system default):")
        for device in audio_devices(kind):
            print(f"  {device.name}{'   <- system default' if device.index == default else ''}")
    print("Cameras (set CAMERA_INDEX to the number):")
    for camera in cameras():
        print(f"  {camera.index}) {camera.name}")


def save_settings(updates: dict[str, str], path: Path = ENV_FILE) -> None:
    """Write settings into .env, replacing their lines and leaving everything else (like the API key) untouched."""
    example = path.with_name(".env.example")
    if not path.exists() and example.exists():
        shutil.copyfile(example, path)
    text = ""
    if path.exists():
        with open(path, encoding="utf-8", newline="") as file:
            text = file.read()
    newline = "\r\n" if "\r\n" in text else "\n"
    for key, value in updates.items():
        line = f"{key}={_quote(value)}"
        pattern = re.compile(rf"^{key}=[^\r\n]*", re.MULTILINE)
        if pattern.search(text):
            text = pattern.sub(lambda match: line, text, count=1)
        else:
            text += ("" if not text or text.endswith("\n") else newline) + line + newline
    with open(path, "w", encoding="utf-8", newline="") as file:
        file.write(text)


def _quote(value: str) -> str:
    if value == "" or value.isdigit():
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def choose_devices() -> None:
    """Ask for each device in turn, try it out, and save the choices to .env."""
    settings = load_settings()
    updates = {}
    mic = _choose_audio("input", settings.mic_device, _test_microphone)
    if mic is not None:
        updates["MIC_DEVICE"] = mic
    speaker = _choose_audio("output", settings.speaker_device, _test_speaker)
    if speaker is not None:
        updates["SPEAKER_DEVICE"] = speaker
    camera = _choose_camera(settings.camera_index)
    if camera is not None:
        updates["CAMERA_INDEX"] = camera
    if updates:
        save_settings(updates, ENV_FILE)
        print(f"\nSaved to {ENV_FILE}.")
    else:
        print("\nNothing changed.")


def _choose_audio(kind: str, setting: int | str | None, test: Callable[[int | None], bool]) -> str | None:
    """The setting to save ("" for the system default), or None to keep the current one."""
    devices = audio_devices(kind)
    try:
        current = find_audio_device(setting, kind)
    except ValueError:
        current = None  # the saved device isn't plugged in any more
    while True:
        print(f"\n{KIND_NAMES[kind].capitalize()}s:")
        print(f"  {'*' if setting is None else ' '} 0) {describe_audio(None, kind)}")
        for number, device in enumerate(devices, 1):
            print(f"  {'*' if setting is not None and device.index == current else ' '} {number}) {device.name}")
        choice = _ask_number(list(range(len(devices) + 1)))
        if choice is None:
            return None
        device = devices[choice - 1] if choice else None
        if test(device.index if device else None):
            return device.name if device else ""


def _choose_camera(current: int) -> str | None:
    found = cameras()
    if not found:
        print("\nCouldn't list the cameras here; set CAMERA_INDEX in .env by hand (0 is usually the first one).")
        return None
    while True:
        # Cameras keep their own index as the number, since that is what CAMERA_INDEX holds.
        print("\nCameras:")
        for camera in found:
            print(f"  {'*' if camera.index == current else ' '} {camera.index}) {camera.name}")
        choice = _ask_number([camera.index for camera in found])
        if choice is None:
            return None
        if _test_camera(choice):
            return str(choice)


def _ask_number(choices: list[int]) -> int | None:
    while True:
        answer = input("Pick a number (Enter keeps the one marked *): ").strip()
        if not answer:
            return None
        if answer.isdigit() and int(answer) in choices:
            return int(answer)
        print("  Type one of the numbers shown.")


def _confirm(question: str) -> bool:
    return input(f"{question} [Y/n] ").strip().lower() in ("", "y", "yes")


def _test_microphone(device: int | None) -> bool:
    print("Say something for a few seconds; the bar should move.")
    try:
        with sd.InputStream(device=device, samplerate=MIC_RATE, channels=1, dtype="float32") as stream:
            end = time.monotonic() + 4
            while time.monotonic() < end:
                block, _ = stream.read(MIC_RATE // 20)
                level = min(1.0, float(np.sqrt(np.mean(np.square(block)))) * 15)
                print(f"\r  [{'#' * round(level * 30):<30}]", end="", flush=True)
    except sd.PortAudioError as error:
        print(f"  Couldn't record from it: {error}")
        return False
    print()
    return _confirm("Did the bar move when you spoke?")


def _test_speaker(device: int | None) -> bool:
    rate = 22050
    t = np.arange(int(rate * 0.25)) / rate
    chime = np.concatenate([0.3 * np.sin(2 * np.pi * hz * t) * np.hanning(len(t)) for hz in (660, 880)])
    try:
        sd.play(chime.astype(np.float32), rate, device=device)
        sd.wait()
    except sd.PortAudioError as error:
        print(f"  Couldn't play on it: {error}")
        return False
    return _confirm("Did you hear a chime?")


def _test_camera(index: int) -> bool:
    capture = open_capture(index)
    if capture is None:
        print("  Couldn't open it.")
        return False
    shown = False
    try:
        print("Showing the camera for a few seconds...")
        end = time.monotonic() + 5
        while time.monotonic() < end:
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.05)
                continue
            cv2.imshow(PREVIEW_WINDOW, frame)
            shown = True
            if cv2.waitKey(1) != -1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    if not shown:
        print("  It opened but sent no picture.")
        return False
    return _confirm("Was that the right camera?")
