import pytest
from dotenv import dotenv_values

from robot_head import devices
from robot_head.devices import Device, find_audio_device, save_settings


def test_saving_replaces_and_appends_settings_but_keeps_everything_else(tmp_path):
    env = tmp_path / ".env"
    env.write_bytes(b"# comment\r\nOPENROUTER_API_KEY=sk-or-secret\r\nMIC_DEVICE=\r\n")
    save_settings({"MIC_DEVICE": "Microphone (Realtek(R) Audio)", "CAMERA_INDEX": "2"}, env)
    assert env.read_bytes() == (
        b'# comment\r\nOPENROUTER_API_KEY=sk-or-secret\r\nMIC_DEVICE="Microphone (Realtek(R) Audio)"\r\n'
        b"CAMERA_INDEX=2\r\n"
    )


def test_saved_device_names_read_back_unchanged(tmp_path):
    env = tmp_path / ".env"
    name = 'Speakers (3- INZONE "Buds") \\ #1'
    save_settings({"SPEAKER_DEVICE": name}, env)
    assert dotenv_values(env)["SPEAKER_DEVICE"] == name


def test_audio_devices_are_found_by_part_of_their_name(monkeypatch):
    mics = [Device(1, "Headset Microphone (Oculus Virt"), Device(6, "Microphone (Realtek(R) Audio)")]
    monkeypatch.setattr(devices, "audio_devices", lambda kind: mics)
    assert find_audio_device("realtek", "input") == 6
    assert find_audio_device(None, "input") is None
    assert find_audio_device(3, "input") == 3
    with pytest.raises(ValueError, match="--choose-devices"):
        find_audio_device("usb mic", "input")
