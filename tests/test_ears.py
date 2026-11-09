import numpy as np
from test_control import robot  # noqa: F401  (fixture)

from robot_head.ears import RATE, SpeechCheck


class FakeVad:
    """Says which samples are speech, in the shape Silero's segmenter in onnx-asr does."""

    def __init__(self, segments):
        self.segments, self.calls = segments, []

    def segment_batch(self, waveforms, lengths, rate, **options):
        self.calls.append((waveforms.shape, int(lengths[0]), rate))
        yield iter(self.segments)


def test_a_sound_needs_some_speech_in_it_to_be_recognised():
    second = np.zeros(RATE, np.float32)
    assert SpeechCheck(FakeVad([(0, 1600), (8000, 12800)])).speech_seconds(second) == 0.4
    assert not SpeechCheck(FakeVad([(0, 1600)])).is_speech(second)  # 0.1 s: a tap, a cough
    assert SpeechCheck(FakeVad([(0, 4800)])).is_speech(second)  # 0.3 s: a short "Yeah." still counts
    vad = FakeVad([])
    assert not SpeechCheck(vad).is_speech(second)
    assert vad.calls == [((1, RATE), RATE, RATE)]
    assert SpeechCheck(FakeVad([])).speech_seconds(np.zeros(0, np.float32)) == 0.0


def test_the_noise_filter_can_be_switched_off(robot, monkeypatch):  # noqa: F811
    monkeypatch.setenv("NOISE_FILTER", "")  # put back afterwards: saving a setting also sets it in this process
    assert robot.noise_filter
    assert robot.update_settings({"NOISE_FILTER": "false"})["applied"] == ["NOISE_FILTER"]
    assert not robot.noise_filter
