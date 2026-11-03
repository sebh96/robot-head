import time

import numpy as np

from robot_head.cues import SentenceStream, parse_segment
from robot_head.expressions import from_settings
from robot_head.motors import JOINTS, SimulatedMotors
from robot_head.reflexes import Reflexes
from robot_head.voice import envelope
from robot_head.wiring import ServoMonitor, pulse_width


def spoken(segments):
    return [(s.text, s.cues) for s in segments]


def test_sentences_are_released_as_they_complete():
    stream = SentenceStream()
    assert stream.feed("[happy] Hel") == []
    assert spoken(stream.feed("lo there! How a")) == [("Hello there!", ["happy"])]
    assert stream.feed("re you?") == []
    assert spoken(stream.flush()) == [("How are you?", [])]


def test_bracketed_text_and_stage_directions_are_not_spoken():
    # Unknown cues like [laughs] are passed on and simply ignored by the head.
    assert spoken([parse_segment("*smiles* Sure [laughs] thing [Look Left].")]) == [("Sure thing.", ["laughs", "look_left"])]


def test_a_cue_with_settings_invents_an_expression():
    segment = parse_segment("[disgusted lids=0.3 look=0, -0.2] Ew, bro.")
    assert (segment.text, segment.cues) == ("Ew, bro.", ["disgusted"])
    assert segment.new_expressions == {"disgusted": {"lids": "0.3", "look": "0,-0.2"}}
    expression = from_settings("disgusted", {"lids": "5", "look": "0,-0.2", "roll": "wobbly"})
    assert (expression.lids, expression.look_y, expression.roll, expression.source) == (1.0, -0.2, 0.0, "llm")


def test_a_cue_after_the_last_sentence_still_fires():
    stream = SentenceStream()
    assert spoken(stream.feed("Done. [nod]")) == [("Done.", [])]
    assert spoken(stream.flush()) == [("", ["nod"])]


def test_a_long_opening_clause_is_spoken_before_its_sentence_ends():
    stream = SentenceStream()
    assert spoken(stream.feed("[smug] Okay, so you're rocking a big straw hat, ")) == [
        ("Okay, so you're rocking a big straw hat,", ["smug"])
    ]
    # Only the first clause of a reply is split off; after that it's whole sentences again.
    assert spoken(stream.feed("and the feather is too much. Also, honestly, bro, ")) == [("and the feather is too much.", [])]


def test_cue_settings_dont_count_as_words_of_the_opening_clause():
    assert SentenceStream().feed("[disgusted lids=0.3 tilt=0.1 roll=-0.2] Bro, console ") == []


def test_decimal_numbers_do_not_end_a_sentence():
    assert SentenceStream().feed("It costs 3.50 euros") == []


def test_joint_angles_are_clamped_to_the_servo_range():
    jaw = JOINTS["jaw"]
    assert jaw.to_angle(-1) == jaw.angle_min
    assert jaw.to_angle(5) == jaw.angle_max


def test_reflexes_keep_every_joint_in_range():
    reflexes = Reflexes(SimulatedMotors())
    for cue in ("smug", "roll_eyes", "nod", "look_left", "tilt_right"):
        reflexes.cue(cue)
    reflexes.set_jaw(3.0)
    start = time.monotonic()
    for i in range(300):
        face = (1.0, -1.0) if i % 50 < 25 else None
        pose = reflexes.step(start + i / 30, 1 / 30, face)
        assert set(pose) == set(JOINTS)
        for name, value in pose.items():
            assert JOINTS[name].low <= value <= JOINTS[name].high, name


def test_wiring_lights_up_only_the_servos_that_move():
    motors = SimulatedMotors()
    monitor = ServoMonitor(motors)

    def show(pose, now):
        motors.write(pose)
        monitor.update(pose, now)

    pose = {name: joint.rest for name, joint in JOINTS.items()}
    show(pose, 0.0)
    show({**pose, "jaw": 0.8}, 0.03)
    assert monitor.moving == {"jaw"}
    jaw = next(s for s in monitor.snapshot()["servos"] if s["name"] == "jaw")
    assert (jaw["moving"], jaw["angle"], jaw["pulse"]) == (True, 118.0, pulse_width(118.0))
    show({**pose, "jaw": 0.8}, 0.5)  # holding still lets it go dark again
    assert monitor.moving == set()
    assert [s["channel"] for s in monitor.servos()] == sorted(joint.channel for joint in JOINTS.values())


def test_servo_pulse_widths_span_the_servokit_range():
    assert (pulse_width(0), pulse_width(90), pulse_width(180)) == (750, 1500, 2250)


def test_jaw_envelope_follows_loudness():
    rate = 16000
    silence = np.zeros(rate // 2, dtype=np.float32)
    tone = 0.5 * np.sin(np.linspace(0, 2000, rate // 2, dtype=np.float32))
    levels = envelope(np.concatenate([silence, tone]), rate)
    assert len(levels) == 50
    assert levels[:24].max() == 0
    assert levels[-24:].min() > 0.5
