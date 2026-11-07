import pytest
from test_control import robot  # noqa: F401  (fixture)

from robot_head.motors import JOINTS, Pca9685, Pca9685Motors, SimulatedMotors, calibrated, pulse_width
from robot_head.voice import voice_or_default
from robot_head.wiring import ServoMonitor


class FakeBus:
    """Records what would go over I2C to the PCA9685."""

    def __init__(self):
        self.bytes, self.blocks = [], []
        self.mode1 = 0x11  # what the chip wakes up with: asleep
        self.failures = 0  # sends that get no answer, like with a loose wire

    def write_byte_data(self, address, register, value):
        self.bytes.append((address, register, value))
        if register == 0x00:
            self.mode1 = value & 0x7F  # the restart bit clears itself

    def read_byte_data(self, address, register):
        return self.mode1

    def write_i2c_block_data(self, address, register, data):
        if self.failures:
            self.failures -= 1
            raise OSError(121, "Remote I/O error")
        self.blocks.append((address, register, list(data)))


def test_the_board_is_set_to_50_pulses_a_second():
    bus = FakeBus()
    Pca9685(bus)
    assert bus.bytes == [(0x40, 0x00, 0x10), (0x40, 0xFE, 121), (0x40, 0x00, 0x20), (0x40, 0x00, 0xA0)]


def test_servo_angles_become_pulses_on_their_channel():
    bus = FakeBus()
    motors = Pca9685Motors(board=Pca9685(bus))
    pose = {name: joint.rest for name, joint in JOINTS.items()}
    motors.write(pose)
    # The jaw (channel 3) rests shut at 90 degrees: 1500 microseconds, 307 of 4096 steps of a 20 ms period.
    assert (0x40, 0x06 + 4 * 3, [0, 0, 307 & 0xFF, 307 >> 8]) in bus.blocks
    assert len(bus.blocks) == len(JOINTS)
    motors.write(pose)
    assert len(bus.blocks) == len(JOINTS)  # nothing changed, nothing sent
    motors.write({**pose, "jaw": 1.0})
    assert bus.blocks[-1] == (0x40, 0x06 + 12, [0, 0, *divmod(round(pulse_width(125) / 20000 * 4096), 256)[::-1]])
    motors.close()
    assert bus.blocks[-1][2] == [0, 0, 0, 0x10]  # full off: the servo goes limp


def test_a_board_that_reset_itself_is_woken_up_and_sent_every_angle_again():
    bus = FakeBus()
    motors = Pca9685Motors(board=Pca9685(bus))
    motors.CHECK_EVERY = 0
    pose = {name: joint.rest for name, joint in JOINTS.items()}
    motors.write(pose)
    assert len(bus.blocks) == len(JOINTS) and motors.resets == 0
    bus.mode1 = 0x11  # its power dipped: it came back asleep, every servo off
    bus.bytes.clear()
    motors.write(pose)
    assert motors.resets == 1
    assert bus.bytes[1] == (0x40, 0xFE, 121) and bus.mode1 & 0x10 == 0  # 50 pulses a second and awake again
    assert len(bus.blocks) == 2 * len(JOINTS)  # every servo got its angle again, though none changed


def test_a_send_that_failed_is_tried_again():
    bus = FakeBus()
    motors = Pca9685Motors(board=Pca9685(bus))
    pose = {name: joint.rest for name, joint in JOINTS.items()}
    bus.failures = 1
    motors.write(pose)
    assert len(bus.blocks) == len(JOINTS) - 1
    motors.write(pose)
    assert len(bus.blocks) == len(JOINTS)


def test_a_servo_being_calibrated_ignores_the_head():
    motors = SimulatedMotors()
    monitor = ServoMonitor(motors)
    pose = {name: joint.rest for name, joint in JOINTS.items()}
    motors.hold("neck_pan", 200)
    motors.write(pose)
    assert motors.angles["neck_pan"] == 180.0  # never past the end of a servo
    monitor.update(pose, 0.0)
    pan = next(s for s in monitor.snapshot()["servos"] if s["name"] == "neck_pan")
    assert pan["angle"] == 180.0
    motors.hold("neck_pan", None)
    motors.write(pose)
    assert motors.angles["neck_pan"] == 90.0


def test_calibration_is_saved_and_applied(robot):  # noqa: F811
    motors = robot.reflexes.motors
    robot.save_servo("jaw", {"channel": 9, "from": 120, "to": 80})  # reversed: shut at 120, open at 80
    assert motors.joints["jaw"].channel == 9 and motors.joints["jaw"].to_angle(1.0) == 80
    assert calibrated(robot.store.servos())["jaw"].angle_min == 120
    with pytest.raises(ValueError, match="already used by lids"):
        robot.save_servo("jaw", {"channel": 0, "from": 90, "to": 120})
    with pytest.raises(ValueError, match="0 to 15"):
        robot.save_servo("jaw", {"channel": 16, "from": 90, "to": 120})
    with pytest.raises(ValueError, match="two different"):
        robot.save_servo("jaw", {"channel": 3, "from": 90, "to": 90})
    robot.reset_servo("jaw")
    assert motors.joints["jaw"] == JOINTS["jaw"]


def test_moving_a_servo_to_another_channel_lets_go_of_the_old_one():
    sent = []

    class Recording(SimulatedMotors):
        def _send(self, channel, angle):
            sent.append((channel, angle))
            return True

    motors = Recording()
    motors.write({})
    motors.set_joints(calibrated({"jaw": {"channel": 9}}))
    assert sent[-1] == (3, None)
    motors.write({})
    assert sent[-1][0] == 9


def test_a_voice_from_the_other_engine_falls_back_to_the_default():
    assert voice_or_default("piper", "am_michael") == "en_US-bryce-medium"
    assert voice_or_default("kokoro", "de_DE-thorsten-medium") == "am_michael"
    assert voice_or_default("piper", "de_DE-thorsten-medium") == "de_DE-thorsten-medium"
    assert voice_or_default("kokoro", "") == "am_michael"
