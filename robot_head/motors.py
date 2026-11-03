"""Joint definitions and the two motor backends.

Every joint is driven with a normalized value (see ``Joint``). Only the motors know
about servo channels and angles, so nothing else changes when you swap the simulator
for real servos. ``JOINTS`` holds the defaults; the channels and angle ranges you
calibrate in the control panel are kept in robot.db and override them.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace

PULSE_MIN, PULSE_MAX = 750, 2250  # microseconds for 0..180 degrees, the common hobby servo range


def pulse_width(angle: float) -> int:
    return round(PULSE_MIN + angle / 180 * (PULSE_MAX - PULSE_MIN))


@dataclass(frozen=True)
class Joint:
    channel: int  # PCA9685 output
    angle_min: float  # servo angle at the low end of the range
    angle_max: float  # servo angle at the high end (swap the two to reverse a servo)
    low: float = -1.0
    high: float = 1.0
    rest: float = 0.0
    servo: str = "MG90S"  # suggested model: small metal-gear for the face, stronger for the neck

    def clamp(self, value: float) -> float:
        return min(self.high, max(self.low, value))

    def to_angle(self, value: float) -> float:
        t = (self.clamp(value) - self.low) / (self.high - self.low)
        return self.angle_min + t * (self.angle_max - self.angle_min)


# Placeholder angles: calibrate each servo in the control panel's Wiring tab before
# attaching any linkage. Channels run from the top of the head down, so the cables fan
# out without crossing.
JOINTS: dict[str, Joint] = {
    "lids": Joint(channel=0, angle_min=30, angle_max=110, low=0.0, high=1.0, rest=0.7),  # 0 shut .. 1 wide
    "eyes_y": Joint(channel=1, angle_min=70, angle_max=110),  # -1 down .. +1 up
    "eyes_x": Joint(channel=2, angle_min=60, angle_max=120),  # -1 left .. +1 right
    "jaw": Joint(channel=3, angle_min=90, angle_max=125, low=0.0, high=1.0, rest=0.0),  # 0 shut .. 1 open
    "neck_tilt": Joint(channel=4, angle_min=60, angle_max=120, servo="MG996R"),  # -1 down .. +1 up
    "neck_roll": Joint(channel=5, angle_min=70, angle_max=110, servo="MG996R"),  # -1 left shoulder .. +1 right
    "neck_pan": Joint(channel=6, angle_min=40, angle_max=140, servo="MG996R"),  # -1 robot's left .. +1 right
}

Pose = dict[str, float]


def rest_pose() -> Pose:
    return {name: joint.rest for name, joint in JOINTS.items()}


def calibrated(saved: dict[str, dict]) -> dict[str, Joint]:
    """The default joints with the channels and angle ranges saved in the control panel."""
    return {name: replace(joint, **saved.get(name, {})) for name, joint in JOINTS.items()}


class Motors:
    """Turns poses into servo angles; a backend only has to send an angle to a channel."""

    def __init__(self, joints: dict[str, Joint] = JOINTS) -> None:
        self._lock = threading.Lock()
        self.joints = dict(joints)
        self.angles: dict[str, float] = {}  # what each servo was sent last
        self._held: dict[str, float] = {}  # joints being calibrated: kept at an exact angle

    def set_joints(self, joints: dict[str, Joint]) -> None:
        with self._lock:
            for name, joint in joints.items():
                old = self.joints.get(name)
                if old is not None and old.channel != joint.channel:
                    self._send(old.channel, None)  # moved to another channel: let go of the old one
                    self.angles.pop(name, None)
            self.joints = dict(joints)

    def hold(self, name: str, angle: float | None) -> None:
        """Keep a joint at a raw servo angle whatever the pose says; None hands it back."""
        with self._lock:
            if angle is None:
                self._held.pop(name, None)
            else:
                self._held[name] = min(180.0, max(0.0, float(angle)))

    def held(self) -> dict[str, float]:
        with self._lock:
            return dict(self._held)

    def write(self, pose: Pose) -> None:
        with self._lock:
            for name, joint in self.joints.items():
                angle = self._held.get(name)
                if angle is None:
                    angle = joint.to_angle(pose.get(name, joint.rest))
                if self.angles.get(name) != angle:
                    self._send(joint.channel, angle)
                    self.angles[name] = angle

    def close(self) -> None:
        with self._lock:
            for joint in self.joints.values():
                self._send(joint.channel, None)

    def _send(self, channel: int, angle: float | None) -> None:
        """Move the servo on ``channel``; None stops its pulses so it goes limp."""


class SimulatedMotors(Motors):
    """Stands in for servos; the control panel draws the pose instead."""


class Pca9685Motors(Motors):
    """Hobby servos on a PCA9685 board over I2C (Raspberry Pi: ``uv sync --extra pi``)."""

    def __init__(self, joints: dict[str, Joint] = JOINTS, board: Pca9685 | None = None) -> None:
        self._board = board or Pca9685()
        self._failing = False
        super().__init__(joints)

    def _send(self, channel: int, angle: float | None) -> None:
        try:
            self._board.pulse(channel, None if angle is None else pulse_width(angle))
        except OSError as error:  # a loose wire: say so once, keep trying
            if not self._failing:
                print(f"Can't reach the servo board: {error}. Check the wires to the PCA9685.")
            self._failing = True
        else:
            self._failing = False


class Pca9685:
    """The PCA9685 16-channel PWM chip, spoken to directly over I2C."""

    MODE1, PRESCALE, LED0 = 0x00, 0xFE, 0x06
    RESTART, AUTO_INCREMENT, SLEEP = 0x80, 0x20, 0x10
    FULL_OFF = 0x10  # in a channel's highest byte: no pulses at all
    OSCILLATOR = 25_000_000  # Hz; real chips are off by a few percent, which calibrating absorbs

    def __init__(self, bus=None, address: int = 0x40, frequency: int = 50) -> None:
        if bus is None:
            from smbus2 import SMBus  # Raspberry Pi only: uv sync --extra pi

            bus = SMBus(1)  # I2C bus 1: Pi pins 3 (SDA) and 5 (SCL)
        self._bus, self._address = bus, address
        self._period = 1_000_000 / frequency  # microseconds
        self._write(self.MODE1, self.SLEEP)  # the pulse frequency can only be set while asleep
        self._write(self.PRESCALE, round(self.OSCILLATOR / (4096 * frequency)) - 1)
        self._write(self.MODE1, self.AUTO_INCREMENT)
        time.sleep(0.001)  # the oscillator needs half a millisecond to start
        self._write(self.MODE1, self.AUTO_INCREMENT | self.RESTART)

    def pulse(self, channel: int, microseconds: float | None) -> None:
        if microseconds is None:
            data = [0, 0, 0, self.FULL_OFF]
        else:
            ticks = round(microseconds / self._period * 4096)  # the pulse ends this many 4096ths into each period
            data = [0, 0, ticks & 0xFF, ticks >> 8]
        self._bus.write_i2c_block_data(self._address, self.LED0 + 4 * channel, data)

    def _write(self, register: int, value: int) -> None:
        self._bus.write_byte_data(self._address, register, value)
