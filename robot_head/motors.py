"""Joint definitions and the two motor backends.

Every joint is driven with a normalized value (see ``Joint``). Only the
hardware backend knows about servo channels and angles, so nothing else
changes when you swap the simulator for real servos.
"""

from __future__ import annotations

from dataclasses import dataclass


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


# Placeholder angles. Before powering real servos, find each mechanism's safe
# limits one joint at a time and put them here. Channels run from the top of
# the head down, so the cables fan out without crossing.
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


class SimulatedMotors:
    """Stands in for servos; the control panel draws the pose instead."""

    def write(self, pose: Pose) -> None:
        pass

    def close(self) -> None:
        pass


class Pca9685Motors:
    """Hobby servos on a PCA9685 board over I2C (Raspberry Pi: ``uv sync --extra pi``)."""

    def __init__(self, joints: dict[str, Joint] = JOINTS) -> None:
        from adafruit_servokit import ServoKit  # only installable on the Pi

        self.joints = joints
        self.kit = ServoKit(channels=16)

    def write(self, pose: Pose) -> None:
        for name, value in pose.items():
            joint = self.joints.get(name)
            if joint is not None:
                self.kit.servo[joint.channel].angle = joint.to_angle(value)

    def close(self) -> None:
        for joint in self.joints.values():
            self.kit.servo[joint.channel].angle = None  # stop the pulses so the servos go limp
