"""What each servo is doing right now: the data behind the control panel's wiring diagram.

Everything comes from ``JOINTS`` in motors.py, so the diagram follows when you change a
channel or an angle range there.
"""

from __future__ import annotations

import threading

from .motors import JOINTS, Joint, Pose, rest_pose

MOVING_SECONDS = 0.25  # how long a servo stays lit after it last moved
MOVING_DEGREES = 0.2  # per frame; slower drifts (like breathing) don't count as moving
PULSE_MIN, PULSE_MAX = 750, 2250  # microseconds for 0..180 degrees, Adafruit ServoKit's default


def pulse_width(angle: float) -> int:
    return round(PULSE_MIN + angle / 180 * (PULSE_MAX - PULSE_MIN))


class ServoMonitor:
    """Fed every pose by the reflex thread; read by the control panel."""

    def __init__(self, joints: dict[str, Joint] = JOINTS) -> None:
        self._joints = sorted(joints.items(), key=lambda item: item[1].channel)
        self._last_angle: dict[str, float] = {}
        self._moving_until: dict[str, float] = {}
        self._lock = threading.Lock()
        self._snapshot: dict = {"pose": rest_pose(), "servos": []}
        self.moving: set[str] = set()  # joints that moved in the last moment

    def servos(self) -> list[dict]:
        """The fixed facts about each servo: channel, model and allowed angles."""
        return [
            {
                "name": name,
                "channel": joint.channel,
                "servo": joint.servo,
                "angle_min": min(joint.angle_min, joint.angle_max),
                "angle_max": max(joint.angle_min, joint.angle_max),
                "rest_angle": joint.to_angle(joint.rest),
            }
            for name, joint in self._joints
        ]

    def update(self, pose: Pose, now: float) -> None:
        servos, moving = [], set()
        for name, joint in self._joints:
            angle = joint.to_angle(pose.get(name, joint.rest))
            if abs(angle - self._last_angle.get(name, angle)) > MOVING_DEGREES:
                self._moving_until[name] = now + MOVING_SECONDS
            self._last_angle[name] = angle
            if now < self._moving_until.get(name, 0.0):
                moving.add(name)
            servos.append({"name": name, "angle": round(angle, 1), "pulse": pulse_width(angle), "moving": name in moving})
        with self._lock:
            self.moving = moving
            self._snapshot = {"pose": {k: round(v, 3) for k, v in pose.items()}, "servos": servos}

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot
