"""What each servo is doing right now: the data behind the control panel's wiring diagram.

Everything comes from the motors' joints, so the diagram follows when you calibrate a
servo's channel or angle range in the control panel.
"""

from __future__ import annotations

import threading

from .motors import JOINTS, Motors, Pose, pulse_width, rest_pose

MOVING_SECONDS = 0.25  # how long a servo stays lit after it last moved
MOVING_DEGREES = 0.2  # per frame; slower drifts (like breathing) don't count as moving


class ServoMonitor:
    """Fed every pose by the reflex thread; read by the control panel."""

    def __init__(self, motors: Motors) -> None:
        self.motors = motors
        self._last_angle: dict[str, float] = {}
        self._moving_until: dict[str, float] = {}
        self._lock = threading.Lock()
        self._snapshot: dict = {"pose": rest_pose(), "servos": []}
        self.moving: set[str] = set()  # joints that moved in the last moment

    def servos(self) -> list[dict]:
        """The fixed facts about each servo: channel, model, allowed angles and its calibration."""
        held = self.motors.held()
        return [
            {
                "name": name,
                "channel": joint.channel,
                "servo": joint.servo,
                "angle_min": min(joint.angle_min, joint.angle_max),
                "angle_max": max(joint.angle_min, joint.angle_max),
                "rest_angle": joint.to_angle(joint.rest),
                # As calibrated: "from" is the angle at the low end of the joint's range, so from > to is a reversed servo.
                "from": joint.angle_min,
                "to": joint.angle_max,
                "default": {"channel": JOINTS[name].channel, "from": JOINTS[name].angle_min, "to": JOINTS[name].angle_max},
                "held": held.get(name),
            }
            for name, joint in sorted(self.motors.joints.items(), key=lambda item: item[1].channel)
        ]

    def update(self, pose: Pose, now: float) -> None:
        servos, moving = [], set()
        for name, joint in sorted(self.motors.joints.items(), key=lambda item: item[1].channel):
            angle = self.motors.angles.get(name, joint.to_angle(pose.get(name, joint.rest)))
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
