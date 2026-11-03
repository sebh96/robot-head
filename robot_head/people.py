"""Telling people apart by their faces.

OpenCV's SFace model turns a face into 128 numbers; two pictures of the same person give
similar numbers. Each person keeps a handful of these "fingerprints" from different angles
and light (in robot.db, never as pictures). A face nobody matches has to show up a few times
in a row before it counts as someone new, so one blurry frame doesn't create a stranger.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

from .downloads import MODELS_DIR, fetch
from .store import Store

SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
LOOK_EVERY = 1.0  # seconds between recognitions
MIN_FACE_PIXELS = 60  # smaller faces are too blurry to recognize
MIN_SCORE = 0.8  # detector confidence needed to trust a face
NEW_PERSON_SIGHTINGS = 3  # an unknown face must show up this many looks in a row to count as someone new
MAX_SAMPLES = 12  # fingerprints kept per person
NEW_SAMPLE_BELOW = 0.8  # only keep another sample when it adds something (a new angle or light)
GONE_AFTER = 5.0  # seconds without seeing someone before they no longer count as there
LAST_SEEN_EVERY = 30.0  # how often "last seen" is written to the database


def normalized(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32).ravel()
    return vector / (np.linalg.norm(vector) or 1.0)


class People:
    def __init__(self, store: Store, camera=None, match: float = 0.4, recognizer=None) -> None:
        self.store = store
        self.camera = camera
        self.match = match  # cosine similarity needed to count as the same person
        self.enabled = True
        self.on_new_person: Callable[[int], None] | None = None
        self._recognizer = recognizer or cv2.FaceRecognizerSF.create(
            str(fetch(SFACE_URL, MODELS_DIR / "face_recognition_sface_2021dec.onnx", "the face recognizer (37 MB)")), ""
        )
        self._lock = threading.Lock()
        self._known = [(person, normalized(vector)) for person, vector in store.faces()]
        self._unknown: list[np.ndarray] = []
        self._present: list[int] = []
        self._seen_at = 0.0
        self._written: dict[int, float] = {}

    def present(self) -> list[int]:
        """Who is in view, closest first (empty when nobody has been recognized for a moment)."""
        if time.monotonic() - self._seen_at > GONE_AFTER:
            return []
        return list(self._present)

    def closest(self) -> int | None:
        present = self.present()
        return present[0] if present else None

    def label(self, person_id: int | None) -> str:
        person = self.store.person(person_id) if person_id is not None else None
        return person["label"] if person else "someone"

    def forget(self, person_id: int) -> None:
        """Drop a person's fingerprints from memory too, after they were deleted or merged."""
        with self._lock:
            self._known = [(p, v) for p, v in self._known if p != person_id]
            self._present = [p for p in self._present if p != person_id]

    def reload(self) -> None:
        with self._lock:
            self._known = [(person, normalized(vector)) for person, vector in self.store.faces()]

    def run(self, stop: threading.Event) -> None:
        while not stop.wait(LOOK_EVERY):
            if not self.enabled:
                continue
            sample = self.camera.faces_sample()
            if sample is not None:
                self.look(*sample, now=time.time())

    def look(self, frame: np.ndarray, faces: list[np.ndarray], now: float) -> list[int]:
        present = []
        for face in faces[:4]:
            if face[2] < MIN_FACE_PIXELS or face[14] < MIN_SCORE:
                continue
            fingerprint = normalized(self._recognizer.feature(self._recognizer.alignCrop(frame, face)))
            person = self.identify(fingerprint, now)
            if person is not None and person not in present:
                present.append(person)
        if present:
            self._present = present
            self._seen_at = time.monotonic()
        return present

    def identify(self, fingerprint: np.ndarray, now: float) -> int | None:
        """Who this face belongs to; a new person once an unknown face has shown up often enough."""
        with self._lock:
            best_person, best = None, -1.0
            for person, known in self._known:
                similarity = float(known @ fingerprint)
                if similarity > best:
                    best_person, best = person, similarity
        if best_person is not None and best >= self.match:
            self._unknown = []
            if best < NEW_SAMPLE_BELOW and sum(p == best_person for p, _ in self._known) < MAX_SAMPLES:
                self._remember(best_person, fingerprint)
            if now - self._written.get(best_person, 0.0) > LAST_SEEN_EVERY:
                self.store.saw_person(best_person, now)
                self._written[best_person] = now
            return best_person
        # Nobody we know. Count it only if it keeps looking like the same new face.
        self._unknown = [f for f in self._unknown if float(f @ fingerprint) >= self.match] + [fingerprint]
        if len(self._unknown) < NEW_PERSON_SIGHTINGS:
            return None
        person = self.store.add_person(now)
        for sample in self._unknown:
            self._remember(person, sample)
        self._unknown = []
        if self.on_new_person:
            self.on_new_person(person)
        return person

    def _remember(self, person: int, fingerprint: np.ndarray) -> None:
        self.store.add_face(person, fingerprint)
        with self._lock:
            self._known.append((person, fingerprint))
