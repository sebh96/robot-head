"""Webcam capture on a background thread, with face detection for the reflex layer."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras

from .downloads import MODELS_DIR, fetch

FACE_TIMEOUT = 0.5  # seconds a face may go undetected before tracking lets go
DETECT_WIDTH = 320  # detect on a downscaled frame; plenty for faces at conversation distance
FACE_CONFIDENCE = 0.6  # YuNet's default of 0.9 misses real faces in small frames
FACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

if sys.platform == "win32":
    CAMERA_BACKEND = cv2.CAP_DSHOW  # opens much faster than Windows' default backend
elif sys.platform == "darwin":
    CAMERA_BACKEND = cv2.CAP_AVFOUNDATION
else:
    CAMERA_BACKEND = cv2.CAP_V4L2


def open_capture(index: int) -> cv2.VideoCapture | None:
    """Open camera ``index`` as numbered by ``CAMERA_BACKEND``, or return None.

    On Windows some virtual cameras (newer DroidCam versions, for one) are listed by
    DirectShow but only open through Media Foundation, so retry there by name.
    """
    capture = cv2.VideoCapture(index, CAMERA_BACKEND)
    if capture.isOpened():
        return capture
    capture.release()
    if sys.platform == "win32":
        name = next((camera.name for camera in enumerate_cameras(cv2.CAP_DSHOW) if camera.index == index), None)
        for camera in enumerate_cameras(cv2.CAP_MSMF):
            if camera.name == name:
                capture = cv2.VideoCapture(camera.index, cv2.CAP_MSMF)
                if capture.isOpened():
                    return capture
                capture.release()
    return None


def face_model() -> Path:
    """OpenCV's YuNet face detector."""
    return fetch(FACE_MODEL_URL, MODELS_DIR / "face_detection_yunet_2023mar.onnx", "the face detector (230 KB)")


class Camera:
    def __init__(self, index: int = 0, flip: bool = False) -> None:
        capture = open_capture(index)
        if capture is None:
            raise RuntimeError(f"Could not open camera {index}. Pick another with --choose-devices, or use --no-camera.")
        self._cap = capture
        self.flip = flip  # mirror the image, for cameras that see the world the other way round
        self._detector = cv2.FaceDetectorYN.create(
            str(face_model()), "", (DETECT_WIDTH, DETECT_WIDTH), score_threshold=FACE_CONFIDENCE
        )
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._face_box: tuple[int, int, int, int] | None = None
        self._faces: list[np.ndarray] = []
        self._face_seen = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def face(self) -> tuple[float, float] | None:
        """Where the nearest face is, from -1 to 1 on each axis (x right, y up), or None."""
        with self._lock:
            if self._frame is None or self._face_box is None or time.monotonic() - self._face_seen > FACE_TIMEOUT:
                return None
            x, y, w, h = self._face_box
            height, width = self._frame.shape[:2]
        return 2 * (x + w / 2) / width - 1, 1 - 2 * (y + h / 2) / height

    def preview_jpeg(self, width: int = 480) -> bytes | None:
        """The latest frame for the control panel: the tracked face boxed, mirrored like a selfie camera."""
        with self._lock:
            frame = self._frame
            box = self._face_box if time.monotonic() - self._face_seen <= FACE_TIMEOUT else None
        if frame is None:
            return None
        frame = frame.copy()
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 220, 80), 3)
        height = round(width * frame.shape[0] / frame.shape[1])
        frame = cv2.flip(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA), 1)
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return jpeg.tobytes() if ok else None

    def latest_frame(self) -> np.ndarray | None:
        """The newest frame (frames are replaced, never changed, so no copy is needed)."""
        with self._lock:
            return self._frame

    def snapshot_jpeg(self, max_side: int = 512) -> bytes | None:
        """The latest frame as a small JPEG for the LLM."""
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        scale = max_side / max(frame.shape[:2])
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return jpeg.tobytes() if ok else None

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1)
        self._cap.release()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            if self.flip:
                frame = cv2.flip(frame, 1)
            faces = self._detect(frame)
            with self._lock:
                self._frame = frame
                if faces:
                    self._faces = faces
                    self._face_box = tuple(int(v) for v in faces[0][:4])
                    self._face_seen = time.monotonic()

    def faces_sample(self) -> tuple[np.ndarray, list[np.ndarray]] | None:
        """The newest frame and the faces in it, biggest (closest) first, for recognizing who's there.
        Each face is YuNet's row: box, five landmarks and the detection score, in frame pixels."""
        with self._lock:
            if self._frame is None or not self._faces or time.monotonic() - self._face_seen > FACE_TIMEOUT:
                return None
            return self._frame, self._faces

    def _detect(self, frame: np.ndarray) -> list[np.ndarray]:
        scale = DETECT_WIDTH / frame.shape[1]
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        self._detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._detector.detect(small)
        if faces is None or len(faces) == 0:
            return []
        rows = []
        for face in sorted(faces, key=lambda f: f[2] * f[3], reverse=True):  # the biggest face is the closest person
            row = face.copy()
            row[:14] /= scale  # box and landmarks back to full-frame pixels; the score stays
            rows.append(row)
        return rows
