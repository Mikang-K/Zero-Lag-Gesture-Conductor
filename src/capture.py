"""
capture.py — Webcam capture module
Handles frame acquisition from camera with latency measurement.
"""

import time
import cv2
import numpy as np


class CameraCapture:
    def __init__(self, camera_index: int = 0, target_fps: int = 60):
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        # Minimize internal buffer to reduce capture latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """
        Returns (success, frame, capture_latency_ms).
        capture_latency_ms measures only the frame grab time.
        """
        if self.cap is None:
            return False, None, 0.0

        t0 = time.perf_counter()
        ret, frame = self.cap.read()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not ret:
            return False, None, latency_ms

        return True, frame, latency_ms

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
