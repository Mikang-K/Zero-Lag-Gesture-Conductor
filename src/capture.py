"""
capture.py — Webcam capture module
Handles frame acquisition from camera with latency measurement.

Two capture modes are supported via the `threaded` constructor flag:

  threaded=False  (legacy)
    Calls cap.read() in the main thread.  The call blocks until the camera
    delivers the next frame (~1/FPS ≈ 16 ms at 60 fps).  Reported latency
    is the grab+decode wall-clock time.

  threaded=True  (default — rhythm-game optimised)
    A background daemon thread continuously calls cap.read() and stores the
    most recently decoded frame in a single-slot queue.  The main thread's
    read() call returns instantly with the freshest available frame; reported
    latency is *frame staleness* (current_time − grab_time), which reflects
    the true age of the image fed to the recognition pipeline.

    Expected capture latency: <3 ms vs. ~30 ms in blocking mode.
    This brings E2E pipeline latency within one 60-fps frame (~16 ms).
"""

import queue
import threading
import time

import cv2
import numpy as np


class CameraCapture:
    def __init__(
        self,
        camera_index: int = 0,
        target_fps: int = 60,
        threaded: bool = True,
    ):
        self.camera_index = camera_index
        self.target_fps = target_fps
        self._threaded = threaded
        self.cap: cv2.VideoCapture | None = None

        # Threaded-mode internals
        # Queue holds at most one (frame, grab_timestamp) tuple.
        # The grab thread always replaces stale frames so the consumer
        # never processes an outdated image.
        self._frame_queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=1)
        self._grab_thread: threading.Thread | None = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        # Keep driver buffer to 1 frame so we never read stale hardware frames.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self._threaded:
            self._running = True
            self._grab_thread = threading.Thread(
                target=self._grab_loop, daemon=True, name="CamGrabThread"
            )
            self._grab_thread.start()

        return True

    def release(self):
        if self._threaded:
            self._running = False
            if self._grab_thread is not None:
                self._grab_thread.join(timeout=1.0)
                self._grab_thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ------------------------------------------------------------------
    # Background grab loop (threaded mode only)
    # ------------------------------------------------------------------

    def _grab_loop(self):
        """Continuously decode frames and keep only the latest one.

        Using a maxsize-1 queue with a non-blocking put:
          - If the consumer is slower than the camera, old frames are
            dropped so the pipeline always works on the freshest image
            (critical for rhythm-game responsiveness).
          - If the consumer is faster than the camera, it blocks briefly
            on queue.get(timeout=...) — identical to blocking read().
        """
        while self._running:
            if self.cap is None:
                break

            t_grab = time.perf_counter()
            ret, frame = self.cap.read()

            if not ret:
                continue

            # Evict the pending frame (if any) and push the new one.
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self._frame_queue.put_nowait((frame, t_grab))
            except queue.Full:
                pass   # race: consumer just put something back; harmless

    # ------------------------------------------------------------------
    # Public read API (same signature as Phase 1)
    # ------------------------------------------------------------------

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """Return (success, frame, latency_ms).

        In threaded mode, latency_ms is the frame staleness — how long
        the image has been waiting in the queue before the caller consumes
        it.  This is the metric that matters for end-to-end rhythm-game lag.

        In blocking mode, latency_ms is the grab+decode wall time.
        """
        if not self._threaded:
            return self._read_blocking()
        return self._read_threaded()

    def _read_blocking(self) -> tuple[bool, np.ndarray | None, float]:
        """Legacy path: blocking cap.read() in the caller's thread."""
        if self.cap is None:
            return False, None, 0.0

        t0 = time.perf_counter()
        ret, frame = self.cap.read()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not ret:
            return False, None, latency_ms
        return True, frame, latency_ms

    def _read_threaded(self) -> tuple[bool, np.ndarray | None, float]:
        """Fast path: pull the pre-grabbed frame from the queue."""
        try:
            frame, t_grab = self._frame_queue.get(timeout=0.05)
            latency_ms = (time.perf_counter() - t_grab) * 1000.0
            return True, frame, latency_ms
        except queue.Empty:
            return False, None, 0.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return "threaded" if self._threaded else "blocking"
