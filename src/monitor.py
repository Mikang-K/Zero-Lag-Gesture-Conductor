"""
monitor.py — Real-time performance monitoring module
Tracks FPS, per-stage latency, and renders an on-screen HUD overlay.
Also saves per-frame benchmark data to a CSV for Phase 2 analysis.
"""

import csv
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameMetrics:
    timestamp: float = 0.0
    capture_ms: float = 0.0
    inference_ms: float = 0.0
    recognize_ms: float = 0.0
    emulate_ms: float = 0.0
    total_ms: float = 0.0    # pipeline latency (capture+inference+recognize+emulate)
    onset_ms: float = 0.0    # gesture structural delay (press→fire); non-zero only on gesture frames
    fps: float = 0.0


class PerformanceMonitor:
    _HUD_COLOR = (0, 255, 0)      # Green text
    _WARN_COLOR = (0, 165, 255)   # Orange text (when latency is high)
    _LATENCY_WARN_MS = 33.0       # > 33 ms (~<30 FPS equivalent) triggers orange

    def __init__(
        self,
        fps_window: int = 30,
        csv_path: Path | None = None,
    ):
        self._frame_times: deque[float] = deque(maxlen=fps_window)
        self._last_time: float = time.perf_counter()

        # CSV logging
        if csv_path is None:
            csv_path = Path(__file__).parent.parent / "benchmarks" / "phase1_metrics.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=list(FrameMetrics.__dataclass_fields__.keys()),
        )
        self._csv_writer.writeheader()

        self._current: FrameMetrics = FrameMetrics()

    # ------------------------------------------------------------------
    # Per-frame update API
    # ------------------------------------------------------------------

    def begin_frame(self):
        """Call at the very start of each frame loop iteration."""
        now = time.perf_counter()
        self._frame_times.append(now - self._last_time)
        self._last_time = now
        self._current = FrameMetrics(timestamp=now)

    def record(
        self,
        capture_ms: float = 0.0,
        inference_ms: float = 0.0,
        recognize_ms: float = 0.0,
        emulate_ms: float = 0.0,
        onset_ms: float = 0.0,
    ):
        """Populate latency fields for the current frame."""
        self._current.capture_ms = capture_ms
        self._current.inference_ms = inference_ms
        self._current.recognize_ms = recognize_ms
        self._current.emulate_ms = emulate_ms
        self._current.total_ms = capture_ms + inference_ms + recognize_ms + emulate_ms
        self._current.onset_ms = onset_ms
        self._current.fps = self.fps

    def commit(self):
        """Write the current frame's metrics to CSV."""
        self._csv_writer.writerow(asdict(self._current))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        if not self._frame_times:
            return 0.0
        avg_dt = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    @property
    def metrics(self) -> FrameMetrics:
        return self._current

    # ------------------------------------------------------------------
    # HUD rendering
    # ------------------------------------------------------------------

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Render latency / FPS overlay onto the frame in-place."""
        m = self._current
        warn = m.total_ms > self._LATENCY_WARN_MS

        perceived = m.onset_ms + m.total_ms
        lines = [
            f"FPS:       {m.fps:5.1f}",
            f"Capture:   {m.capture_ms:5.1f} ms",
            f"Inference: {m.inference_ms:5.1f} ms",
            f"Recognize: {m.recognize_ms:5.1f} ms",
            f"Emulate:   {m.emulate_ms:5.1f} ms",
            f"Pipeline:  {m.total_ms:5.1f} ms" + (" !" if warn else ""),
            f"Onset:     {m.onset_ms:5.1f} ms",
            f"Perceived: {perceived:5.1f} ms",
        ]

        x, y0, dy = 10, 30, 22
        color = self._WARN_COLOR if warn else self._HUD_COLOR
        for i, line in enumerate(lines):
            cv2.putText(
                frame, line,
                (x, y0 + i * dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA,
            )
        return frame

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        self._csv_file.flush()
        self._csv_file.close()
