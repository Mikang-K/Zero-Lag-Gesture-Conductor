"""
tracker.py — MediaPipe hand landmark extraction module (Tasks API)
Uses HandLandmarker (MediaPipe 0.10+) with float16 model.
Extracts 21 hand landmarks per hand with inference latency measurement.
"""

import time
from pathlib import Path
from typing import Any
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks
from mediapipe.tasks.python.vision.drawing_styles import (
    get_default_hand_landmarks_style,
    get_default_hand_connections_style,
)
from dataclasses import dataclass, field


_HAND_CONNECTIONS = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS


# MediaPipe hand landmark indices (for readability)
class LandmarkIndex:
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17


@dataclass
class HandLandmarks:
    """Normalized (0.0–1.0) x/y/z coordinates for 21 hand landmarks."""
    points: list[tuple[float, float, float]] = field(default_factory=list)
    handedness: str = "Unknown"

    def get(self, index: int) -> tuple[float, float, float]:
        return self.points[index]


@dataclass
class TrackResult:
    hands: list[HandLandmarks] = field(default_factory=list)
    inference_ms: float = 0.0
    # Raw NormalizedLandmark lists for drawing (one list per detected hand)
    _raw_landmarks: list[list[Any]] = field(default_factory=list, repr=False)


_DEFAULT_MODEL = Path(__file__).parent.parent / "models" / "hand_landmarker.task"


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        model_path: Path | None = None,
        use_gpu: bool = False,
    ):
        model_path = model_path or _DEFAULT_MODEL
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Run: curl -o models/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        delegate = (
            mp_python.BaseOptions.Delegate.GPU
            if use_gpu
            else mp_python.BaseOptions.Delegate.CPU
        )
        base_opts = mp_python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=delegate,
        )
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self._frame_ts_ms: int = 0

    def process(self, frame_bgr: np.ndarray) -> TrackResult:
        """Run HandLandmarker inference on a BGR frame. Returns TrackResult."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts_ms += 1   # must be strictly increasing in VIDEO mode

        t0 = time.perf_counter()
        detection = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        hands: list[HandLandmarks] = []
        raw_landmarks: list[list[Any]] = []

        for lm_list, handedness_list in zip(
            detection.hand_landmarks,
            detection.handedness,
        ):
            points = [(lm.x, lm.y, lm.z) for lm in lm_list]
            label = handedness_list[0].category_name
            hands.append(HandLandmarks(points=points, handedness=label))
            raw_landmarks.append(lm_list)   # list[NormalizedLandmark]

        return TrackResult(
            hands=hands,
            inference_ms=inference_ms,
            _raw_landmarks=raw_landmarks,
        )

    def draw(self, frame_bgr: np.ndarray, result: TrackResult) -> np.ndarray:
        """Draw landmark overlays onto frame in-place and return it."""
        for lm_list in result._raw_landmarks:
            draw_landmarks(
                frame_bgr,
                lm_list,
                _HAND_CONNECTIONS,
                get_default_hand_landmarks_style(),
                get_default_hand_connections_style(),
            )
        return frame_bgr

    def close(self):
        self._landmarker.close()
