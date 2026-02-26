"""
recognizer.py — Gesture recognition engine
Detects Tap, Long Press, and Slide gestures from hand landmark sequences.
Uses a moving-average filter on finger tip positions to reduce noise.
"""

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

from tracker import HandLandmarks, LandmarkIndex


class GestureType(Enum):
    NONE = auto()
    TAP = auto()
    SLIDE_LEFT = auto()
    SLIDE_RIGHT = auto()


@dataclass
class GestureEvent:
    gesture: GestureType
    hand: str           # "Left" or "Right"
    confidence: float   # 0.0–1.0
    onset_ms: float = 0.0  # time from press detection to event firing (ms)


class _MovingAverage:
    """Simple moving average over the last `window` values."""
    def __init__(self, window: int = 5):
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> float:
        self._buf.append(value)
        return sum(self._buf) / len(self._buf)

    @property
    def value(self) -> float:
        return sum(self._buf) / len(self._buf) if self._buf else 0.0


class _FingerState:
    """Per-hand state machine tracking a single finger tip (INDEX_TIP by default)."""

    # Tunable thresholds
    TAP_VELOCITY_THRESHOLD = 0.025   # normalized units/frame — downward speed (lowered for higher sensitivity)
    SLIDE_DISPLACEMENT_X = 0.15      # normalized units — horizontal travel

    def __init__(self):
        self._avg_y = _MovingAverage(window=5)
        self._avg_x = _MovingAverage(window=5)
        self._prev_y: float | None = None
        self._prev_x: float | None = None

        # Tap
        self._pressed: bool = False
        self._press_time: float = 0.0

        # Slide
        self._slide_origin_x: float | None = None

    def update(self, hand: HandLandmarks) -> tuple[GestureType, float]:
        raw_x, raw_y, _ = hand.get(LandmarkIndex.INDEX_TIP)
        # Wrist Y used as reference to make detection scale-invariant
        _, wrist_y, _ = hand.get(LandmarkIndex.WRIST)
        _, index_mcp_y, _ = hand.get(LandmarkIndex.INDEX_MCP)

        smooth_y = self._avg_y.update(raw_y)
        smooth_x = self._avg_x.update(raw_x)

        gesture = GestureType.NONE
        onset_ms = 0.0

        # --- Tap detection via Y velocity ---
        if self._prev_y is not None:
            vel_y = smooth_y - self._prev_y   # positive = moving down (screen coords)

            if vel_y > self.TAP_VELOCITY_THRESHOLD and not self._pressed:
                self._pressed = True
                self._press_time = time.perf_counter()
                self._slide_origin_x = smooth_x

            elif self._pressed:
                if vel_y < -self.TAP_VELOCITY_THRESHOLD:
                    # Finger lifted → TAP; measure time from press detection
                    onset_ms = (time.perf_counter() - self._press_time) * 1000.0
                    gesture = GestureType.TAP
                    self._pressed = False
                    self._slide_origin_x = None

        # --- Slide detection via X displacement while pressed ---
        if self._pressed and self._slide_origin_x is not None:
            dx = smooth_x - self._slide_origin_x
            if abs(dx) > self.SLIDE_DISPLACEMENT_X:
                onset_ms = (time.perf_counter() - self._press_time) * 1000.0
                gesture = GestureType.SLIDE_RIGHT if dx > 0 else GestureType.SLIDE_LEFT
                self._pressed = False
                self._slide_origin_x = None

        self._prev_y = smooth_y
        self._prev_x = smooth_x
        return gesture, onset_ms


class GestureRecognizer:
    def __init__(self):
        # Separate state per hand label
        self._states: dict[str, _FingerState] = {}

    def update(self, hands: list[HandLandmarks]) -> list[GestureEvent]:
        events: list[GestureEvent] = []

        for hand in hands:
            label = hand.handedness
            if label not in self._states:
                self._states[label] = _FingerState()

            gesture, onset_ms = self._states[label].update(hand)
            if gesture != GestureType.NONE:
                events.append(GestureEvent(
                    gesture=gesture,
                    hand=label,
                    confidence=1.0,   # placeholder; refine in Phase 2
                    onset_ms=onset_ms,
                ))

        return events
