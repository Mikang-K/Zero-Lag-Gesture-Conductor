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
    """Per-hand state machine tracking a single finger tip (INDEX_TIP by default).

    Rhythm-game optimised: TAP fires on the DOWNSTROKE (finger moving down),
    not on release. This eliminates the press-to-release structural delay so
    that the input event reaches the game at the earliest possible moment.

    State machine:
      UP   → downstroke detected  → fire TAP immediately, enter DOWN state
      DOWN → upstroke detected    → re-enter UP state (ready for next tap)
           → horizontal travel    → fire SLIDE, re-enter UP state
           → cooldown expired     → force re-enter UP state (finger held down)
    """

    # --- Tunable thresholds ---
    # Downward velocity (normalized units/frame) to register a press.
    # Lowering this increases sensitivity but also false-positive rate.
    TAP_VELOCITY_THRESHOLD = 0.025

    # Upward velocity to register a release (gentler than press so a soft
    # lift still resets the state and allows the next tap).
    RELEASE_VELOCITY_THRESHOLD = 0.010

    # Minimum time between consecutive TAP events on the same finger.
    # 100 ms → max ~BPM 600, well above any rhythm game requirement.
    # Prevents a single slow downstroke from firing multiple frames.
    TAP_COOLDOWN_S = 0.10

    # Horizontal displacement (normalized) required to register a slide
    # while the finger is in the DOWN state.
    SLIDE_DISPLACEMENT_X = 0.15

    def __init__(self):
        self._avg_y = _MovingAverage(window=5)
        self._avg_x = _MovingAverage(window=5)
        self._prev_y: float | None = None
        self._prev_x: float | None = None

        # DOWN state flag and the timestamp when we entered it
        self._pressed: bool = False
        self._press_time: float = 0.0

        # X position at the moment the finger went down (slide origin)
        self._slide_origin_x: float | None = None

    def update(self, hand: HandLandmarks) -> tuple[GestureType, float]:
        raw_x, raw_y, _ = hand.get(LandmarkIndex.INDEX_TIP)
        smooth_y = self._avg_y.update(raw_y)
        smooth_x = self._avg_x.update(raw_x)

        gesture = GestureType.NONE
        onset_ms = 0.0

        if self._prev_y is not None:
            vel_y = smooth_y - self._prev_y   # positive = moving down (screen coords)
            now = time.perf_counter()
            cooldown_elapsed = (now - self._press_time) >= self.TAP_COOLDOWN_S

            if not self._pressed:
                # --- UP state: wait for downstroke ---
                if vel_y > self.TAP_VELOCITY_THRESHOLD:
                    # Fire TAP immediately on first downward frame.
                    # onset_ms = 0 because there is no press-to-fire delay.
                    self._pressed = True
                    self._press_time = now
                    self._slide_origin_x = smooth_x
                    gesture = GestureType.TAP
                    onset_ms = 0.0
            else:
                # --- DOWN state: watch for release or held-past-cooldown ---
                if vel_y < -self.RELEASE_VELOCITY_THRESHOLD and cooldown_elapsed:
                    # Finger lifted cleanly after the cooldown window → reset
                    self._pressed = False
                    self._slide_origin_x = None
                elif cooldown_elapsed and vel_y > self.TAP_VELOCITY_THRESHOLD:
                    # Rapid re-tap: cooldown already elapsed and another
                    # downstroke is detected; treat as a new tap immediately.
                    self._press_time = now
                    self._slide_origin_x = smooth_x
                    gesture = GestureType.TAP
                    onset_ms = 0.0

        # --- Slide detection: horizontal travel while in DOWN state ---
        # Checked independently so a slide can follow a TAP in the same stroke.
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
