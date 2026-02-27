"""
recognizer.py — Gesture recognition engine
Detects Press/Release gestures from hand landmark sequences.
Uses a moving-average filter on finger tip positions to reduce noise.
Tracks 4 fingers per hand (index, middle, ring, pinky) independently.

State machine per finger:
  UP   → downstroke (vel_y > threshold) → fire PRESS,  enter DOWN state
  DOWN → upstroke   (vel_y < -threshold) → fire RELEASE, enter UP state
"""

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

from tracker import HandLandmarks, LandmarkIndex


# Fingers to track and their corresponding landmark tip indices
TRACKED_FINGERS: dict[str, int] = {
    "index":  LandmarkIndex.INDEX_TIP,
    "middle": LandmarkIndex.MIDDLE_TIP,
    "ring":   LandmarkIndex.RING_TIP,
    "pinky":  LandmarkIndex.PINKY_TIP,
}


class GestureType(Enum):
    NONE = auto()
    PRESS = auto()    # finger moves down  → keyDown
    RELEASE = auto()  # finger moves back up → keyUp


@dataclass
class GestureEvent:
    gesture: GestureType
    hand: str           # "Left" or "Right"
    finger: str         # "index", "middle", "ring", "pinky"
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
    """Per-finger state machine tracking a single finger tip.

    PRESS fires immediately on downstroke so the key reaches the game at the
    earliest possible moment. RELEASE fires when the finger lifts back up,
    enabling natural long-note (hold) behaviour.
    """

    # --- Tunable thresholds ---
    # Downward velocity (normalized units/frame) to register a press.
    PRESS_VELOCITY_THRESHOLD = 0.025

    # Upward velocity to register a release. Gentler than press so a soft
    # lift still resets the state.
    RELEASE_VELOCITY_THRESHOLD = 0.010

    # Minimum time after pressing before a release is accepted.
    # Prevents spurious immediate releases from tracking noise.
    PRESS_HOLD_MIN_S = 0.05

    def __init__(self, tip_index: int):
        self._tip = tip_index
        self._avg_y = _MovingAverage(window=5)
        self._prev_y: float | None = None

        # True while the finger is considered "held down"
        self._pressed: bool = False
        self._press_time: float = 0.0

    def update(self, hand: HandLandmarks) -> tuple[GestureType, float]:
        raw_x, raw_y, _ = hand.get(self._tip)
        smooth_y = self._avg_y.update(raw_y)

        gesture = GestureType.NONE
        onset_ms = 0.0

        if self._prev_y is not None:
            vel_y = smooth_y - self._prev_y   # positive = moving down (screen coords)
            now = time.perf_counter()

            if not self._pressed:
                # --- UP state: wait for downstroke ---
                if vel_y > self.PRESS_VELOCITY_THRESHOLD:
                    self._pressed = True
                    self._press_time = now
                    gesture = GestureType.PRESS
            else:
                # --- DOWN state: wait for genuine upstroke ---
                hold_elapsed = (now - self._press_time) >= self.PRESS_HOLD_MIN_S
                if hold_elapsed and vel_y < -self.RELEASE_VELOCITY_THRESHOLD:
                    self._pressed = False
                    gesture = GestureType.RELEASE

        self._prev_y = smooth_y
        return gesture, onset_ms


class GestureRecognizer:
    def __init__(self):
        # Separate state per (hand_label, finger_name)
        self._states: dict[tuple[str, str], _FingerState] = {}

    def update(self, hands: list[HandLandmarks]) -> list[GestureEvent]:
        events: list[GestureEvent] = []

        for hand in hands:
            label = hand.handedness
            for finger_name, tip_idx in TRACKED_FINGERS.items():
                key = (label, finger_name)
                if key not in self._states:
                    self._states[key] = _FingerState(tip_index=tip_idx)

                gesture, onset_ms = self._states[key].update(hand)
                if gesture != GestureType.NONE:
                    events.append(GestureEvent(
                        gesture=gesture,
                        hand=label,
                        finger=finger_name,
                        confidence=1.0,   # placeholder; refine in Phase 2
                        onset_ms=onset_ms,
                    ))

        return events
