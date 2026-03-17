"""
recognizer.py — Gesture recognition engine
Detects Press/Release for the index finger using a 3-state machine.

Gesture mapping:
  Index finger fully extended, pressed DOWN  → PRESS  (space keyDown)
  Index finger lifted back UP                → RELEASE (space keyUp)

Extension gate: events only fire when the index TIP-to-MCP normalized
distance is above MIN_EXTENDED_DIST at the moment of arming (RESTING →
READY). This prevents false triggers when the finger is curled inward.

Assumes a top-down camera looking at the back of the hand, with the hand
resting flat on a surface.

3-state machine:
  RESTING → (lift: dist decreases fast AND finger was extended) → READY
  READY   → (lower: dist increases fast) → PRESSED  [fires PRESS]
  PRESSED → (lift: dist decreases fast)  → READY    [fires RELEASE]
  READY   → (timeout 0.8 s)             → RESTING   [accidental lift reset]
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

from tracker import HandLandmarks, LandmarkIndex


# Normalized TIP-to-MCP distance below which the finger is NOT considered extended
MIN_EXTENDED_DIST = 0.50  # relative to palm size

# Velocity thresholds (normalized by palm size per frame)
_LIFT_VEL  = 0.022   # lifting speed to enter READY    (negative direction)
_LOWER_VEL = 0.018   # lowering speed to fire PRESS    (positive direction)


def _palm_size(hand: HandLandmarks) -> float:
    """WRIST→MIDDLE_MCP distance used to normalize distances."""
    wx, wy, _ = hand.get(LandmarkIndex.WRIST)
    mx, my, _ = hand.get(LandmarkIndex.MIDDLE_MCP)
    return max(math.hypot(wx - mx, wy - my), 1e-6)


class GestureType(Enum):
    NONE    = auto()
    PRESS   = auto()   # finger lowered after being lifted → keyDown
    RELEASE = auto()   # finger lifted again after press   → keyUp


class _State(Enum):
    RESTING = auto()   # finger flat/extended on surface
    READY   = auto()   # finger lifted, armed for a press
    PRESSED = auto()   # finger pressed back down, key held


@dataclass
class GestureEvent:
    gesture: GestureType
    hand: str           # "Left" or "Right"
    finger: str         # always "index"
    confidence: float   # 0.0–1.0
    onset_ms: float = 0.0


class _MovingAverage:
    """Simple moving average over the last `window` values."""
    def __init__(self, window: int = 5):
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> float:
        self._buf.append(value)
        return sum(self._buf) / len(self._buf)


class _IndexFingerState:
    """3-state machine for the index finger using TIP→MCP curl distance.

    Top-down camera geometry:
      - Finger flat/extended on surface : TIP far from MCP  → large distance
      - Finger lifted up                : TIP closer to MCP → distance decreases
      - Finger pressed back down        : TIP far from MCP  → distance increases
    """

    READY_TIMEOUT_S = 0.8

    def __init__(self):
        self._avg_dist = _MovingAverage(window=5)
        self._smooth_dist: float = 0.0
        self._initialized: bool = False
        self._state: _State = _State.RESTING
        self._state_entered: float = 0.0
        self._last_onset_ms: float = 0.0

    def update(self, hand: HandLandmarks) -> GestureType:
        tx, ty, _ = hand.get(LandmarkIndex.INDEX_TIP)
        mx, my, _ = hand.get(LandmarkIndex.INDEX_MCP)
        raw_dist = math.hypot(tx - mx, ty - my) / _palm_size(hand)
        new_dist = self._avg_dist.update(raw_dist)

        if not self._initialized:
            self._smooth_dist = new_dist
            self._initialized = True
            return GestureType.NONE

        vel = new_dist - self._smooth_dist   # negative = lifting, positive = lowering
        self._smooth_dist = new_dist
        now = time.perf_counter()

        if self._state == _State.RESTING:
            # Only arm if the finger was clearly extended before lifting
            if vel < -_LIFT_VEL and new_dist >= MIN_EXTENDED_DIST:
                self._state = _State.READY
                self._state_entered = now

        elif self._state == _State.READY:
            if (now - self._state_entered) > self.READY_TIMEOUT_S:
                self._state = _State.RESTING
            elif vel > _LOWER_VEL:
                self._last_onset_ms = (now - self._state_entered) * 1000.0
                self._state = _State.PRESSED
                self._state_entered = now
                return GestureType.PRESS

        elif self._state == _State.PRESSED:
            if vel < -_LIFT_VEL:
                self._last_onset_ms = 0.0
                self._state = _State.READY
                self._state_entered = now
                return GestureType.RELEASE

        return GestureType.NONE


class GestureRecognizer:
    def __init__(self):
        # One state machine per detected hand label ("Left" / "Right")
        self._states: dict[str, _IndexFingerState] = {}

    def update(self, hands: list[HandLandmarks]) -> list[GestureEvent]:
        events: list[GestureEvent] = []

        for hand in hands:
            label = hand.handedness
            if label not in self._states:
                self._states[label] = _IndexFingerState()

            gesture = self._states[label].update(hand)
            if gesture != GestureType.NONE:
                events.append(GestureEvent(
                    gesture=gesture,
                    hand=label,
                    finger="index",
                    confidence=1.0,
                    onset_ms=self._states[label]._last_onset_ms,
                ))

        return events
