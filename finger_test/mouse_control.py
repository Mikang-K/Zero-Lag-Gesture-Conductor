"""
finger_test/mouse_control.py — Mouse control test using hand landmarks

Purpose:
  Isolated testbed for tuning finger detection thresholds before integrating
  back into the main recognizer pipeline.

Behaviour:
  - INDEX_TIP position  → mouse movement (EMA smoothed, X mirrored)
  - Index finger curl   → left mouse button click
    (curl ratio = INDEX_TIP→INDEX_MCP distance / palm size)
    Small curl ratio  = finger curled   → click
    Large curl ratio  = finger extended → release

Keyboard controls (while OpenCV window is focused):
  +  / =   raise click threshold (less sensitive)
  -        lower click threshold (more sensitive)
  Q / ESC  quit
"""

import math
import os
import sys
import time

import cv2
import numpy as np
import pyautogui

# Allow importing from src/ regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from capture import CameraCapture
from tracker import HandTracker, LandmarkIndex


# ---------------------------------------------------------------------------
# Tunable config — adjust while running with +/- keys
# ---------------------------------------------------------------------------

# EMA alpha: lower = smoother pointer but more lag (0.1–0.5 range)
SMOOTH_ALPHA: float = 0.25

# Curl ratio below this → click down.
# Typical range: 0.5 (relaxed) to 0.9 (straight).
# Start at 0.65 and tune with +/-
CLICK_THRESHOLD: float = 0.65

# Hysteresis: curl must rise above CLICK_THRESHOLD * HYSTERESIS_FACTOR to release.
# Prevents rapid toggling at the boundary.
HYSTERESIS_FACTOR: float = 1.15

# ---------------------------------------------------------------------------
# Screen dimensions
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = pyautogui.size()

# Disable pyautogui fail-safe (moving to corner would raise exception otherwise)
pyautogui.FAILSAFE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average for smoothing 1-D signals."""

    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self._val: float | None = None

    def update(self, x: float) -> float:
        if self._val is None:
            self._val = x
        else:
            self._val = self.alpha * x + (1.0 - self.alpha) * self._val
        return self._val

    def reset(self) -> None:
        self._val = None


def _palm_size(hand) -> float:
    """WRIST→MIDDLE_MCP distance (normalisation reference)."""
    wx, wy, _ = hand.get(LandmarkIndex.WRIST)
    mx, my, _ = hand.get(LandmarkIndex.MIDDLE_MCP)
    return max(math.hypot(wx - mx, wy - my), 1e-6)


def index_curl_ratio(hand) -> float:
    """Normalised INDEX_TIP→INDEX_MCP distance / palm size.

    ~0.4–0.6 when finger is fully curled,
    ~0.8–1.2 when finger is fully extended.
    """
    tx, ty, _ = hand.get(LandmarkIndex.INDEX_TIP)
    mx, my, _ = hand.get(LandmarkIndex.INDEX_MCP)
    return math.hypot(tx - mx, ty - my) / _palm_size(hand)


def map_to_screen(nx: float, ny: float) -> tuple[int, int]:
    """Normalised [0,1] hand coords → screen pixel coords (X mirrored)."""
    sx = int((1.0 - nx) * SCREEN_W)
    sy = int(ny * SCREEN_H)
    sx = max(0, min(SCREEN_W - 1, sx))
    sy = max(0, min(SCREEN_H - 1, sy))
    return sx, sy


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    global CLICK_THRESHOLD

    cap = CameraCapture(threaded=True)
    if not cap.open():
        print("ERROR: Failed to open camera.")
        return

    tracker = HandTracker(max_num_hands=1)

    smooth_x = EMA(SMOOTH_ALPHA)
    smooth_y = EMA(SMOOTH_ALPHA)

    clicking = False          # current mouse-button state
    frame_times: list[float] = []

    print("=== Mouse Control Test ===")
    print("  Move hand  → mouse pointer follows INDEX_TIP")
    print("  Curl index → left click")
    print("  +/-        → adjust sensitivity")
    print("  Q / ESC    → quit")
    print()

    try:
        while True:
            ok, frame, _ = cap.read()
            if not ok or frame is None:
                continue

            result = tracker.process(frame)

            # ----------------------------------------------------------------
            # Hand → mouse mapping
            # ----------------------------------------------------------------
            curl: float | None = None

            if result.hands:
                hand = result.hands[0]

                # Mouse movement
                ix, iy, _ = hand.get(LandmarkIndex.INDEX_TIP)
                sx = smooth_x.update(ix)
                sy = smooth_y.update(iy)
                screen_x, screen_y = map_to_screen(sx, sy)
                pyautogui.moveTo(screen_x, screen_y, duration=0)

                # Click detection via index curl ratio
                curl = index_curl_ratio(hand)
                release_threshold = CLICK_THRESHOLD * HYSTERESIS_FACTOR

                if not clicking and curl < CLICK_THRESHOLD:
                    clicking = True
                    pyautogui.mouseDown()
                elif clicking and curl > release_threshold:
                    clicking = False
                    pyautogui.mouseUp()

            else:
                # No hand: reset smoothers so pointer doesn't drift on re-entry
                smooth_x.reset()
                smooth_y.reset()
                if clicking:
                    clicking = False
                    pyautogui.mouseUp()

            # ----------------------------------------------------------------
            # Visualisation / HUD
            # ----------------------------------------------------------------
            frame = tracker.draw(frame, result)

            # FPS (rolling 1-second window)
            now = time.perf_counter()
            frame_times.append(now)
            frame_times = [t for t in frame_times if t > now - 1.0]
            fps = len(frame_times)

            # Overlay text
            state_label = "CLICK" if clicking else "idle"
            state_color = (0, 0, 255) if clicking else (0, 255, 0)
            curl_label  = f"{curl:.3f}" if curl is not None else "---"

            cv2.putText(frame, f"FPS: {fps}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            cv2.putText(frame, f"Curl: {curl_label}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, f"State: {state_label}",
                        (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2)
            cv2.putText(frame, f"Threshold: {CLICK_THRESHOLD:.2f}  (±: tune)",
                        (10, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            # Draw a circle on INDEX_TIP for visual feedback
            if result.hands:
                h, w = frame.shape[:2]
                ix, iy, _ = result.hands[0].get(LandmarkIndex.INDEX_TIP)
                tip_px = (int(ix * w), int(iy * h))
                dot_color = (0, 0, 255) if clicking else (0, 255, 255)
                cv2.circle(frame, tip_px, 10, dot_color, -1)

            cv2.imshow("Mouse Control Test [finger_test]", frame)

            # ----------------------------------------------------------------
            # Keyboard input
            # ----------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):          # Q or ESC → quit
                break
            elif key in (ord("+"), ord("=")):  # raise threshold (less sensitive)
                CLICK_THRESHOLD = min(CLICK_THRESHOLD + 0.02, 1.5)
                print(f"Threshold → {CLICK_THRESHOLD:.2f}")
            elif key == ord("-"):              # lower threshold (more sensitive)
                CLICK_THRESHOLD = max(CLICK_THRESHOLD - 0.02, 0.1)
                print(f"Threshold → {CLICK_THRESHOLD:.2f}")

    finally:
        if clicking:
            pyautogui.mouseUp()
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
