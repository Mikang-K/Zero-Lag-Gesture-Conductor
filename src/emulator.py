"""
emulator.py — Input emulation module
Converts GestureEvent objects into OS-level keyboard events via pyautogui.

Default key mapping (configurable via config/keymap.json):
    Left  index→f, middle→d, ring→s, pinky→a
    Right index→j, middle→k, ring→l, pinky→;

Key lookup priority in emit():
    1. "{gesture}_{hand}_{finger}"  e.g. "TAP_Left_index"
    2. "{gesture}_{finger}"         e.g. "TAP_index"
    3. "{gesture}"                  e.g. "TAP"  (fallback for old keymaps)
"""

import json
import time
import pyautogui
from pathlib import Path

from recognizer import GestureEvent


# Disable pyautogui's built-in failsafe pause (we handle timing ourselves)
pyautogui.PAUSE = 0.0


_DEFAULT_KEYMAP: dict[str, str] = {
    # Left hand: index→f, middle→d, ring→s, pinky→a
    "TAP_Left_index":  "f",
    "TAP_Left_middle": "d",
    "TAP_Left_ring":   "s",
    "TAP_Left_pinky":  "a",
    # Right hand: index→j, middle→k, ring→l, pinky→;
    "TAP_Right_index":  "j",
    "TAP_Right_middle": "k",
    "TAP_Right_ring":   "l",
    "TAP_Right_pinky":  ";",
}


def _load_keymap(path: Path) -> dict[str, str]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return _DEFAULT_KEYMAP


class InputEmulator:
    def __init__(self, keymap_path: Path | None = None):
        default_path = Path(__file__).parent.parent / "config" / "keymap.json"
        self._keymap = _load_keymap(keymap_path or default_path)

    def emit(self, event: GestureEvent) -> float:
        """
        Send the OS-level key event for the given gesture.
        Returns the emulation latency in milliseconds.
        """
        g = event.gesture.name
        key = (
            self._keymap.get(f"{g}_{event.hand}_{event.finger}")
            or self._keymap.get(f"{g}_{event.finger}")
            or self._keymap.get(g)
        )
        if key is None:
            return 0.0

        t0 = time.perf_counter()
        pyautogui.press(key)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return latency_ms
