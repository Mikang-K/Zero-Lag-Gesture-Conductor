"""
emulator.py — Input emulation module
Converts GestureEvent objects into OS-level keyboard events via pyautogui.

Default key mapping (configurable via config/keymap.json):
    TAP          -> Space
    SLIDE_LEFT   -> Left arrow
    SLIDE_RIGHT  -> Right arrow
"""

import json
import time
import pyautogui
from pathlib import Path

from recognizer import GestureEvent, GestureType


# Disable pyautogui's built-in failsafe pause (we handle timing ourselves)
pyautogui.PAUSE = 0.0


_DEFAULT_KEYMAP: dict[str, str] = {
    GestureType.TAP.name:         "space",
    GestureType.SLIDE_LEFT.name:  "left",
    GestureType.SLIDE_RIGHT.name: "right",
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
        key = self._keymap.get(event.gesture.name)
        if key is None:
            return 0.0

        t0 = time.perf_counter()
        pyautogui.press(key)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return latency_ms
