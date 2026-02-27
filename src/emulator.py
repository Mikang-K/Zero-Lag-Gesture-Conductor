"""
emulator.py — Input emulation module
Converts GestureEvent objects into OS-level keyboard events via pydirectinput.

Default key mapping (configurable via config/keymap.json):
    Left  index→f, middle→d, ring→s, pinky→a
    Right index→j, middle→k, ring→l, pinky→;

Key lookup uses "{hand}_{finger}" (e.g. "Left_index").
PRESS  → keyDown  (finger goes down, key held)
RELEASE → keyUp   (finger lifts,    key released)
"""

import json
import time
import pyautogui
import pydirectinput
from pathlib import Path

from recognizer import GestureEvent, GestureType


# Disable built-in failsafe pauses (we handle timing ourselves)
pyautogui.PAUSE = 0.0
pydirectinput.PAUSE = 0.0


_DEFAULT_KEYMAP: dict[str, str] = {
    # Left hand: index→f, middle→d, ring→s, pinky→a
    "Left_index":  "f",
    "Left_middle": "d",
    "Left_ring":   "s",
    "Left_pinky":  "a",
    # Right hand: index→j, middle→k, ring→l, pinky→;
    "Right_index":  "j",
    "Right_middle": "k",
    "Right_ring":   "l",
    "Right_pinky":  ";",
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
        self._held: set[str] = set()   # keys currently held down

    def emit(self, event: GestureEvent) -> float:
        """
        Send keyDown on PRESS, keyUp on RELEASE.
        Returns the emulation latency in milliseconds.
        """
        key = (
            self._keymap.get(f"{event.hand}_{event.finger}")
            or self._keymap.get(event.finger)
        )
        if key is None:
            return 0.0

        t0 = time.perf_counter()
        if event.gesture == GestureType.PRESS and key not in self._held:
            pydirectinput.keyDown(key)
            self._held.add(key)
        elif event.gesture == GestureType.RELEASE and key in self._held:
            pydirectinput.keyUp(key)
            self._held.discard(key)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return latency_ms

    def release_all(self) -> None:
        """Release every key currently held. Call on shutdown or emulation toggle-off."""
        for key in list(self._held):
            pydirectinput.keyUp(key)
        self._held.clear()
