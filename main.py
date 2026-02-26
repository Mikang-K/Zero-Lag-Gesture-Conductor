"""
main.py — Zero-Lag Gesture Conductor entry point (Phase 1 Prototype)

Controls:
    Q / ESC  : Quit
    E        : Toggle input emulation on/off (safe mode for testing)
    S        : Save current frame as screenshot to benchmarks/
"""

import sys
import time
from pathlib import Path

import cv2

# Add src/ to path so modules import cleanly without a package install
sys.path.insert(0, str(Path(__file__).parent / "src"))

from capture import CameraCapture
from tracker import HandTracker
from recognizer import GestureRecognizer
from emulator import InputEmulator
from monitor import PerformanceMonitor


def main():
    camera_index = 0
    target_fps = 60

    cam = CameraCapture(camera_index=camera_index, target_fps=target_fps)
    if not cam.open():
        print(f"[ERROR] Cannot open camera index {camera_index}")
        sys.exit(1)

    tracker = HandTracker(max_num_hands=2)
    recognizer = GestureRecognizer()
    emulator = InputEmulator()
    monitor = PerformanceMonitor()

    emulation_enabled = False
    print("[INFO] Starting. Press 'E' to toggle emulation, 'Q'/ESC to quit.")

    try:
        while True:
            monitor.begin_frame()

            # --- 1. Capture ---
            ok, frame, capture_ms = cam.read()
            if not ok:
                print("[WARN] Frame read failed, skipping.")
                continue

            # --- 2. Track (MediaPipe inference) ---
            track_result = tracker.process(frame)
            inference_ms = track_result.inference_ms

            # --- 3. Recognize gestures ---
            t0 = time.perf_counter()
            events = recognizer.update(track_result.hands)
            recognize_ms = (time.perf_counter() - t0) * 1000.0

            # --- 4. Emulate input ---
            emulate_ms = 0.0
            if emulation_enabled:
                for event in events:
                    emulate_ms += emulator.emit(event)

            # Onset: max across all gesture events this frame (0 if none)
            onset_ms = max((e.onset_ms for e in events), default=0.0)

            # Log & HUD
            monitor.record(
                capture_ms=capture_ms,
                inference_ms=inference_ms,
                recognize_ms=recognize_ms,
                emulate_ms=emulate_ms,
                onset_ms=onset_ms,
            )
            monitor.commit()

            # Print recognized gestures to console for debugging
            for event in events:
                print(f"  [{event.hand}] {event.gesture.name}")

            # Draw overlays
            tracker.draw(frame, track_result)
            monitor.draw_hud(frame)

            # Emulation status badge
            badge = "EMU: ON" if emulation_enabled else "EMU: OFF"
            badge_color = (0, 200, 0) if emulation_enabled else (0, 0, 200)
            cv2.putText(frame, badge, (frame.shape[1] - 120, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, badge_color, 2, cv2.LINE_AA)

            cv2.imshow("Zero-Lag Gesture Conductor — Phase 1", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # Q or ESC
                break
            elif key == ord("e"):
                emulation_enabled = not emulation_enabled
                print(f"[INFO] Emulation {'ENABLED' if emulation_enabled else 'DISABLED'}")
            elif key == ord("s"):
                shot_path = Path("benchmarks") / f"frame_{int(time.time())}.png"
                shot_path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(shot_path), frame)
                print(f"[INFO] Screenshot saved: {shot_path}")

    finally:
        cam.release()
        tracker.close()
        monitor.close()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    main()
