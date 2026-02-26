"""
main_phase2.py — Zero-Lag Gesture Conductor Phase 2 entry point

Optimizations over Phase 1:
  - MediaPipe GPU delegate (--backend mediapipe-gpu)
  - ONNX Runtime CUDA (--backend onnx-cuda)         [default]
  - ONNX Runtime TensorRT FP16 (--backend tensorrt)
  - Multiprocessing pipeline (--pipeline mp)         [default]

Usage:
    python main_phase2.py
    python main_phase2.py --backend onnx-cuda --pipeline mp
    python main_phase2.py --backend tensorrt
    python main_phase2.py --backend mediapipe-gpu --pipeline single

Controls:
    E   — toggle input emulation
    S   — save screenshot
    Q / ESC — quit
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, str(Path(__file__).parent / "src"))

from recognizer import GestureRecognizer
from emulator import InputEmulator
from monitor import PerformanceMonitor, FrameMetrics


# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 — GPU-accelerated gesture conductor")
    p.add_argument(
        "--backend",
        choices=["mediapipe-cpu", "mediapipe-gpu", "onnx-cuda", "tensorrt"],
        default="onnx-cuda",
        help="Inference backend",
    )
    p.add_argument(
        "--pipeline",
        choices=["single", "mp"],
        default="mp",
        help="single=sequential loop, mp=multiprocessing pipeline",
    )
    p.add_argument("--camera", type=int, default=0)
    return p.parse_args()


# ──────────────────────────────────────────────
# Single-process loop (for debugging / comparison)
# ──────────────────────────────────────────────

def _run_single(args: argparse.Namespace):
    from capture import CameraCapture

    if args.backend == "mediapipe-cpu":
        from tracker import HandTracker
        tracker = HandTracker(max_num_hands=2, use_gpu=False)
    elif args.backend == "mediapipe-gpu":
        from tracker import HandTracker
        tracker = HandTracker(max_num_hands=2, use_gpu=True)
    elif args.backend == "onnx-cuda":
        from tracker_onnx import OnnxHandTracker
        tracker = OnnxHandTracker(max_num_hands=2, use_tensorrt=False)
    else:   # tensorrt
        from tracker_onnx import OnnxHandTracker
        tracker = OnnxHandTracker(max_num_hands=2, use_tensorrt=True)

    cam = CameraCapture(camera_index=args.camera, target_fps=60)
    if not cam.open():
        print("[ERROR] Cannot open camera")
        sys.exit(1)

    recognizer = GestureRecognizer()
    emulator = InputEmulator()
    monitor = PerformanceMonitor(
        csv_path=Path("benchmarks") / f"phase2_{args.backend}_single.csv"
    )

    emulation_enabled = False
    print(f"[Phase 2 / {args.backend} / single] E=emulate, S=screenshot, Q=quit")

    try:
        while True:
            monitor.begin_frame()

            ok, frame, cap_ms = cam.read()
            if not ok:
                continue

            track = tracker.process(frame)

            t0 = time.perf_counter()
            events = recognizer.update(track.hands)
            rec_ms = (time.perf_counter() - t0) * 1000.0

            emu_ms = 0.0
            if emulation_enabled:
                for ev in events:
                    emu_ms += emulator.emit(ev)

            onset_ms = max((ev.onset_ms for ev in events), default=0.0)

            monitor.record(
                capture_ms=cap_ms,
                inference_ms=track.inference_ms,
                recognize_ms=rec_ms,
                emulate_ms=emu_ms,
                onset_ms=onset_ms,
            )
            monitor.commit()

            for ev in events:
                print(f"  [{ev.hand}] {ev.gesture.name}")

            tracker.draw(frame, track)
            monitor.draw_hud(frame)

            badge = f"[{args.backend}] EMU:{'ON' if emulation_enabled else 'OFF'}"
            cv2.putText(frame, badge, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1, cv2.LINE_AA)

            cv2.imshow("Phase 2 — Gesture Conductor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("e"):
                emulation_enabled = not emulation_enabled
            elif key == ord("s"):
                p = Path("benchmarks") / f"frame_{int(time.time())}.png"
                cv2.imwrite(str(p), frame)
                print(f"[INFO] Screenshot: {p}")
    finally:
        cam.release()
        tracker.close()
        monitor.close()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown.")


# ──────────────────────────────────────────────
# Multiprocessing pipeline loop
# ──────────────────────────────────────────────

def _run_multiprocess(args: argparse.Namespace):
    """
    Capture and inference run in separate processes.
    Main process only handles: gesture recognition, emulation, display.
    """
    from pipeline_mp import MultiprocessPipeline
    from tracker import HandLandmarks

    use_gpu = args.backend == "mediapipe-gpu"
    use_onnx = args.backend in ("onnx-cuda", "tensorrt")
    use_tensorrt = args.backend == "tensorrt"

    pipeline = MultiprocessPipeline(
        camera_index=args.camera,
        use_gpu=use_gpu,
        use_onnx=use_onnx,
        use_tensorrt=use_tensorrt,
    )
    pipeline.start()
    print(f"[Phase 2 / {args.backend} / multiprocess] Pipeline started. E=emulate, Q=quit")

    recognizer = GestureRecognizer()
    emulator = InputEmulator()
    monitor = PerformanceMonitor(
        csv_path=Path("benchmarks") / f"phase2_{args.backend}_mp.csv"
    )

    emulation_enabled = False
    frame_w, frame_h = 640, 480
    display_frame = None

    try:
        while True:
            monitor.begin_frame()

            result = pipeline.get_result(timeout=0.05)
            if result is None:
                if display_frame is not None:
                    cv2.imshow("Phase 2 — Gesture Conductor (MP)", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            # Reconstruct HandLandmarks from serialised data
            hands = [
                HandLandmarks(points=pts, handedness=label)
                for pts, label in result.hands
            ]

            t0 = time.perf_counter()
            events = recognizer.update(hands)
            rec_ms = (time.perf_counter() - t0) * 1000.0

            emu_ms = 0.0
            if emulation_enabled:
                for ev in events:
                    emu_ms += emulator.emit(ev)

            onset_ms = max((ev.onset_ms for ev in events), default=0.0)

            monitor.record(
                capture_ms=result.capture_ms,
                inference_ms=result.inference_ms,
                recognize_ms=rec_ms,
                emulate_ms=emu_ms,
                onset_ms=onset_ms,
            )
            monitor.commit()

            for ev in events:
                print(f"  [{ev.hand}] {ev.gesture.name}")

            # Decode camera frame for display; fall back to black canvas if unavailable
            if result.frame_jpeg:
                import numpy as np
                jpg_arr = np.frombuffer(result.frame_jpeg, dtype=np.uint8)
                base_frame = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)
                if base_frame is None:
                    base_frame = None  # will trigger fallback below
            else:
                base_frame = None

            display_frame = _build_display_frame(
                frame_w, frame_h, hands, monitor.metrics,
                emulation_enabled, args.backend, base_frame=base_frame
            )
            cv2.imshow("Phase 2 — Gesture Conductor (MP)", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("e"):
                emulation_enabled = not emulation_enabled
            elif key == ord("s"):
                p = Path("benchmarks") / f"frame_{int(time.time())}.png"
                if display_frame is not None:
                    cv2.imwrite(str(p), display_frame)
                    print(f"[INFO] Screenshot: {p}")
    finally:
        pipeline.stop()
        monitor.close()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown.")


def _build_display_frame(
    w: int, h: int,
    hands,
    metrics: FrameMetrics,
    emu_on: bool,
    backend: str,
    base_frame: "np.ndarray | None" = None,
) -> "np.ndarray":
    """Draw HUD on top of the camera frame (or black canvas if unavailable)."""
    import numpy as np
    if base_frame is not None:
        frame = cv2.resize(base_frame, (w, h))
    else:
        frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Landmark dots
    for hand in hands:
        for x, y, _ in hand.points:
            px, py = int(x * w), int(y * h)
            cv2.circle(frame, (px, py), 4, (0, 255, 100), -1)

    # HUD
    perceived = metrics.onset_ms + metrics.total_ms
    warn = metrics.total_ms > 33.0
    lines = [
        f"FPS:       {metrics.fps:5.1f}",
        f"Capture:   {metrics.capture_ms:5.1f} ms",
        f"Inference: {metrics.inference_ms:5.1f} ms",
        f"Pipeline:  {metrics.total_ms:5.1f} ms" + (" !" if warn else ""),
        f"Onset:     {metrics.onset_ms:5.1f} ms",
        f"Perceived: {perceived:5.1f} ms",
        f"Backend:   {backend}",
        f"EMU:       {'ON' if emu_on else 'OFF'}",
    ]
    color = (0, 165, 255) if warn else (0, 255, 0)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return frame


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Required for multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()

    args = _parse_args()
    if args.pipeline == "mp":
        _run_multiprocess(args)
    else:
        _run_single(args)
