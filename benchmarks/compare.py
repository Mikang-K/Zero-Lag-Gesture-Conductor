"""
compare.py — Phase 1 → Phase 2 → Threaded-capture headless benchmark

Produces a three-section report that tells the full optimisation story:

  Section 1  Tracker benchmark (blocking read, all backends)
             Reproduces the Phase 1 vs Phase 2 inference comparison.

  Section 2  Capture-mode benchmark (best tracker backend)
             blocking read()  vs  threaded grab
             The rhythm-game headline: how much wall-clock lag the player
             actually experiences from motion to emulated keypress.

  Section 3  Rhythm-game readiness summary
             Pass/Fail per target against < 16 ms E2E (one 60-fps frame).

Usage:
    python benchmarks/compare.py [--frames 120] [--camera 0]
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Rhythm-game target: all frames must be processed within one 60-fps frame.
RHYTHM_GAME_TARGET_MS = 16.7   # ms — 1 / 60 fps


# ──────────────────────────────────────────────────────────────────────
# Core benchmark runner
# ──────────────────────────────────────────────────────────────────────

def _bench(name: str, tracker, cam, n_frames: int) -> dict:
    """Run n_frames through the given tracker+camera pair and return stats."""
    cap_list, inf_list, e2e_list = [], [], []

    for _ in range(n_frames):
        ok, frame, cap_ms = cam.read()
        if not ok:
            continue

        result = tracker.process(frame)
        inf_ms = result.inference_ms
        e2e_ms = cap_ms + inf_ms

        cap_list.append(cap_ms)
        inf_list.append(inf_ms)
        e2e_list.append(e2e_ms)

    def _stats(lst: list[float]):
        if not lst:
            return 0.0, 0.0, 0.0, 0.0
        s = sorted(lst)
        return (
            sum(s) / len(s),          # avg
            s[0],                     # min
            s[-1],                    # max
            s[int(len(s) * 0.95)],   # p95
        )

    return {
        "name":      name,
        "frames":    len(cap_list),
        "capture":   _stats(cap_list),
        "inference": _stats(inf_list),
        "e2e":       _stats(e2e_list),
        "fps":       1000.0 / (sum(e2e_list) / len(e2e_list)) if e2e_list else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────
# Table rendering
# ──────────────────────────────────────────────────────────────────────

def _print_tracker_table(results: list[dict]):
    """Section 1: tracker backend comparison (capture mode fixed to blocking)."""
    COLS = ["Backend", "Frames", "Cap avg", "Inf avg", "Inf p95", "E2E avg", "FPS"]
    rows = []
    for r in results:
        rows.append([
            r["name"],
            str(r["frames"]),
            f"{r['capture'][0]:.1f} ms",
            f"{r['inference'][0]:.1f} ms",
            f"{r['inference'][3]:.1f} ms",
            f"{r['e2e'][0]:.1f} ms",
            f"{r['fps']:.1f}",
        ])
    _render_table(COLS, rows)

    # Speedup column vs Phase 1 baseline
    baseline_inf = results[0]["inference"][0]
    print("\nInference speedup vs Phase 1 (MediaPipe CPU + blocking read):")
    for r in results:
        sp = baseline_inf / r["inference"][0] if r["inference"][0] > 0 else 0
        print(f"  {r['name']:<36} {sp:.2f}x")


def _print_capture_table(results: list[dict], baseline_e2e: float):
    """Section 2: capture-mode comparison.

    `baseline_e2e` is the Phase 1 E2E average for improvement % calculation.
    """
    COLS = [
        "Config", "Frames",
        "Cap avg", "Cap p95",
        "Inf avg",
        "E2E avg", "E2E p95",
        "FPS", "vs Phase1",
    ]
    rows = []
    for r in results:
        cap_avg  = r["capture"][0]
        cap_p95  = r["capture"][3]
        inf_avg  = r["inference"][0]
        e2e_avg  = r["e2e"][0]
        e2e_p95  = r["e2e"][3]
        fps      = r["fps"]
        improve  = (baseline_e2e - e2e_avg) / baseline_e2e * 100.0 if baseline_e2e > 0 else 0.0
        rows.append([
            r["name"],
            str(r["frames"]),
            f"{cap_avg:.1f} ms",
            f"{cap_p95:.1f} ms",
            f"{inf_avg:.1f} ms",
            f"{e2e_avg:.1f} ms",
            f"{e2e_p95:.1f} ms",
            f"{fps:.1f}",
            f"{improve:+.0f}%",
        ])
    _render_table(COLS, rows)


def _print_rhythm_game_summary(results: list[dict]):
    """Section 3: rhythm-game readiness — pass / fail vs 16.7 ms target."""
    print(f"\nRhythm-game target: E2E avg < {RHYTHM_GAME_TARGET_MS:.1f} ms  "
          f"(1 frame @ 60 fps)")
    for r in results:
        e2e = r["e2e"][0]
        e2e_p95 = r["e2e"][3]
        avg_ok  = e2e     < RHYTHM_GAME_TARGET_MS
        p95_ok  = e2e_p95 < RHYTHM_GAME_TARGET_MS
        avg_tag = "PASS" if avg_ok  else "FAIL"
        p95_tag = "PASS" if p95_ok  else "FAIL"
        print(
            f"  {r['name']:<36}"
            f"  E2E avg {e2e:5.1f} ms [{avg_tag}]"
            f"  p95 {e2e_p95:5.1f} ms [{p95_tag}]"
        )


def _render_table(cols: list[str], rows: list[list[str]]):
    col_w = [
        max(len(cols[i]), max(len(row[i]) for row in rows))
        for i in range(len(cols))
    ]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    fmt = "|" + "|".join(f" {{:<{w}}} " for w in col_w) + "|"
    print(sep)
    print(fmt.format(*cols))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Zero-Lag Gesture Conductor — benchmark suite"
    )
    parser.add_argument("--frames", type=int, default=120,
                        help="Number of frames to measure per configuration")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    from capture import CameraCapture

    # ── Section 1: Tracker backend comparison (blocking capture) ──────
    print("\n" + "=" * 70)
    print("SECTION 1 — Tracker backend comparison  (blocking cap.read())")
    print("=" * 70)

    cam_blocking = CameraCapture(camera_index=args.camera, target_fps=60, threaded=False)
    if not cam_blocking.open():
        print("ERROR: Cannot open camera")
        sys.exit(1)

    tracker_results: list[dict] = []

    # Phase 1 baseline
    print("Benchmarking MediaPipe CPU (Phase 1 baseline)...")
    from tracker import HandTracker
    t = HandTracker(max_num_hands=2, use_gpu=False)
    tracker_results.append(_bench("Phase 1 — MediaPipe CPU", t, cam_blocking, args.frames))
    t.close()
    phase1_e2e = tracker_results[0]["e2e"][0]   # reference for improvement %

    # MediaPipe GPU delegate
    print("Benchmarking MediaPipe GPU delegate...")
    try:
        t = HandTracker(max_num_hands=2, use_gpu=True)
        tracker_results.append(_bench("Phase 2 — MediaPipe GPU", t, cam_blocking, args.frames))
        t.close()
    except Exception as e:
        print(f"  Skipped (GPU delegate unavailable): {e}")

    # ONNX Runtime CUDA
    print("Benchmarking ONNX Runtime CUDA...")
    try:
        from tracker_onnx import OnnxHandTracker
        t = OnnxHandTracker(max_num_hands=2, use_tensorrt=False)
        tracker_results.append(_bench("Phase 2 — ONNX RT CUDA", t, cam_blocking, args.frames))
        t.close()
    except Exception as e:
        print(f"  Skipped: {e}")

    # ONNX Runtime TensorRT FP16
    print("Benchmarking ONNX Runtime TensorRT FP16...")
    try:
        from tracker_onnx import OnnxHandTracker
        t = OnnxHandTracker(max_num_hands=2, use_tensorrt=True)
        tracker_results.append(_bench("Phase 2 — ORT TensorRT FP16", t, cam_blocking, args.frames))
        t.close()
    except Exception as e:
        print(f"  Skipped: {e}")

    cam_blocking.release()
    _print_tracker_table(tracker_results)

    # ── Section 2: Capture-mode comparison (best tracker backend) ─────
    print("\n" + "=" * 70)
    print("SECTION 2 — Capture mode comparison  (ONNX RT CUDA tracker)")
    print("  Metric: frame staleness (threaded) vs grab+decode time (blocking)")
    print("=" * 70)

    capture_results: list[dict] = []

    try:
        from tracker_onnx import OnnxHandTracker

        # 2a. Blocking read (Phase 2 status-quo)
        print("Benchmarking ONNX CUDA + blocking read()...")
        cam_block = CameraCapture(camera_index=args.camera, target_fps=60, threaded=False)
        cam_block.open()
        t = OnnxHandTracker(max_num_hands=2, use_tensorrt=False)
        capture_results.append(
            _bench("ONNX CUDA + blocking read  (Phase 2)", t, cam_block, args.frames)
        )
        t.close()
        cam_block.release()

        # 2b. Threaded grab (new)
        print("Benchmarking ONNX CUDA + threaded grab (new)...")
        cam_thread = CameraCapture(camera_index=args.camera, target_fps=60, threaded=True)
        cam_thread.open()
        t = OnnxHandTracker(max_num_hands=2, use_tensorrt=False)
        capture_results.append(
            _bench("ONNX CUDA + threaded grab  (new)", t, cam_thread, args.frames)
        )
        t.close()
        cam_thread.release()

    except Exception as e:
        print(f"  Section 2 skipped (ONNX RT CUDA unavailable): {e}")

    if capture_results:
        _print_capture_table(capture_results, baseline_e2e=phase1_e2e)

    # ── Section 3: Rhythm-game readiness summary ──────────────────────
    print("\n" + "=" * 70)
    print("SECTION 3 — Rhythm-game readiness")
    print("=" * 70)
    _print_rhythm_game_summary(tracker_results + capture_results)


if __name__ == "__main__":
    main()
