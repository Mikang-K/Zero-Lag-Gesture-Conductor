"""
compare.py — Phase 1 vs Phase 2 headless benchmark comparison

Runs N frames through each backend and prints a comparison table.
Usage:
    python benchmarks/compare.py [--frames 120]
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


def _bench(name: str, tracker, cam, n_frames: int) -> dict:
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

    def _stats(lst):
        return (
            sum(lst) / len(lst),
            min(lst),
            max(lst),
            sorted(lst)[int(len(lst) * 0.95)],   # p95
        )

    return {
        "name":     name,
        "frames":   len(cap_list),
        "capture":  _stats(cap_list),
        "inference":_stats(inf_list),
        "e2e":      _stats(e2e_list),
        "fps":      1000.0 / (sum(e2e_list) / len(e2e_list)) if e2e_list else 0,
    }


def _print_table(results: list[dict]):
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

    col_w = [max(len(COLS[i]), max(len(row[i]) for row in rows)) for i in range(len(COLS))]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    fmt = "|" + "|".join(f" {{:<{w}}} " for w in col_w) + "|"

    print(sep)
    print(fmt.format(*COLS))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)

    # Speedup vs Phase 1 baseline
    baseline = results[0]["inference"][0]
    print("\nInference speedup vs Phase 1 (MediaPipe CPU):")
    for r in results:
        sp = baseline / r["inference"][0] if r["inference"][0] > 0 else 0
        print(f"  {r['name']:<30} {sp:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    from capture import CameraCapture
    cam = CameraCapture(camera_index=args.camera, target_fps=60)
    if not cam.open():
        print("ERROR: Cannot open camera")
        sys.exit(1)

    results = []

    # ── Phase 1 baseline: MediaPipe CPU ──
    print("Benchmarking MediaPipe CPU (Phase 1 baseline)...")
    from tracker import HandTracker
    t = HandTracker(max_num_hands=2, use_gpu=False)
    results.append(_bench("MediaPipe CPU", t, cam, args.frames))
    t.close()

    # ── MediaPipe GPU delegate ──
    print("Benchmarking MediaPipe GPU delegate...")
    try:
        t = HandTracker(max_num_hands=2, use_gpu=True)
        results.append(_bench("MediaPipe GPU", t, cam, args.frames))
        t.close()
    except Exception as e:
        print(f"  Skipped (GPU delegate unavailable): {e}")

    # ── ONNX Runtime CUDA ──
    print("Benchmarking ONNX Runtime CUDA...")
    try:
        from tracker_onnx import OnnxHandTracker
        t = OnnxHandTracker(max_num_hands=2, use_tensorrt=False)
        results.append(_bench("ONNX RT CUDA", t, cam, args.frames))
        t.close()
    except Exception as e:
        print(f"  Skipped: {e}")

    # ── ONNX Runtime TensorRT FP16 ──
    print("Benchmarking ONNX Runtime TensorRT FP16...")
    try:
        from tracker_onnx import OnnxHandTracker
        t = OnnxHandTracker(max_num_hands=2, use_tensorrt=True)
        results.append(_bench("ORT TensorRT FP16", t, cam, args.frames))
        t.close()
    except Exception as e:
        print(f"  Skipped: {e}")

    cam.release()

    print("\n" + "=" * 70)
    print("PHASE 2 BENCHMARK RESULTS")
    print("=" * 70)
    _print_table(results)


if __name__ == "__main__":
    main()
