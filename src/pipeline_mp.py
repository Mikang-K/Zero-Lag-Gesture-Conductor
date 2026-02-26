"""
pipeline_mp.py — Multiprocessing pipeline for Phase 2

Architecture:
    [CaptureProcess]  →  SharedMemory ring buffer  →  [InferenceProcess]
                                                             │
                                                        result Queue
                                                             │
                                                       [Main Process]
                                                    (recognize + emulate + display)

Key benefit: capture latency (~25ms) and inference latency (~14ms) overlap.
Expected throughput improvement: ~40% over single-threaded baseline.
"""

import os
import sys
import time
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory, Queue, Event, Value
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np


# ──────────────────────────────────────────────
# Shared data structures
# ──────────────────────────────────────────────

FRAME_W = 640
FRAME_H = 480
FRAME_C = 3
FRAME_BYTES = FRAME_W * FRAME_H * FRAME_C   # 921600 bytes

# Ring buffer size: 2 slots (double-buffering)
RING_SIZE = 2


@dataclass
class InferenceResult:
    """Serialisable result passed from InferenceProcess to Main via Queue."""
    # List of (points, handedness) per detected hand
    hands: list[tuple[list[tuple[float, float, float]], str]] = field(
        default_factory=list
    )
    capture_ms: float = 0.0
    inference_ms: float = 0.0
    frame_id: int = 0
    # JPEG-encoded camera frame for display (quality 60); empty bytes = not available
    frame_jpeg: bytes = b""


# ──────────────────────────────────────────────
# Capture Process
# ──────────────────────────────────────────────

def _capture_worker(
    shm_name: str,
    write_slot: "mp.Value",          # current slot index to write (0 or 1)
    frame_ready: "mp.Event",
    stop_event: "mp.Event",
    camera_index: int,
    capture_ms_out: "mp.Value",
    frame_id_out: "mp.Value",
):
    """Continuously captures frames into shared memory ring buffer."""
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray(
        (RING_SIZE, FRAME_H, FRAME_W, FRAME_C),
        dtype=np.uint8,
        buffer=existing_shm.buf,
    )

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fid = 0
    slot = 0
    while not stop_event.is_set():
        t0 = time.perf_counter()
        ret, frame = cap.read()
        ms = (time.perf_counter() - t0) * 1000.0

        if not ret:
            continue

        buf[slot] = frame
        write_slot.value = slot
        capture_ms_out.value = ms
        frame_id_out.value = fid
        frame_ready.set()

        slot = (slot + 1) % RING_SIZE
        fid += 1

    cap.release()
    existing_shm.close()


# ──────────────────────────────────────────────
# Inference Process
# ──────────────────────────────────────────────

def _inference_worker(
    shm_name: str,
    read_slot: "mp.Value",
    frame_ready: "mp.Event",
    stop_event: "mp.Event",
    result_queue: Queue,
    capture_ms_in: "mp.Value",
    frame_id_in: "mp.Value",
    model_path: str,
    use_gpu: bool,
    use_onnx: bool,
    use_tensorrt: bool,
):
    """Reads frames from shared memory, runs hand tracker, pushes InferenceResult."""
    # Import inside worker to avoid pickling issues
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    sys.path.insert(0, str(Path(__file__).parent))

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray(
        (RING_SIZE, FRAME_H, FRAME_W, FRAME_C),
        dtype=np.uint8,
        buffer=existing_shm.buf,
    )

    try:
        if use_onnx:
            from tracker_onnx import OnnxHandTracker
            tracker = OnnxHandTracker(max_num_hands=2, use_tensorrt=use_tensorrt)
            print(f"[InferenceWorker] Backend: {'TensorRT' if use_tensorrt else 'ONNX CUDA'}", flush=True)
        else:
            from tracker import HandTracker
            tracker = HandTracker(
                max_num_hands=2,
                model_path=Path(model_path),
                use_gpu=use_gpu,
            )
            print(f"[InferenceWorker] Backend: MediaPipe ({'GPU' if use_gpu else 'CPU'})", flush=True)
    except Exception as e:
        print(f"[InferenceWorker] FATAL: tracker init failed — {e}", flush=True)
        existing_shm.close()
        return

    try:
        while not stop_event.is_set():
            fired = frame_ready.wait(timeout=0.05)
            if not fired:
                continue
            frame_ready.clear()

            slot = read_slot.value
            frame = buf[slot].copy()   # copy to avoid race condition during next write

            cap_ms = capture_ms_in.value
            fid = frame_id_in.value

            result = tracker.process(frame)

            hands_data = [
                (h.points, h.handedness) for h in result.hands
            ]

            # Encode frame as JPEG for display in main process (quality 60 ≈ ~30KB)
            ok, jpg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frame_jpeg = bytes(jpg_buf) if ok else b""

            result_queue.put(
                InferenceResult(
                    hands=hands_data,
                    capture_ms=cap_ms,
                    inference_ms=result.inference_ms,
                    frame_id=fid,
                    frame_jpeg=frame_jpeg,
                ),
                block=False,
            )
    except Exception as e:
        print(f"[InferenceWorker] ERROR in loop — {e}", flush=True)
    finally:
        tracker.close()
        existing_shm.close()


# ──────────────────────────────────────────────
# Pipeline coordinator
# ──────────────────────────────────────────────

class MultiprocessPipeline:
    """
    Manages the capture and inference processes.
    Main process calls .get_result() to retrieve the latest InferenceResult.
    """

    def __init__(
        self,
        camera_index: int = 0,
        model_path: Path | None = None,
        use_gpu: bool = False,
        use_onnx: bool = False,
        use_tensorrt: bool = False,
    ):
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "hand_landmarker.task"

        self._shm = shared_memory.SharedMemory(
            create=True, size=RING_SIZE * FRAME_BYTES
        )
        self._write_slot = mp.Value(ctypes.c_int, 0)
        self._frame_ready = mp.Event()
        self._stop = mp.Event()
        self._capture_ms = mp.Value(ctypes.c_double, 0.0)
        self._frame_id = mp.Value(ctypes.c_int, 0)
        self._result_queue: Queue = mp.Queue(maxsize=4)

        self._cap_proc = mp.Process(
            target=_capture_worker,
            args=(
                self._shm.name,
                self._write_slot,
                self._frame_ready,
                self._stop,
                camera_index,
                self._capture_ms,
                self._frame_id,
            ),
            daemon=True,
        )
        self._inf_proc = mp.Process(
            target=_inference_worker,
            args=(
                self._shm.name,
                self._write_slot,
                self._frame_ready,
                self._stop,
                self._result_queue,
                self._capture_ms,
                self._frame_id,
                str(model_path),
                use_gpu,
                use_onnx,
                use_tensorrt,
            ),
            daemon=True,
        )

    def start(self):
        self._cap_proc.start()
        self._inf_proc.start()

    def get_result(self, timeout: float = 0.05) -> InferenceResult | None:
        """Return the latest result, draining stale ones first."""
        result = None
        try:
            while True:
                result = self._result_queue.get_nowait()
        except Exception:
            pass

        if result is not None:
            return result

        # If queue was empty, block briefly for the next one
        try:
            return self._result_queue.get(timeout=timeout)
        except Exception:
            return None

    def stop(self):
        self._stop.set()
        self._cap_proc.join(timeout=2)
        self._inf_proc.join(timeout=2)
        self._shm.close()
        self._shm.unlink()
