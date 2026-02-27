"""
tracker_onnx.py — ONNX Runtime GPU-accelerated hand landmark tracker (Phase 2)

Replaces MediaPipe Tasks API with a custom pipeline running on ORT CUDA/TensorRT:
  1. BlazePalm detector  (hand_detector.onnx      — 192×192 input)
  2. Hand landmark model (hand_landmarks_detector.onnx — 224×224 input)

Pipeline per frame:
  BGR frame
    → resize 192×192, normalize [-1,1]
    → ORT detector  → [1,2016,18] boxes + [1,2016,1] scores
    → anchor decode + sigmoid + NMS
    → affine crop of best detection → 224×224
    → ORT landmark  → 21 landmarks (normalized to crop space)
    → inverse affine → image-space landmark coordinates
"""

import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np

# Make ORT CUDA provider find cuDNN bundled inside PyTorch
try:
    import torch
    _torch_lib = Path(torch.__file__).parent / "lib"
    if _torch_lib.exists() and sys.platform == "win32":
        os.add_dll_directory(str(_torch_lib))
except ImportError:
    pass

import onnxruntime as ort

from tracker import HandLandmarks, LandmarkIndex, TrackResult


# ──────────────────────────────────────────────────────────
# Constants matching MediaPipe BlazePalm configuration
# ──────────────────────────────────────────────────────────

_DETECTOR_SIZE = 192
_LANDMARK_SIZE = 224

_ANCHOR_STRIDES = [8, 16, 16, 16]
_MIN_SCALE = 0.1484375
_MAX_SCALE = 0.75
_SCORE_THRESHOLD = 0.50 # raised from 0.5 — reduces background false positives
_NMS_IOU_THRESHOLD = 0.3
_ROI_SCALE_FACTOR = 2.6    # padding multiplier around detected palm
_ROI_SHIFT_Y = -0.15       # shift ROI upward slightly to include wrist


# ──────────────────────────────────────────────────────────
# Anchor generation
# ──────────────────────────────────────────────────────────

def _generate_anchors() -> np.ndarray:
    """
    Generate BlazePalm SSD anchors matching MediaPipe's configuration.
    Returns float32 array of shape [2016, 4]: (cx, cy, w, h) in [0,1].
    """
    anchors = []
    num_layers = len(_ANCHOR_STRIDES)
    for layer_id, stride in enumerate(_ANCHOR_STRIDES):
        scale = _MIN_SCALE + (_MAX_SCALE - _MIN_SCALE) * layer_id / (num_layers - 1)
        next_scale = (
            _MIN_SCALE + (_MAX_SCALE - _MIN_SCALE) * (layer_id + 1) / (num_layers - 1)
            if layer_id + 1 < num_layers else 1.0
        )
        interp_scale = float(np.sqrt(scale * next_scale))
        fm = _DETECTOR_SIZE // stride
        for y in range(fm):
            for x in range(fm):
                cx = (x + 0.5) / fm
                cy = (y + 0.5) / fm
                anchors.append([cx, cy, scale, scale])
                anchors.append([cx, cy, interp_scale, interp_scale])

    return np.array(anchors, dtype=np.float32)   # [2016, 4]


_ANCHORS = _generate_anchors()


# ──────────────────────────────────────────────────────────
# SSD decode + NMS helpers
# ──────────────────────────────────────────────────────────

def _decode_boxes(raw_boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """
    Decode SSD raw predictions into [cx, cy, w, h, kp0_x, kp0_y, ..., kp6_x, kp6_y].
    raw_boxes: [N, 18], anchors: [N, 4] (cx, cy, w, h)
    Returns: [N, 18] with coordinates in [0,1] image-normalized space.

    Box center offsets use anchor scale; w/h and keypoints are in raw pixel
    units of the 192×192 detector input (divide by _DETECTOR_SIZE only).
    """
    out = raw_boxes.copy()
    # Box center: offset scaled by anchor size, added to anchor center
    out[:, 0] = raw_boxes[:, 0] / _DETECTOR_SIZE * anchors[:, 2] + anchors[:, 0]  # cx
    out[:, 1] = raw_boxes[:, 1] / _DETECTOR_SIZE * anchors[:, 3] + anchors[:, 1]  # cy
    # Box size: raw pixel displacement relative to 192×192 input (no anchor scale)
    out[:, 2] = raw_boxes[:, 2] / _DETECTOR_SIZE                                   # w
    out[:, 3] = raw_boxes[:, 3] / _DETECTOR_SIZE                                   # h
    # Keypoints: same encoding as center
    for k in range(7):
        out[:, 4 + k * 2]     = raw_boxes[:, 4 + k * 2]     / _DETECTOR_SIZE * anchors[:, 2] + anchors[:, 0]
        out[:, 4 + k * 2 + 1] = raw_boxes[:, 4 + k * 2 + 1] / _DETECTOR_SIZE * anchors[:, 3] + anchors[:, 1]
    return out


def _iou(box_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU of one box vs many, cx/cy/w/h format."""
    ax1 = box_a[0] - box_a[2] / 2
    ay1 = box_a[1] - box_a[3] / 2
    ax2 = box_a[0] + box_a[2] / 2
    ay2 = box_a[1] + box_a[3] / 2

    bx1 = boxes_b[:, 0] - boxes_b[:, 2] / 2
    by1 = boxes_b[:, 1] - boxes_b[:, 3] / 2
    bx2 = boxes_b[:, 0] + boxes_b[:, 2] / 2
    by2 = boxes_b[:, 1] + boxes_b[:, 3] / 2

    inter_w = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    inter_h = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter   = inter_w * inter_h
    union   = box_a[2] * box_a[3] + boxes_b[:, 2] * boxes_b[:, 3] - inter
    return inter / (union + 1e-6)


def _nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    """Weighted-score NMS. Returns indices of surviving detections."""
    order = np.argsort(scores)[::-1]
    kept = []
    while order.size:
        i = order[0]
        kept.append(int(i))
        if order.size == 1:
            break
        ious = _iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious < _NMS_IOU_THRESHOLD]
    return kept


# ──────────────────────────────────────────────────────────
# Affine crop helpers
# ──────────────────────────────────────────────────────────

def _compute_roi_transform(
    detection: np.ndarray,
    img_h: int,
    img_w: int,
) -> tuple[np.ndarray, float, float, float, float]:
    """
    Compute the affine matrix to crop the hand ROI from the image.
    Returns (M_3x3, roi_cx_px, roi_cy_px, roi_w_px, roi_h_px).
    """
    cx, cy, w, h = detection[:4]
    # Keypoints 0=wrist, 2=middle-MCP define the rotation
    kp0_x, kp0_y = detection[4], detection[5]
    kp2_x, kp2_y = detection[8], detection[9]

    # Rotation angle: angle from wrist to middle MCP, shifted so "up" = 0
    angle = np.arctan2(kp2_y - kp0_y, kp2_x - kp0_x) - np.pi / 2.0

    # ROI center slightly shifted toward fingertips
    roi_cx = cx + np.sin(angle) * _ROI_SHIFT_Y * h
    roi_cy = cy - np.cos(angle) * _ROI_SHIFT_Y * h

    # ROI size with padding
    side = max(w, h) * _ROI_SCALE_FACTOR

    # Convert to pixel coords
    roi_cx_px = roi_cx * img_w
    roi_cy_px = roi_cy * img_h
    roi_w_px  = side * img_w
    roi_h_px  = side * img_h

    # Affine: rotate around ROI center, then scale to LANDMARK_SIZE
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    scale_x = _LANDMARK_SIZE / roi_w_px
    scale_y = _LANDMARK_SIZE / roi_h_px

    # Build rotation + scale matrix
    M = np.array([
        [ cos_a * scale_x, sin_a * scale_x, (-roi_cx_px * cos_a - roi_cy_px * sin_a) * scale_x + _LANDMARK_SIZE / 2],
        [-sin_a * scale_y, cos_a * scale_y, ( roi_cx_px * sin_a - roi_cy_px * cos_a) * scale_y + _LANDMARK_SIZE / 2],
    ], dtype=np.float64)

    return M, roi_cx_px, roi_cy_px, roi_w_px, roi_h_px


def _inverse_affine(M: np.ndarray) -> np.ndarray:
    """2×3 affine inverse → 2×3."""
    M_full = np.vstack([M, [0, 0, 1]])
    return np.linalg.inv(M_full)[:2]


# ──────────────────────────────────────────────────────────
# ORT Session factory
# ──────────────────────────────────────────────────────────

def _make_session(model_path: str, providers: list[str]) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Thread count: 1 so worker process doesn't contend with main
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


# ──────────────────────────────────────────────────────────
# Main tracker class
# ──────────────────────────────────────────────────────────

class OnnxHandTracker:
    """
    Hand landmark tracker using ONNX Runtime with CUDA (or TensorRT) backend.
    Drop-in replacement for HandTracker in terms of process() / close() API.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        model_dir: Path | None = None,
        use_tensorrt: bool = False,
    ):
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / "models"

        det_path = str(model_dir / "hand_detector.onnx")
        lm_path  = str(model_dir / "hand_landmarks_detector.onnx")

        if use_tensorrt:
            providers = [
                ("TensorrtExecutionProvider", {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(model_dir / "trt_cache"),
                }),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._det_sess = _make_session(det_path, providers)
        self._lm_sess  = _make_session(lm_path,  providers)
        self._max_hands = max_num_hands

        # Cache input names
        self._det_input  = self._det_sess.get_inputs()[0].name
        self._lm_input   = self._lm_sess.get_inputs()[0].name

        # Warm up GPU kernels
        dummy_det = np.zeros((1, _DETECTOR_SIZE, _DETECTOR_SIZE, 3), dtype=np.float32)
        dummy_lm  = np.zeros((1, _LANDMARK_SIZE, _LANDMARK_SIZE, 3), dtype=np.float32)
        self._det_sess.run(None, {self._det_input: dummy_det})
        self._lm_sess.run(None, {self._lm_input: dummy_lm})

    # ──────────────────────────────────
    def process(self, frame_bgr: np.ndarray) -> TrackResult:
        t0 = time.perf_counter()

        img_h, img_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── 1. Palm detection ──
        det_input = cv2.resize(rgb, (_DETECTOR_SIZE, _DETECTOR_SIZE))
        det_input = det_input.astype(np.float32) / 255.0            # [0, 1]
        det_input = det_input[np.newaxis]                            # [1,192,192,3]

        raw_boxes_batch, raw_scores_batch = self._det_sess.run(
            None, {self._det_input: det_input}
        )
        raw_boxes  = raw_boxes_batch[0]    # [2016, 18]
        raw_scores = raw_scores_batch[0]   # [2016, 1]

        # Clip before exp to avoid overflow warning (values outside ±88 are saturated anyway)
        scores = 1.0 / (1.0 + np.exp(-np.clip(raw_scores[:, 0], -88.0, 88.0)))
        mask   = scores > _SCORE_THRESHOLD
        if not mask.any():
            inference_ms = (time.perf_counter() - t0) * 1000.0
            return TrackResult(hands=[], inference_ms=inference_ms)

        sel_boxes  = _decode_boxes(raw_boxes[mask], _ANCHORS[mask])
        sel_scores = scores[mask]
        kept_idx   = _nms(sel_boxes, sel_scores)
        kept_idx   = kept_idx[:self._max_hands]

        # ── 2. Landmark detection (one pass per detected hand) ──
        hands: list[HandLandmarks] = []
        for idx in kept_idx:
            det = sel_boxes[idx]
            M, _, _, _, _ = _compute_roi_transform(det, img_h, img_w)

            # Warp image to 224×224 crop
            crop = cv2.warpAffine(
                rgb, M, (_LANDMARK_SIZE, _LANDMARK_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            lm_input = (crop.astype(np.float32) / 255.0)[np.newaxis]   # [1,224,224,3], [0,1]

            lm_out = self._lm_sess.run(None, {self._lm_input: lm_input})
            lm_coords   = lm_out[0][0]    # [63] = 21×3 in crop [0,224] space
            presence    = float(lm_out[1][0][0])   # hand flag: high = hand present
            handedness  = float(lm_out[2][0][0])   # >0.5 → Right

            if presence < 0.5:
                continue

            # Inverse affine: crop coords → image coords
            M_inv = _inverse_affine(M)
            pts_crop = lm_coords.reshape(21, 3)[:, :2]   # [21, 2] in [0,224]

            # Apply inverse affine (homogeneous)
            pts_hom = np.hstack([pts_crop, np.ones((21, 1))])  # [21, 3]
            pts_img = (M_inv @ pts_hom.T).T                     # [21, 2] in image px

            # Normalize to [0, 1]
            pts_norm_x = pts_img[:, 0] / img_w
            pts_norm_y = pts_img[:, 1] / img_h
            pts_z      = lm_coords.reshape(21, 3)[:, 2] / _LANDMARK_SIZE

            points = [
                (float(pts_norm_x[i]), float(pts_norm_y[i]), float(pts_z[i]))
                for i in range(21)
            ]
            hand_label = "Right" if handedness > 0.5 else "Left"
            hands.append(HandLandmarks(points=points, handedness=hand_label))

        inference_ms = (time.perf_counter() - t0) * 1000.0
        return TrackResult(hands=hands, inference_ms=inference_ms)

    def draw(self, frame_bgr: np.ndarray, result: TrackResult) -> np.ndarray:
        """Draw minimal landmark dots + connections without MediaPipe drawing utils."""
        if not result.hands:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]
        for hand in result.hands:
            pts = [(int(p[0]*w), int(p[1]*h)) for p in hand.points]
            for a, b in CONNECTIONS:
                cv2.line(frame_bgr, pts[a], pts[b], (0, 220, 0), 1, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame_bgr, pt, 3, (0, 0, 255), -1)
        return frame_bgr

    def close(self):
        pass   # ORT sessions are released by GC
