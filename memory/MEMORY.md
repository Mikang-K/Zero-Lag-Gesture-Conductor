# Project Memory — Zero-Lag Gesture Conductor

## Current Phase
Phase 2 (Optimization) — 완료. Phase 3 (C++) 대기 중.

## Project Structure
```
zero-lag-gesture-conductor/
├── main.py               # Phase 1 entry point
├── main_phase2.py        # Phase 2 entry point (--backend, --pipeline 옵션)
├── requirements.txt
├── config/keymap.json
├── models/
│   ├── hand_landmarker.task
│   ├── hand_detector.tflite / .onnx
│   └── hand_landmarks_detector.tflite / .onnx
├── src/
│   ├── capture.py        # CameraCapture (cv2.CAP_DSHOW, BUFFERSIZE=1)
│   ├── tracker.py        # HandTracker (MediaPipe Tasks API, use_gpu 옵션)
│   ├── tracker_onnx.py   # OnnxHandTracker (ORT CUDA, full BlazePalm pipeline)
│   ├── pipeline_mp.py    # MultiprocessPipeline (SharedMemory ring buffer)
│   ├── recognizer.py     # GestureRecognizer (Tap/LongPress/Slide)
│   ├── emulator.py       # InputEmulator (pyautogui, PAUSE=0)
│   └── monitor.py        # PerformanceMonitor (HUD + CSV)
└── benchmarks/
    ├── compare.py
    └── phase1_metrics.csv
```

## Phase 2 벤치마크 결과 (RTX 4070, 120 frames)
| Backend           | Inf avg | Inf p95 | E2E avg | FPS  |
|-------------------|---------|---------|---------|------|
| MediaPipe CPU     | 8.3 ms  | 10.0 ms | 39.3 ms | 25.4 |
| ONNX RT CUDA      | 6.4 ms  | 9.0 ms  | 39.7 ms | 25.2 |

**병목:** Capture (~30ms) >> Inference (6-8ms)
→ 멀티프로세싱 파이프라인이 핵심 개선 방향 (캡처/추론 오버랩)

## 환경 제약 사항
- MediaPipe GPU delegate: Python 빌드에서 비활성화
- TensorRT: nvinfer_10.dll 미설치 → ORT CUDA로 폴백
- ORT cuDNN: PyTorch lib 경로 `os.add_dll_directory` 추가 (tracker_onnx.py 상단)
- tf2onnx NumPy 2.0 비호환 → np.cast 몽키패치로 우회 (변환은 이미 완료)

## OnnxHandTracker 내부 파이프라인
1. 192×192 resize, normalize [-1,1] → hand_detector.onnx (CUDA)
2. BlazePalm anchor 2016개 decode + sigmoid + NMS (threshold=0.5/0.3)
3. affine crop → 224×224 → hand_landmarks_detector.onnx (CUDA)
4. inverse affine → image-space landmark coords

## Gesture Thresholds (recognizer.py)
- TAP_VELOCITY_THRESHOLD = 0.035
- LONG_PRESS_DURATION_S = 0.55
- SLIDE_DISPLACEMENT_X = 0.15
- MovingAverage window = 5

## Python Environment
- Python 3.13.7, CUDA 12.6, RTX 4070
- opencv-python 4.13, mediapipe 0.10.32, onnxruntime-gpu 1.24.2
- pyautogui 0.9.54, numpy 2.2.6, onnx 1.20.1

## Phase 3 계획
- C++로 캡처+ONNX 추론 모듈 재작성 (OpenCV C++ + ONNX Runtime C++ API)
- PyBind11 or 독립 실행형 C++ 앱
