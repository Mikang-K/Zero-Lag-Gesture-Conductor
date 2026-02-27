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
- TAP fires on DOWNSTROKE immediately (onset_ms = 0) — rhythm-game optimised
- TAP_VELOCITY_THRESHOLD = 0.025 (downward, normalized/frame)
- RELEASE_VELOCITY_THRESHOLD = 0.010 (upward, gentler — resets DOWN state)
- TAP_COOLDOWN_S = 0.10 (100ms — max ~BPM 600, prevents double-fire)
- SLIDE_DISPLACEMENT_X = 0.15 (horizontal travel while in DOWN state)
- MovingAverage window = 5

## Python Environment
- Python 3.13.7, CUDA 12.6, RTX 4070
- opencv-python 4.13, mediapipe 0.10.32, onnxruntime-gpu 1.24.2
- pyautogui 0.9.54, numpy 2.2.6, onnx 1.20.1

## CameraCapture (capture.py)
- `threaded=True` (기본값): 백그라운드 스레드가 cap.read() 루프, main은 queue.get()으로 즉시 획득
- `threaded=False`: 기존 blocking cap.read() (레거시/비교용)
- 측정 latency: threaded → 프레임 staleness(큐 대기 시간), blocking → grab+decode 시간
- pipeline_mp.py 내부 _capture_worker는 별도 프로세스이므로 변경 없음

## compare.py 구조 (3개 섹션)
- Section 1: 트래커 백엔드 비교 (blocking read 고정)
- Section 2: 캡처 방식 비교 (ONNX CUDA × {blocking, threaded})
- Section 3: 리듬게임 기준 (E2E < 16.7ms) PASS/FAIL 요약

## Phase 3 계획
- C++로 캡처+ONNX 추론 모듈 재작성 (OpenCV C++ + ONNX Runtime C++ API)
- PyBind11 or 독립 실행형 C++ 앱
