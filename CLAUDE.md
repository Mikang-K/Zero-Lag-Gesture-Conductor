# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language & Style
- **Communication Language:** All responses and explanations must be in Korean. (모든 답변과 설명은 한국어로 작성할 것)
- **Code Comments:** Keep code comments in English for universality. (코멘트는 관례에 따라 영어로 유지)

# [프로젝트 명세서] Zero-Lag Gesture Conductor

**부제: 실시간 엣지 AI 경량화 및 가속화를 통한 저지연 제스처 인식 시스템**

## 1. 프로젝트 개요

- **목표:** 웹캠 영상을 실시간으로 분석하여 사용자의 손동작을 인식하고, 이를 리듬 게임의 입력 신호로 변환하는 고성능 엣지 AI 시스템 개발.
- **핵심 가치:** 0.01초의 오차도 허용하지 않는 리듬 게임 환경을 위해 **모델 경량화(Lightweight)**와 **저지연(Low Latency)** 기술력을 증명함.
- **주요 타겟:** 온디바이스 AI, 실시간 비전 시스템, HMI(Human-Machine Interface) 관련 엔지니어링 직무.

## 2. 주요 기능 및 요구사항

1. **실시간 손가락/포즈 트래킹:** 웹캠 입력(30/60 FPS)에서 지연 없이 사용자의 관절 랜드마크 추출.
2. **제스처 인식 엔진:** 특정 좌표의 변화나 손모양을 분석하여 게임 입력(Tap, Long Press, Slide)으로 매핑.
3. **입력 에뮬레이션:** 인식된 제스처를 OS 수준의 키보드/마우스 이벤트로 변환하여 기존 리듬 게임과 연동.
4. **성능 모니터링:** 화면 상에 실시간 FPS, Inference Time(추론 시간), End-to-End Latency 측정값 표시.

## 3. 기술 스택 (예정)

- **Language:** Python (Prototype), C++ (Optimization)
- **AI/Vision:** MediaPipe, OpenCV, PyTorch
- **Optimization:** ONNX Runtime, TensorRT, Quantization (INT8/FP16)
- **OS:** Windows / Linux (Ubuntu)
- **Tools:** VS Code, Git, CMake

## 4. 개발 단계 (Milestones)

### Phase 1: Python Prototype (기능 구현 중심)

- OpenCV와 MediaPipe를 활용한 기본 제스처 인식 로직 개발.
- PyAutoGUI를 활용해 외부 게임(예: 웹 기반 리듬 게임) 조작 테스트.
- 현재 시스템의 병목 구간(Bottleneck) 파악 및 지연 시간 데이터 수집.

### Phase 2: Model Optimization (경량화 및 가속화)

- MediaPipe 모델을 ONNX 포맷으로 변환 및 하드웨어 가속기(TensorRT) 적용.
- 가중치 양자화(Weight Quantization)를 통한 추론 속도 향상.
- 멀티프로세싱을 적용하여 영상 캡처와 추론 로직 분리.

### Phase 3: C++ Implementation (고성능 이식)

- 가장 지연이 심한 모듈을 C++로 재작성.
- Python-C++ 인터페이스(PyBind11) 또는 독립 실행형 C++ 애플리케이션 구축.
- 최종 성능 지표 비교 (Python vs C++ 가속 버전).

## 5. 기대 효과 및 활용 방안

- **기술적 차별점:** 단순 라이브러리 활용을 넘어, 실질적인 **'성능 최적화 과정'**을 수치로 증명 가능.
- **확장성:** 향후 VR/AR 인터페이스, 장애인을 위한 보조 입력 장치, 비접촉 키오스크 시스템 등으로 기술 확장 가능.

Zero-Lag Gesture Conductor — 개발 단계 계획
Phase 1: Python Prototype (기능 구현 중심)
목표: 동작하는 End-to-End 파이프라인 구축 및 병목 구간 데이터 수집

1-1. 환경 세팅
 Python 가상환경 구성 (venv 또는 conda)
 의존성 설치: opencv-python, mediapipe, pyautogui, numpy
 프로젝트 폴더 구조 설계

zero-lag-gesture-conductor/
├── src/
│   ├── capture.py       # 웹캠 캡처 모듈
│   ├── tracker.py       # MediaPipe 랜드마크 추출
│   ├── recognizer.py    # 제스처 인식 엔진
│   ├── emulator.py      # 키보드/마우스 에뮬레이션
│   └── monitor.py       # 성능 측정 (FPS, Latency)
├── tests/
├── benchmarks/
└── main.py

1-2. 실시간 트래킹 구현
 OpenCV 웹캠 스트림 획득 (30/60 FPS 설정)
 MediaPipe Hands로 21개 손 랜드마크 추출
 시각화: 랜드마크 오버레이 렌더링
1-3. 제스처 인식 엔진 구현
 Tap: 특정 관절(예: 검지 끝)의 Y축 속도 임계값 감지
 Long Press: Tap 지속 시간 측정 (예: 300ms 이상)
 Slide: 손목 또는 손바닥 중심의 X축 이동 벡터 분석
 노이즈 필터링: 이동 평균(Moving Average) 또는 Kalman Filter 적용
1-4. 입력 에뮬레이션
 pyautogui로 인식된 제스처 → 키보드/마우스 이벤트 변환
 웹 기반 리듬 게임(예: Osu!, Friday Night Funkin')과 연동 테스트
 입력 딜레이 허용 범위 검증 (목표: < 16ms @ 60FPS)
1-5. 성능 모니터링 구현
 화면에 실시간 오버레이 표시
Capture Latency (웹캠 프레임 획득 시간)
Inference Time (MediaPipe 추론 시간)
End-to-End Latency (캡처 → 입력 에뮬레이션 총 시간)
FPS
 병목 구간 로그 CSV 저장 → Phase 2 최적화 우선순위 결정 자료로 활용
Phase 2: Model Optimization (경량화 및 가속화)
목표: Phase 1 대비 추론 속도 2~4배 향상, Latency < 10ms 달성

2-1. ONNX 변환
 MediaPipe 내부 TFLite 모델 → ONNX 포맷 변환
 ONNX Runtime으로 추론 파이프라인 교체
 ONNX Runtime CPU provider 벤치마크 비교
2-2. 하드웨어 가속
 ONNX Runtime GPU provider (CUDA): GPU 보유 시 우선 적용
 TensorRT 변환: ONNX → TensorRT engine 빌드 (NVIDIA GPU 필수)
 가중치 양자화:
FP32 → FP16 (반정밀도): 속도/정확도 균형 우선
FP32 → INT8 (정수 양자화): 최대 경량화 (정확도 하락 허용 범위 검증)
2-3. 병렬 처리 파이프라인
 multiprocessing 모듈로 프로세스 분리
Process A: 웹캠 캡처 → 공유 메모리(Shared Memory) 전달
Process B: 추론 및 제스처 인식
Process C: 입력 에뮬레이션
 Queue / SharedMemory로 프로세스 간 통신 설계 (GIL 회피)
2-4. 최적화 성과 측정
 Phase 1 대비 성능 지표 비교표 작성

|항목 |	Phase 1 (Baseline) | Phase 2 (최적화) |	개선율 |
| --- | --- | --- | --- |
|Inference Time | ? ms | ? ms | ? % |
| End-to-End Latency | ? ms	| ? ms | ? % |
| FPS | ? |	? |	? |

Phase 3: C++ Implementation (고성능 이식)
목표: 가장 지연이 심한 모듈의 C++ 재작성, 최종 성능 지표 확보

3-1. 병목 모듈 선정 및 C++ 재작성
 Phase 2 벤치마크 데이터를 근거로 C++ 이식 우선 모듈 결정
 CMake 빌드 시스템 구성
 OpenCV C++ API로 캡처 및 전처리 모듈 재작성
 ONNX Runtime C++ API로 추론 모듈 재작성
3-2. Python-C++ 연동
 옵션 A (우선): PyBind11로 C++ 모듈을 Python에서 호출 가능한 라이브러리로 노출
 옵션 B (심화): 독립 실행형 C++ 애플리케이션으로 완전 이식
3-3. 최종 성능 검증
 Python Baseline vs Python 최적화 vs C++ 버전 3-way 비교
 목표 지표:
End-to-End Latency < 5ms
FPS 60 이상 (안정적)
제스처 인식 정확도 > 95%
단계별 산출물 요약

| Phase | 주요 산출물 |
| --- | --- |
| Phase 1 |	동작하는 Python 프로토타입 |
| Phase 2 |	최적화된 추론 파이프라인 |
| Phase 3 | C++ 모듈 또는 독립 실행형 앱 |
