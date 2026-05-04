# SAE-Guided Mechanism-Aware Multi-Signal Debiasing for BBQ

> **7개의 신뢰도 신호와 MoE 라우팅을 결합한 LLM 사회적 편향 완화 파이프라인**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Benchmark: BBQ](https://img.shields.io/badge/Benchmark-BBQ-green.svg)](https://github.com/nyu-mll/BBQ)

---

## 1. 연구 개요

LLM이 demographic 정보(성별, 인종, 종교 등)에 의존하여 편향된 답을 생성하는 문제를 완화하기 위한 **post-processing 디바이어싱 시스템**입니다. 모델의 답 자체를 바꾸지 않고, **답이 편향에 기반했는지 판별**하여 신뢰도가 낮을 때만 "unknown"으로 override합니다.

### 핵심 아이디어
- 같은 질문에 **4가지 prompt variant**로 응답을 받아 일관성을 측정합니다.
- **7개의 신뢰도 신호**(텍스트, logit, 모델 내부, SAE feature 등)를 추출합니다.
- **MoE Aggregator**가 카테고리별 cluster에 맞춰 신호를 가중 통합합니다.
- **Threshold override**로 confidence가 낮으면 "unknown" 처리합니다.

---

## 2. 4-Stage Pipeline

```
[Stage 1] 4-Prompt Inference
   같은 BBQ 질문에 4가지 prompt variant로 LLM 응답 수집
   → vanilla, debiasing_instruction, cot, counterfactual_swap

[Stage 2] 7-Signal Extraction
   각 instance에 대해 7개 신뢰도 신호 추출
   → s1~s7 (아래 표 참조)

[Stage 3] MoE Aggregator
   Question embedding으로 4 cluster 가중치 결정 + 4 expert MLP soft routing
   → p ∈ [0, 1] 정답 확신도

[Stage 4] Threshold Override
   p < tau → "unknown" override
   p ≥ tau → 모델 답 유지
```

---

## 3. 7개 신호 (Signals)

| 신호 | 이름 | 추출 방법 | 의미 |
|------|------|----------|------|
| **s1** | Evidence | context-answer token overlap | 답이 context에 명시적으로 있는지 |
| **s2** | Counterfactual Consistency | 그룹 swap 후 답 비교 | swap에도 답이 일관되는지 |
| **s3** | Self-Confidence | 첫 토큰 logit softmax | 모델이 답에 자신 있는지 |
| **s4** | Self-Consistency | n=5 sampling 다수결 비율 | 답이 흔들리는지 |
| **s5** | Bias-head Activation | 특정 attention head의 demographic token attention | bias head가 활성화되는지 |
| **s6** | Prompt Sensitivity | 4 prompt 답 일관성 | prompt에 robust한지 |
| **s7** | SAE Feature Activation | SAE의 bias-related feature 활성화 | 내부 회로가 편향에 의존하는지 |

> **Cross-LLM 비교**: Llama/Gemma는 SAE 지원으로 7-signal, Qwen은 SAE 미지원으로 6-signal version을 사용합니다.

---

## 4. MoE Cluster 정의

| Cluster | 카테고리 | 특성 |
|---------|---------|------|
| **1. Lexically-Substitutable** | Gender_identity, Religion | 단어 치환만으로 swap 가능 |
| **2. Numerically-Verifiable** | Age, SES | 숫자/명시적 정보로 검증 가능 |
| **3. Cultural-Contextual** | Race_ethnicity | 문화적 맥락 의존 |
| **4. Identity-Sensitive** | Disability_status, Sexual_orientation | 정체성 민감 |

각 cluster마다 별도 expert MLP가 학습되며, gating network가 question embedding을 보고 cluster 가중치(softmax)를 결정합니다.

---

## 5. 데이터셋

- **BBQ (Bias Benchmark for QA)**: 9개 사회 편향 카테고리 중 **7개 사용**.
- 사용 카테고리: Gender_identity, Race_ethnicity, Age, Religion, Disability_status, SES, Sexual_orientation.
- **카테고리당 300 instance random sampling** (seed=42, context_condition stratified).
- 총 **2,100 instance** 사용.

### 추가 평가 (zero-shot transfer)
- ImplicitBBQ
- OpenBiasBench

---

## 6. 모델

| 역할 | 모델 | 용도 |
|------|------|------|
| **Main** | meta-llama/Llama-3.1-8B-Instruct | 메인 실험 (SAE 지원: Llama-Scope) |
| **Cross-LLM** | google/gemma-2-9b-it | 일반화 검증 (SAE 지원: Gemma Scope) |
| **Cross-LLM** | Qwen/Qwen2.5-7B-Instruct | 일반화 검증 (SAE 미지원, 6-signal version) |

### SAE
- **Llama-Scope** (Fudan University): `fnlp/Llama3_1-8B-Base-LXR-32x`
- **Gemma Scope** (DeepMind): `google/gemma-scope-9b-it-res`

---

## 7. 평가 지표

| 지표 | 정의 |
|------|------|
| **accuracy_amb** | 모호 맥락 정확도 |
| **accuracy_dis** | 비모호 맥락 정확도 |
| **bias_score_amb** | 모호 맥락 편향 점수 ∈ [-1, 1] |
| **bias_score_dis** | 비모호 맥락 편향 점수 |
| **false_abstention_rate** | 비모호 맥락에서 unknown으로 답한 비율 (과교정 신호) |

### 통계 검증
- **1000-bootstrap 95% CI** (모든 지표)
- **Paired bootstrap p-value** (baseline 대비 개선의 유의성)

---

## 8. Baseline 비교

| Baseline | 출처 | 접근법 |
|----------|------|--------|
| Self-Debiasing-Reprompting | Gallegos et al. 2025 | 초기 답 후 편향 검토 재프롬프팅 |
| DeCAP | Bae et al. 2025 | Demographic Counterfactual-Augmented Prompting |
| FairSteer | Li et al. 2025 | Activation steering vector |
| Composite Prompting | - | 공정성 지시 + CoT + 역할 통합 |

---

## 9. 프로젝트 구조

```
project/
├── data/
│   ├── bbq/                              # BBQ 원본 JSONL
│   └── sampled/                          # 샘플링된 2,100 instance
├── src/
│   ├── signals/
│   │   ├── prompts.py                    # 4-prompt variant 정의
│   │   ├── inference.py                  # Stage 1: 4-prompt inference
│   │   ├── evidence.py                   # s1
│   │   ├── counterfactual.py             # s2
│   │   ├── confidence.py                 # s3
│   │   ├── consistency.py                # s4
│   │   ├── bias_head.py                  # s5
│   │   ├── prompt_sensitivity.py         # s6
│   │   ├── sae_feature.py                # s7
│   │   └── extract_all.py                # Stage 2: 7-signal 통합 추출
│   ├── models/
│   │   ├── moe_aggregator.py             # Stage 3: MoE 모델 정의
│   │   ├── trainer.py                    # MoE 학습 코드
│   │   ├── embedding.py                  # Question embedding 생성
│   │   └── override.py                   # Stage 4: Threshold override
│   ├── evaluation/
│   │   ├── bbq_evaluator.py              # BBQ 표준 지표
│   │   ├── bootstrap_ci.py               # 1000-bootstrap CI + paired p-value
│   │   └── baselines.py                  # 4가지 baseline 구현
│   └── utils/
│       ├── data_loader.py                # BBQ 로더, 샘플링
│       ├── sampling.py                   # 샘플링 실행 스크립트
│       └── llm_utils.py                  # LLMWrapper (Llama/Gemma/Qwen 통합)
├── notebooks/
│   ├── 01_data_exploration.ipynb         # (예정)
│   ├── 02_signal_extraction.ipynb        # (예정)
│   ├── 03_moe_training.ipynb             # (예정)
│   └── 04_evaluation.ipynb               # (예정)
├── configs/
│   └── default.yaml                      # 모든 하이퍼파라미터
├── results/                              # 실험 결과
├── tests/                                # unit tests
├── requirements.txt
├── README.md
└── run_pipeline.py                       # End-to-End 실행 스크립트
```

---

## 10. 환경 및 설치

### 환경
- **메인 개발**: Mac M4 Pro 64GB (PyTorch with MPS)
- **Cross-LLM 일부**: RunPod A100 (PyTorch with CUDA)

### 설치
```bash
git clone https://github.com/KMS-gif375/LLM-Bias-Mitigation.git
cd LLM-Bias-Mitigation
pip install -r requirements.txt
```

HuggingFace 토큰 설정 (.env):
```
HF_TOKEN=your_huggingface_token
```

### 실행 순서

```bash
# 0. BBQ 데이터 다운로드 (data/bbq/에 11개 JSONL 배치)

# 1. 카테고리당 300개 샘플링
python -m src.utils.sampling

# 2. Stage 1: 4-Prompt Inference
python run_pipeline.py --model main --stages 1

# 3. Stage 2: 7-Signal Extraction
python run_pipeline.py --model main --stages 2

# 4. Stage 3: MoE 학습 (notebook 권장)
jupyter notebook notebooks/03_moe_training.ipynb

# 5. Stage 4: 평가 (notebook 권장)
jupyter notebook notebooks/04_evaluation.ipynb
```

---

## 11. 코딩 원칙

1. **모듈화**: 각 신호 추출은 독립 파일.
2. **재현성**: seed 고정(42), 모든 중간 결과 JSONL 저장.
3. **가독성**: docstring + type hint 필수.
4. **테스트**: 각 모듈 unit test (`tests/`).
5. **로깅**: tqdm으로 진행 상황 출력.
6. **메모리**: Mac MPS 환경 고려, batch 작게.
7. **체크포인트**: instance 단위로 중간 저장 → 재실행 가능.

---

## 12. 한계 및 향후 연구

### 한계
- **벤치마크 의존성**: 라벨된 BBQ 사용. 실제 배포 시 모호/비모호 자동 판별 모듈 필요.
- **영어 중심**: 다른 언어/문화권으로 확장은 추가 검증 필요.
- **객관식 형식**: 자유 형식 답변 대응은 별도 연구.
- **SAE 의존성**: Qwen은 SAE 미지원 → 6-signal version으로 대체.

### 향후 연구
- 모호/비모호 자동 판별 (모델 신뢰도 기반).
- 다국어 BBQ로 확장 (KoBBQ, JBBQ 등).
- Open-BBQ에서의 자유 형식 평가.
- Cluster 자동 발견 (현재는 수동 정의).

---

## License

MIT License — see [LICENSE](LICENSE).
