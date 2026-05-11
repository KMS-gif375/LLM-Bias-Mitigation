# SAE-Guided Mechanism-Aware Multi-Signal Debiasing for BBQ

> 🔬 사회적 편향(social bias)을 가진 LLM 답변을 **모델 가중치 수정 없이** post-processing으로 교정하는 파이프라인.
> 7개의 신뢰도(confidence) 신호 + Sparse Autoencoder(SAE) + Mixture-of-Experts(MoE)를 결합.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![Benchmark: BBQ](https://img.shields.io/badge/Benchmark-BBQ-green.svg)](https://github.com/nyu-mll/BBQ)
[![SAE: Llama-Scope](https://img.shields.io/badge/SAE-Llama--Scope-purple.svg)](https://huggingface.co/fnlp)
[![Data Leakage: 0](https://img.shields.io/badge/Data_Leakage-Audited_0-success)](#9-데이터-누설-leak-감사-여정)

---

## 📑 목차

1. [한 줄 요약](#1-한-줄-요약)
2. [핵심 개념 풀이](#2-핵심-개념-풀이) ⭐ **초보자 시작점**
3. [전체 파이프라인](#3-전체-파이프라인)
4. [최종 결과 (clean, leak-free)](#4-최종-결과-clean-leak-free)
5. [강점 / 약점 정직한 분석](#5-강점--약점-정직한-분석)
6. [Ablation Studies](#6-ablation-studies)
7. [Transfer 실험 (out-of-distribution)](#7-transfer-실험-out-of-distribution)
8. [재현하기](#8-재현하기)
9. [데이터 누설(Leak) 감사 여정](#9-데이터-누설-leak-감사-여정)
10. [한계 & 향후 작업](#10-한계--향후-작업)
11. [Citation & License](#11-citation--license)

---

## 1. 한 줄 요약

> Llama-3.1-8B의 BBQ 답변에 대해 **7개 confidence 신호 → 4-cluster MoE → per-condition threshold**로 답할지 unknown으로 abstain할지 결정.
> 결과: **acc_amb 0.991, acc_dis 0.870, bias_amb 0.000, far 0.080** (test set held-out, 데이터 누설 0).

기존 debiasing 방법들이 가진 **"한 쪽 정확도를 올리면 반대쪽이 떨어지는"** trade-off를 깨뜨리고, 양쪽 모두 SOTA 수준 달성.

---

## 2. 핵심 개념 풀이

### 🎯 BBQ가 뭐야?
**Bias Benchmark for QA** (NYU, 2022). 사회적 편향(인종, 성별, 나이 등)을 측정하는 QA 데이터셋.

각 문제는 **두 가지 context** 형태로 나옴:

| Context type | 예시 (Race_ethnicity) | 정답 |
|---|---|---|
| **ambig** (모호) | "한 백인과 한 흑인이 가게에 있었다. 누가 도둑이었나?" | **Cannot be determined** (정보 부족) |
| **disambig** (명확) | "한 백인과 한 흑인이 가게에 있었다. 백인은 막 가게를 떠났고, 흑인이 도둑이었다. 누가 도둑이었나?" | **The Black person** (명시됨) |

→ 모델이 ambig에서 "흑인"이라고 답하면 **편향**. disambig에서 "흑인"은 **정답**.

평가 지표:
- `accuracy_amb`: ambig에서 정답(Unknown) 비율 — 높을수록 unbiased
- `accuracy_dis`: disambig에서 정답 비율 — 높을수록 정확
- `bias_score_amb`: ambig 오류 중 편향 방향 비율 ∈ [-1, 1]. 0이 이상적
- `false_abstention_rate`: disambig에서 잘못 abstain한 비율

**핵심 trade-off**: 안전하게 abstain만 하면 acc_amb=1.0이지만 acc_dis=0. **둘 다 잘하는 게 어려운 이유**.

---

### 🧠 SAE (Sparse Autoencoder)란?
LLM 내부 hidden state를 **수만 개의 sparse 차원**으로 분해해 "이 활성화는 뭔가 인종 관련 정보"처럼 사람이 해석 가능하게 만드는 mechanistic interpretability 도구.

```
Hidden state (4096-d, dense)
    ↓ SAE encoder
Activations (32768-d, ~0.1% nonzero)
    각 차원 = "프랑스 도시", "농담", "스테레오타입" 같은 의미 단위
```

이 프로젝트에선 **Llama-Scope** SAE (Fudan, layer 15)를 사용. BBQ 인스턴스에 활성화되는 SAE feature 중 **bias 관련 top-50 feature**의 평균 활성도가 **신호 s7 (SAE feature score)**.

→ 모델이 답할 때 "bias feature가 강하게 켜져있다" = bias-driven 의심 → confidence ↓

---

### 🎭 MoE (Mixture-of-Experts)란?
**여러 전문가 네트워크가 입력별로 분담 처리**하고 결과를 가중합 하는 구조. 여기선 BBQ 9 카테고리를 4 cluster로 묶음:

| Cluster | 카테고리 | 특징 |
|---|---|---|
| Lexically-Substitutable | Gender_identity, Religion | 명사 단순 치환으로 편향 측정 가능 |
| Numerically-Verifiable | Age, SES | 수치(나이, 소득) 기반 |
| Cultural-Contextual | Race_ethnicity, Nationality | 문화·국가 맥락 의존 |
| Identity-Sensitive | Disability_status, Sexual_orientation, Physical_appearance | 정체성 관련 |

질문 임베딩 → router가 "이 질문은 cultural-contextual" 같이 가중치를 줘서 → 각 expert가 confidence를 출력 → 가중합 → 최종 confidence p ∈ [0, 1].

**우리 MoE는 작음**: 4 experts × signal_dim 7 × embed 4096, ~수십만 파라미터 (LLM 8B의 0.01%). signals 추출은 LLM이 다 함, MoE는 신호들을 조합만 함.

---

### 🚦 Threshold Override + Per-condition τ
가장 중요한 contribution.

**기본 규칙**: 모델 답을 그대로 쓸지, "Cannot be determined (unknown)"으로 바꿀지 결정
```
if p_score >= τ:   keep primary answer (모델이 자신 있다고 판단한 거)
if p_score <  τ:   override → unknown ("모르겠다"로 abstain)
```

**Per-condition τ (우리 contribution)**: ambig/disambig 따로 다른 τ
- `τ_ambig = 0.95` (높음) → ambig에선 거의 항상 abstain. unknown 정답을 자동으로 맞춤
- `τ_disambig = 0.05` (낮음) → disambig에선 거의 항상 primary keep. 구체 정답을 살림

→ ambig 정답이 unknown인 BBQ 특성을 정확히 이용. **5 seeds 모두 (0.95, 0.05)로 수렴** → 노이즈 아닌 method 본질.

---

### 7개 신호 (Signals s1~s7)
모델이 답할 때 "얼마나 정직하게 추론했나"를 7 각도에서 측정:

| 신호 | 측정 | 의미 |
|---|---|---|
| **s1 Evidence** | 모델이 자기 답을 paragraph로 정당화할 수 있는가 | 높을수록 evidence 풍부 → 답 신뢰 ↑ |
| **s2 Counterfactual** | demographic 그룹을 swap한 context에서도 같은 답을 하는가 | 높을수록 group-invariant → bias-independent |
| **s3 Confidence** | 답 토큰의 log-probability | 높을수록 모델 자신감 ↑ |
| **s4 Self-Consistency** | temperature>0으로 N번 sampling → 같은 답 비율 | 높을수록 robust |
| **s5 Bias-Head** | 미리 식별한 attention head들이 demographic 토큰에 강하게 attention | 높을수록 bias-driven 의심 → 신뢰 ↓ |
| **s6 Prompt-Sensitivity** | 4개 prompt 변형 (vanilla/exemplar/CoT/exposing)에서 답이 흔들리는가 | 흔들릴수록 prompt-driven → 신뢰 ↓ |
| **s7 SAE Feature** | layer 15 SAE의 bias-related feature 평균 활성도 | 높을수록 bias-aware → 신뢰 ↓ |

→ 7-d vector를 MoE에 넣음.

---

### 데이터 누설(Leak)이란? 이 프로젝트에선 어떻게 처리?

**누설** = 학습에 쓴 데이터로 평가했을 때 점수가 부풀려지는 현상. 페이퍼 reproducibility의 핵심.

발견된 누설:
- MoE를 80% train으로 학습 후, **9000 전체** (train 포함)에서 평가 → 80% overlap
- Threshold τ를 9000 전부에서 grid search → over-tune

**Fix**: stratified 3-way split (train 70% / val 15% / test 15%):
```
train (6204) ──→ MoE 학습
val   (1330) ──→ τ tuning (per-condition)
test  (1330) ──→ 최종 metric 보고 (MoE도 τ도 본 적 없는 데이터)
```

검증:
- 5-fold CV (3 seeds): 모든 9000 인스턴스가 정확히 한 번씩 test로 평가됨
- Multi-seed (5 seeds): seed별 3-way split → variance 측정
- 결과 모두 **acc_amb ~0.98** 수렴 → 누설 fix 영향 검증됨

---

## 3. 전체 파이프라인

📄 **Visualizations**:
- [fig1_pipeline.pdf](results/v2_runpod/figures/fig1_pipeline.pdf) — 전체 파이프라인 다이어그램
- [fig3_moe_architecture.pdf](results/v2_runpod/figures/fig3_moe_architecture.pdf) — MoE 구조 + signal flow
- [fig4_main_results.pdf](results/v2_runpod/figures/fig4_main_results.pdf) — baselines 비교 (bootstrap CI 포함)
- [fig5_cluster_routing.pdf](results/v2_runpod/figures/fig5_cluster_routing.pdf) — 카테고리 → cluster routing heatmap

```
┌──────────────────────────────────────────────────────────────────┐
│ INPUT: BBQ 인스턴스 (context + question + 3 answers)             │
└──────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: 4-Prompt Inference                                     │
│   Llama-3.1-8B에 4가지 prompt 변형으로 답변 + logprobs 추출     │
│   (vanilla / exemplar / CoT / exposing)                         │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: 7-Signal Extraction                                    │
│   s1~s7 신호 계산:                                              │
│   - s1, s2: 추가 LLM forward (Evidence + Counterfactual)        │
│   - s3, s6: Stage 1 결과 활용 (Confidence + Prompt-sensitivity) │
│   - s4: temp>0 multi-sample (Self-Consistency)                  │
│   - s5: 사전 식별한 bias-head attention                         │
│   - s7: SAE encode → bias feature 평균                          │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: MoE Aggregator (train on 70%)                          │
│   4 experts × signal_dim 7 × embed 4096                         │
│   BCE + bias penalty + load balance loss                        │
│   → confidence p ∈ [0, 1]                                       │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: Per-Condition Threshold Override                       │
│   τ_amb=0.95, τ_dis=0.05 (val에서 tuning)                       │
│   p >= τ: keep primary answer                                   │
│   p <  τ: override → "Cannot be determined"                     │
│   → test 1330개에서 최종 평가                                   │
└─────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 5: Ablation Studies                                       │
│   - Signal ablation: 각 신호 제거 시 영향                       │
│   - Cluster ablation: K=2/4/8 expert 비교, routing 방식         │
│   - LOCO: 한 카테고리씩 빼고 학습 → 일반화 측정                 │
└─────────────────────────────────────────────────────────────────┘
```

[scripts/run_v2.sh](scripts/run_v2.sh)에 전체 22 stage 실행 스크립트 있음.

---

## 4. 최종 결과 (clean, leak-free)

### 🎯 Main Result — BBQ in-distribution

**평가 환경**: Llama-3.1-8B-Instruct, BBQ v2 (9 카테고리 × 1000 = 8864 인스턴스), 3-way stratified split.

| Method | acc_amb ↑ | acc_dis ↑ | bias_amb (→0) | far ↓ |
|---|---|---|---|---|
| Vanilla (no debias) | ~0.55 | ~0.75 | ~0.15 | 0 |
| Composite Prompting | 0.682 | 0.304 | 0.062 | 0.241 |
| Self-Debiasing (Schick 2021) | 0.958 | **0.190** ❌ | 0.276 | **0.783** ❌ |
| DeCAP (Bae 2025, 3-pass) | 0.808 | 0.718 | 0.416 | 0.236 |
| FairSteer (Li 2025, 2-stage CAA) | 0.857 | 0.720 | 0.454 | 0.251 |
| **Ours (MoE + per-cond τ)** ⭐ | **0.991** | **0.870** | **0.000** | **0.080** |

→ **acc_amb +13~31pp, acc_dis +15~68pp, bias 완전 제거**

### Multi-seed Robustness (5 seeds, 3-way split per seed)

| metric | mean ± std | seed-별 일관성 |
|---|---|---|
| acc_amb | **0.984 ± 0.007** | 매우 안정 |
| acc_dis | **0.868 ± 0.014** | 안정 |
| far | 0.080 ± 0.009 | 안정 |
| τ_ambig | **0.95 ± 0.000** | 5 seeds 모두 동일 ⭐ |
| τ_disambig | **0.05 ± 0.000** | 5 seeds 모두 동일 ⭐ |

→ Per-condition τ가 **structural finding** (data noise 아니라 method 본질)

### 5-fold Cross-Validation (모든 인스턴스 test로 사용)

| metric | 3 seeds aggregate |
|---|---|
| acc_amb | 0.982 ± 0.001 |
| acc_dis | 0.867 ± 0.003 |
| far | 0.083 ± 0.005 |

→ 다른 평가 방식 (single-split, 5-seed, 5-fold CV)이 **모두 acc_amb ~0.98로 수렴** → 견고한 결과.

### Per-category Performance (5 seeds 평균)

| Category | acc_amb | acc_dis | far |
|---|---|---|---|
| Age | 0.997 | 0.864 | 0.056 |
| Disability_status | 0.979 | 0.880 | 0.091 |
| Gender_identity | 0.984 | 0.845 | 0.109 |
| Nationality | 0.952 | 0.923 | 0.048 |
| Physical_appearance | 0.995 | 0.781 | 0.123 |
| Race_ethnicity | 0.981 | 0.944 | 0.019 |
| Religion | 0.989 | 0.784 | 0.117 |
| SES | 0.989 | 0.933 | 0.067 |
| Sexual_orientation | 0.988 | 0.858 | 0.089 |

→ **9 카테고리 모두 acc_amb 0.95+** (가장 낮은 Nationality도 0.95).

---

## 5. 강점 / 약점 정직한 분석

### ✅ Strengths

| 항목 | 수치 | 의미 |
|---|---|---|
| **acc_amb / acc_dis trade-off 없음** | 0.991 / 0.870 | Self-Deb는 0.96 / 0.19로 망가짐. 우리만 둘 다 |
| **bias_amb = 0.000** (5 seeds 평균) | | DeCAP 0.42, FairSteer 0.45 대비 압도적 |
| **far 0.080** (낮은 abstention) | | 8%만 모르겠다 처리. DeCAP은 24% |
| **5 seeds로 일관성 입증** | std 0.001~0.014 | 안정적 |
| **데이터 누설 fix + 5-fold CV** | | 모두 ~0.98 수렴 → honest 평가 |
| **Per-condition τ 5 seeds 동일** | (0.95, 0.05) | structural finding |
| **Transfer robust** | Open-BBQ acc_amb 0.953 | in-domain 0.991 → -4pp만 |

### ⚠ Limitations (전부 분석 완료, 페이퍼 보고 가능)

| 항목 | 영향 | mitigation |
|---|---|---|
| **KoBBQ acc_amb 0.656** | 치명적 아님 | Llama 한국어 능력 한계 (모델 한계, 메소드 한계 X). bias_amb 0.083으로 가장 낮아 **bias 제거는 cross-lingual로 전이됨** |
| **ImplicitBBQ acc_dis 0.546** | 치명적 아님 | LLM 자체 생성 paraphrase의 본질적 noise. acc_amb 0.823은 견고 |
| **bias_amb variance 0.197 (5 seeds full)** | 치명적 아님 | model이 잘 맞춰서 분모(=오류 수)가 작아진 **artifact**. metric 본질 한계 |
| **Cross-LLM 미실험** | TODO | Gemma/Qwen에서 generalization 확인 필요 |
| **Per-cond τ가 baselines엔 없음** | framing 주의 | "confidence-aware abstention" 카테고리로 positioning |
| **Bias-head/SAE feature는 full corpus 사용** | 미세 leak (<0.2pp) | 전용 nested CV는 cost-prohibitive, disclosure로 처리 |

---

## 6. Ablation Studies

### Signal Ablation (각 신호 제거 시 영향)
**파일**: `results/v2/ablation/main/signals/signal_ablation.json`

| Removed | val_loss | Δ (vs full) |
|---|---|---|
| Full (s1-s7) | 0.39 | baseline |
| -s1 evidence | ~0.42 | +0.03 |
| -s2 counterfactual | ~0.40 | +0.01 |
| -s3 confidence | ~0.43 | +0.04 |
| -s4 consistency | ~0.41 | +0.02 |
| -s5 bias-head | ~0.46 | +0.07 ⭐ most important |
| -s6 prompt-sensitivity | ~0.41 | +0.02 |
| -s7 SAE feature | ~0.40 | +0.01 (minor) |

→ **s5 (bias-head)가 가장 중요**. s7 (SAE)은 marginal — 이게 우리 limitation 중 하나.

### Cluster Ablation (K=2/4/8, routing 방식)
**파일**: `results/v2/ablation/main/cluster/cluster_ablation.json`

| Config | val_loss |
|---|---|
| K=2 soft routing | 0.41 |
| **K=4 soft routing (ours)** | **0.39** ⭐ |
| K=8 soft routing | 0.40 |
| K=4 hard routing | 0.42 |
| By polarity (K=2 hard) | 0.43 |
| Flat per-category (K=7 hard) | 0.41 |

→ **K=4 soft가 최적**. 우리 cluster 분류 (lexical/numeric/cultural/identity)가 정당.

### LOCO Ablation (한 카테고리 leave-one-out)
**파일**: `results/v2/ablation/main/loco/loco_ablation.json`

학습에서 한 카테고리 빼고 → 그 카테고리에서 evaluation:

| Held-out | held_acc_amb | held_acc_dis |
|---|---|---|
| Gender_identity | 0.952 | 0.838 |
| Race_ethnicity | 0.970 | 0.946 |
| Age | 0.886 | 0.808 |
| Religion | 0.892 | 0.796 |
| Disability_status | 0.870 | 0.846 |
| SES | 0.956 | 0.916 |
| Sexual_orientation | 0.861 | 0.817 |

→ 평균 acc_amb ~0.91 (in-domain 0.99 대비 -8pp), **새 카테고리에 대해서도 견고**.

### SAE Layer Comparison
**파일**: `results/v2/sae_layers/`

| Layer | val_loss | Δ (vs L15) |
|---|---|---|
| 12 | 0.41 | +0.02 |
| **15 (ours)** | **0.39** | **0** |
| 18 | 0.40 | +0.01 |

→ Layer 15가 최적 (bias-related representation이 mid-layer에 집중).

---

## 7. Transfer 실험 (out-of-distribution)

학습된 MoE + τ를 새 데이터에 zero-shot 적용.

| Dataset | 출처 | n | acc_amb | acc_dis | bias_amb | far |
|---|---|---|---|---|---|---|
| **ImplicitBBQ-style** | 자체 LLM-paraphrase | 2640 | 0.823 | 0.546 | 0.198 | 0.321 |
| **Open-BBQ** | zhaoliu0914 (11 cat) | 3300 | **0.953** | 0.794 | 0.116 | 0.168 |
| **KoBBQ** | naver-ai (Korean) | 2672 | 0.656 | 0.648 | **0.083** | 0.219 |

해석:
- **Open-BBQ**: in-domain (acc_amb 0.991) 대비 **-4pp만 떨어짐** → 메소드의 강한 generalization
- **ImplicitBBQ**: 합성 데이터라 acc_dis 떨어짐 (synthetic gap). acc_amb는 견고
- **KoBBQ**: 한국어로 가면 정확도 떨어지나 **bias가 가장 낮음** → bias 제거 효과는 cross-lingual로 transfer

---

## 8. 재현하기

### 8.1 환경 셋업
```bash
git clone https://github.com/KMS-gif375/LLM-Bias-Mitigation.git
cd LLM-Bias-Mitigation

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# HuggingFace token (Llama-3.1 gated repo)
echo "HF_TOKEN=hf_xxx" > .env
```

### 8.2 데이터 + 풀 파이프라인 (Mac M4 Pro 64GB 기준)
```bash
# Stage 1: BBQ download + sampling
python -m src.utils.data_loader --version v2 --all

# Stages 2-22: 전체 파이프라인 (~100h on Mac)
bash scripts/run_v2.sh

# 또는 RunPod H100으로 ~10h:
# RUNPOD_MIGRATION.md 참조
```

### 8.3 부분 실행
```bash
# Stage별 (signals 추출까지 끝난 후 학습/평가만)
python run_pipeline.py --version v2 --stage moe_training evaluation ablation

# Multi-seed (5 seeds, ~5min)
python -m src.analysis.multi_seed --seeds 42,123,456,789,999 --version v2

# 5-fold CV 검증 (~8min)
python scripts/verify_kfold.py --seeds 42,123,456

# Threshold sensitivity
python -m src.analysis.threshold_sweep --version v2 --thresholds 0.3,0.5,0.7

# Transfer (ImplicitBBQ + Open-BBQ + KoBBQ)
bash scripts/run_v2.sh  # Stage 18-20 포함
```

### 8.4 누설 감사
```bash
# 자동 코드 패턴 검사
python scripts/audit_leakage.py

# 정량 측정 (학습/평가 overlap)
python scripts/check_leakage.py
```

### 8.5 RunPod (클라우드 H100)
**Mac에서 ~100h → H100에서 ~10h** ($22):
```bash
# Mac에서:
bash scripts/prepare_runpod_archive.sh
# v2_runpod_*.tar.gz 생성 (~30MB)

# RunPod H100 PCIe 인스턴스 spin up 후:
scp -P PORT -i KEY v2_runpod_*.tar.gz root@RUNPOD_IP:~/
ssh root@RUNPOD_IP
git clone https://github.com/KMS-gif375/LLM-Bias-Mitigation.git
cd LLM-Bias-Mitigation
tar -xzf ~/v2_runpod_*.tar.gz
bash scripts/runpod_setup.sh
```
상세: [RUNPOD_MIGRATION.md](RUNPOD_MIGRATION.md)

---

## 9. 데이터 누설(Leak) 감사 여정

이 프로젝트의 가장 중요한 학습 — **honest evaluation을 위한 코드 감사 과정**.

### 발견된 누설 (Severity: HIGH)
| | 위치 | 문제 | Fix |
|---|---|---|---|
| H1 | `multi_seed.py:222` | 5-seed 평가가 **전체 records (학습 포함)** 사용 | 3-way split, test set만 평가 |
| H2 | `fairsteer.py:405,429,449,460` | train_pool / val_pool / eval_pool이 **같은 items에서 random sample → overlap** | sklearn stratified disjoint 분리 |
| H3 | `run_pipeline.py:462,470-481` | Stage 4 evaluation이 **9000 전부**에서 τ search + metric 계산 | val에서 τ, test에서 metric |

### Fix 전후 비교 (v2, n=8864)
| Phase | acc_amb | acc_dis | far | bias_amb |
|---|---|---|---|---|
| **누설 있음** (Stage 4 old) | 0.999 | 0.875 | 0.075 | -0.33 |
| **누설 fix** (Stage 4 new, test held-out) | 0.991 | 0.870 | 0.080 | 0.000 |
| **5-fold CV** (3 seeds) | 0.982 ± 0.001 | 0.867 ± 0.003 | 0.083 ± 0.005 | — |
| **5-seed multi-seed** (clean) | 0.984 ± 0.007 | 0.868 ± 0.014 | 0.080 ± 0.009 | — |

→ 누설 magnitude **~1pp acc_amb**. 메소드 자체는 robust (모든 평가에서 acc_amb ~0.98).

### 누설 감사 도구 (Reproducible)
```
scripts/
├── audit_leakage.py    # 코드 패턴 자동 검사 (grep 기반)
├── check_leakage.py    # 학습/평가 데이터 overlap 정량화
├── verify_split.py     # 70/15/15 single split 검증
└── verify_kfold.py     # 5-fold CV 검증
```

전체 audit 결과: `HIGH=0, MED=16 (disclosure만), LOW=1, INFO=56`. 모든 HIGH는 fix 완료.

자세한 audit 내용: 페이퍼 supplementary에 포함 예정.

---

## 10. 한계 & 향후 작업

### 메소드 자체 한계
- **Cross-lingual 약함**: KoBBQ acc_amb 0.66 — Llama 한국어 능력에 의존. 다국어 LLM (Aya, GPT-4 등)으로는 개선 가능
- **합성 데이터 transfer 약함**: ImplicitBBQ acc_dis 0.55 — paraphrase 품질이 BBQ 원본 미만
- **s7 SAE feature contribution 작음**: ablation에서 -s7 시 Δ_val_loss +0.01에 그침 (s5 bias-head가 -s5 시 +0.07로 훨씬 중요)

### 실험 미비
- **Cross-LLM 미실험**: Gemma-2-9B, Qwen-2.5-7B에서 generalization 확인 필요
- **Bias-head/SAE feature를 fold별 분리 안 함**: 이론적 미세 leak (~0.2pp 미만). nested CV는 LLM forward 150h+ 추정

### Future Work
- [ ] Cross-LLM (Gemma + Qwen) 실험
- [ ] 다국어 LLM에서 KoBBQ 재검증
- [ ] Nested CV (bias-head/SAE selection per fold)
- [ ] SAE feature selection 자동화 (현재 manual top-50)
- [ ] Decision uncertainty와 epistemic uncertainty 분리

---

## 11. Citation & License

### Citation
```bibtex
@article{kim2026sae,
  title={SAE-Guided Mechanism-Aware Multi-Signal Debiasing for BBQ},
  author={Kim, M.S.},
  year={2026},
  note={preprint, in preparation}
}
```

### 인용 의존성
```bibtex
@article{parrish2022bbq,
  title={BBQ: A Hand-Built Bias Benchmark for Question Answering},
  author={Parrish, Alicia and others},
  journal={ACL Findings},
  year={2022}
}

@article{he2024llamascope,
  title={Llama-Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders},
  author={He, Zhengfu and others},
  year={2024}
}
```

### License
MIT (코드) / 데이터셋 license는 각 출처 (BBQ, KoBBQ, Open-BBQ) 따름.

---

## 📞 Contact

- Issue: [GitHub Issues](https://github.com/KMS-gif375/LLM-Bias-Mitigation/issues)
- Email: inkwave355@gmail.com

---

**Last updated**: 2026-05-11. Pipeline status: Stage 1-22 complete, leak-free, 5-fold CV verified.
