# Confidence-Aware Multi-Signal Debiasing

BBQ 계열 질의응답에서 사회적 편향을 줄이기 위한 연구 코드입니다. 모델 가중치는 고정하고, 여러 신뢰도/편향 신호를 추출한 뒤 문맥 조건에 따라 기권 여부를 조정합니다. 목표는 모호한 문맥에서는 unknown 답변을 안정적으로 유지하고, 명시 문맥에서는 불필요한 기권을 줄여 유용성을 살리는 것입니다.

이 README는 제출 전 점검, 재현, 논문 그림 재사용에 필요한 내용만 간결하게 정리했습니다.

## 현재 상태

최신 reviewer-defense 패키지 기준일: **2026-05-26**

논문에서 안전하게 밀 수 있는 핵심 주장:

> 제안 방법은 테스트 시 oracle condition label 없이도 ambiguous abstention accuracy를 높게 유지하면서 disambiguated utility를 개선하고 false abstention을 줄인다.

과하게 쓰면 위험한 주장:

- ambiguous residual bias score가 항상 최고라고 주장하지 않습니다. ambiguous accuracy가 거의 만점이면 residual non-unknown 표본이 너무 적어서 `abs_bias_amb`가 흔들립니다.
- SAE feature `s7`이 성능의 주된 원인이라고 주장하지 않습니다. `s7`은 포함되고 audit되었지만, 단독 ablation 효과는 작습니다.
- FairSteer를 본문 핵심 baseline처럼 세우지 않습니다. matched-ID overlap이 작아서 appendix의 보조 비교로 두는 것이 안전합니다.

## 핵심 결과

### Main Clean BBQ

Llama-3.1-8B, clean acceptance package, 5 seeds, `n_test=1,328`.

| 변형 | acc_amb | acc_dis | FAR | 해석 |
|---|---:|---:|---:|---|
| predicted-condition | **0.9946 ± 0.0054** | **0.8732 ± 0.0108** | **0.0843 ± 0.0193** | oracle 없이 쓰는 main claim |
| oracle per-condition | 0.9946 ± 0.0054 | 0.8738 ± 0.0109 | 0.0837 ± 0.0194 | 상한선 비교 |
| single-threshold | 0.9494 ± 0.0126 | 0.8413 ± 0.0184 | 0.1325 ± 0.0240 | 단순 배포형 fallback |

### Reviewer-Defense 실험

| 실험 | 설정 | 결과 | 방어 포인트 |
|---|---|---|---|
| Clean LOCO | 9개 held-out category × 5 seeds | acc_amb **0.9214 ± 0.0421**, acc_dis **0.8331 ± 0.0793**, FAR **0.1161 ± 0.0551** | category memorization 공격 방어 |
| Open-BBQ fresh transfer | 11 categories, `n=3,300` | acc_amb **0.9915**, acc_dis **0.8358**, FAR **0.1012** | original BBQ split overfit 공격 방어 |
| Cross-LLM | Qwen + Mistral, 각 5 seeds | Qwen **0.9895/0.8147/FAR 0.1672**; Mistral **0.9940/0.7798/FAR 0.1916** | Llama 전용 튜닝이 아니라는 근거 |
| Threshold repetition | Llama/Qwen/Mistral × 15 runs | `tau_dis = 0.05`, std **0.000** | 반복 실험에서 같은 grid-boundary 패턴 확인 |
| SAE/s7 audit | Open-BBQ signal extraction | `s7_bias_sae_feature_count=56` | `s7` 신호 경로가 실제로 활성화됨 |

## 논문용 Figure

논문에는 `results/figures/`의 PDF를 쓰는 것을 권장합니다. README 미리보기용 PNG는 `docs/figures/`에 같은 이름으로 저장됩니다.

### Figure 1. 전체 파이프라인

![전체 파이프라인](docs/figures/fig1_pipeline.png)

### Figure 3. MoE 집계기 구조

![MoE 집계기 구조](docs/figures/fig3_moe_architecture.png)

### Figure 4. BBQ 주요 비교

본문 그림은 `acc_amb`, `acc_dis`, FAR만 전면에 둡니다. ambiguous residual bias는 표본 수가 작아 raw count/CI를 appendix 표로 보고하는 쪽이 안전합니다.

![BBQ 주요 비교](docs/figures/fig4_main_results.png)

### Figure 5. 카테고리별 게이트 가중치

![카테고리별 게이트 가중치](docs/figures/fig5_cluster_routing.png)

### 추가 진단 Figure

![위험-커버리지 곡선](docs/figures/risk_coverage_curve.png)

![편향 관련 attention head heatmap](docs/figures/bias_heads_heatmap.png)

## 방법 요약

파이프라인은 네 단계입니다.

1. 네 가지 prompt 변형을 실행합니다: vanilla, debiasing prompt, chain-of-thought, counterfactual swap.
2. 일곱 개의 신뢰도/편향 신호를 추출합니다.
   - `s1`: logit confidence
   - `s2`: multi-prompt consistency
   - `s3`: counterfactual stability
   - `s4`: evidence-quote consistency
   - `s5`: self-consistency
   - `s6`: bias-head attention
   - `s7`: SAE bias-feature activation
3. question embedding으로 condition된 4-expert MoE가 신호를 집계합니다.
4. threshold override를 적용합니다. confidence가 낮으면 unknown 답변으로 바꿉니다.

현재 canonical grid에서 반복적으로 관찰된 패턴은 아래와 같습니다.

| 모델 | Seeds | `tau_dis` |
|---|---:|---:|
| Llama-3.1-8B | 5 | 0.05 ± 0.000 |
| Qwen-2.5-7B | 5 | 0.05 ± 0.000 |
| Mistral-7B-v0.3 | 5 | 0.05 ± 0.000 |

이 값은 현재 grid에서 낮은 threshold 경계가 포화된 패턴으로 해석해야 합니다. `0.05`가 연속 공간의 진짜 최적값이라고 과장하지 않습니다.

## 저장소 구조

| 경로 | 역할 |
|---|---|
| `run_pipeline.py` | BBQ main pipeline entry point |
| `src/signals/` | 신호 추출 |
| `src/models/` | MoE aggregator와 threshold override |
| `src/transfer/` | Open-BBQ / KoBBQ / transfer 실험 |
| `src/analysis/` | multi-seed, ablation, qualitative, plotting utility |
| `src/paper/figures.py` | 논문용 figure 생성기 |
| `scripts/run_clean_experiments.py` | clean main-suite runner |
| `scripts/run_loco_clean.py` | clean leave-one-category-out runner |
| `scripts/run_acceptance_package.py` | reviewer-defense package runner |
| `scripts/build_acceptance_report.py` | appendix/report table builder |
| `docs/figures/` | README용 PNG 미리보기 |
| `results/figures/` | 논문용 PDF/PNG figure |

큰 prediction 파일과 run output은 대부분 local artifact입니다. `results/` 아래의 모든 파일을 커밋 대상으로 보지 않습니다.

## 재현 방법

### 환경 준비

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# gated Llama weight 접근에 필요
echo "HF_TOKEN=hf_..." > .env
```

권장 하드웨어:

| 작업 | 권장 사양 |
|---|---|
| Llama-3.1-8B inference | CUDA GPU 16GB+ 또는 Apple Silicon 64GB |
| SAE feature extraction | CUDA GPU 권장 |
| Clean LOCO / transfer package | H100 권장 |
| MoE training / report building | CPU로 충분 |

### Main BBQ Pipeline

```bash
python run_pipeline.py --version v2 --model main --stage all
```

### Clean Main Suite

```bash
python scripts/run_clean_experiments.py \
  --seeds 42 123 456 789 999 \
  --out-dir results/v2/clean_experiments \
  --run-signal-ablation
```

### Reviewer-Defense Package

한 번에 실행:

```bash
python scripts/run_acceptance_package.py
```

핵심 실험만 분리해서 실행:

```bash
# Leave-one-category-out
python scripts/run_loco_clean.py \
  --seeds 42 123 456 789 999 \
  --out-dir results/v2/acceptance_package/loco

# Open-BBQ fresh transfer
# --max-samples 300은 11개 category × 300개 = 총 n=3,300을 의미
python -m src.transfer.run_open_bbq \
  --max-samples 300 \
  --out-dir results/v2/acceptance_package/open_bbq \
  --force --model main

# 기존 signal 기반 cross-LLM 5-seed summary
python -m src.analysis.multi_seed --version v2 --model qwen \
  --seeds 42,123,456,789,999 \
  --out-dir results/v2/cross_llm/qwen/multi_seed_5seed

python -m src.analysis.multi_seed --version v2 --model mistral \
  --seeds 42,123,456,789,999 \
  --out-dir results/v2/cross_llm/mistral/multi_seed_5seed

# 논문/appendix 표 생성
python scripts/build_acceptance_report.py
```

### Figure 재생성

```bash
# 논문용 main figures
python -m src.paper.figures --figs 1 3 4 5 --out-dir results/figures

# README용 PNG/PDF copies
python -m src.paper.figures --figs 1 3 4 5 --out-dir docs/figures

# 진단 figures
python -m src.analysis.qualitative \
  --tasks bias_heads_heatmap risk_coverage \
  --out-dir results/figures

python -m src.analysis.qualitative \
  --tasks bias_heads_heatmap risk_coverage \
  --out-dir docs/figures
```

## 논문 작성 메모

써도 안전한 문장:

- 제안 방법은 ambiguous abstention accuracy를 유지하면서 disambiguated utility를 개선한다.
- predicted-condition 결과가 oracle 없이 배포 가능한 main setting이다.
- LOCO와 Open-BBQ transfer는 BBQ category pattern memorization 가능성을 낮춘다.
- Cross-LLM 결과는 Llama 하나에만 맞춘 방법이 아니라는 근거를 제공한다.

피해야 할 문장:

- "We achieve the lowest ambiguous bias score."
- "s7 is the reason the method works."
- "FairSteer proves superiority as a full baseline."
- "0.05 is the true continuous optimum."

## License와 Data

데이터셋과 모델은 각 원 라이선스를 따릅니다.

- BBQ: NYU MLL, CC-BY-4.0
- Open-BBQ: CC-BY-4.0
- KoBBQ: CC-BY-SA-4.0
- Winogender: Rudinger et al., NAACL 2018
- Llama-3.1-8B: Meta Llama license
- Qwen-2.5-7B: Apache 2.0
- Mistral-7B-v0.3: Apache 2.0

## Citation

```bibtex
@misc{confidence_aware_bias_mitigation_2026,
  title = {Confidence-Aware Multi-Signal Debiasing with Condition-Aware Abstention},
  author = {KMS},
  year = {2026},
  note = {Research artifact}
}
```
