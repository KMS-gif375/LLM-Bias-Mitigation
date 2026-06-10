# 논문 해석본: 실험과 결과 상세 설명

이 문서는 IEEE Access 원고 `Condition-Aware Selective Abstention for Bias Benchmark Question Answering: A Multi-Signal Audit of BBQ Condition Separability`를 한국어로 풀어쓴 해석본이다. 원고의 전체 문장을 직역하기보다, 논문이 무엇을 주장하는지, 각 실험이 왜 필요했는지, 결과 숫자가 어떤 의미인지, 그리고 투고 시 어떤 표현은 조심해야 하는지를 중심으로 설명한다.

작성 기준 원고: `paper/ieee_access/access.tex`

## 1. 논문의 핵심 요지

이 논문의 핵심은 다음 한 문장으로 요약할 수 있다.

> BBQ 원본 split에서는 ambiguous와 disambiguated 조건이 입력 표현만으로도 매우 쉽게 분리되며, 이 구조적 특성 때문에 복잡한 7-signal MoE보다 condition-aware selective abstention이 clean split에서 가장 강한 성능을 낸다.

즉, 이 논문은 "7개의 신호와 MoE가 모든 성능 향상의 주원인이다"라고 주장하지 않는다. 오히려 그 반대에 가깝다. 처음에는 evidence, counterfactual stability, confidence, self-consistency, prompt sensitivity, attention, SAE feature 같은 여러 신호를 결합하면 더 좋을 것이라고 가정했지만, 실험 결과 clean BBQ split에서는 condition prediction 자체가 거의 문제를 풀어버린다는 사실을 확인했다.

따라서 논문의 contribution은 다음 세 가지다.

1. BBQ 원본 split의 ambiguous/disambiguated boundary가 매우 잘 분리된다는 구조적 특성을 실험적으로 보였다.
2. test time에 oracle condition label을 쓰지 않고도 predicted condition만으로 강한 abstention policy를 만들 수 있음을 보였다.
3. 7-signal MoE는 clean split의 주된 성능 원인은 아니지만, condition label이 적거나 condition classifier가 불확실할 때 fallback으로 가치가 있음을 보였다.

## 2. BBQ 태스크와 평가 지표

BBQ는 bias-sensitive multiple-choice QA benchmark다. 각 예시는 크게 두 조건 중 하나에 속한다.

- Ambiguous context: 문맥만으로는 정답을 알 수 없다. 이때 올바른 답은 unknown이다.
- Disambiguated context: 문맥에 정답 근거가 있다. 이때 올바른 답은 unknown이 아니라 근거에 맞는 선택지다.

이 구조 때문에 일반적인 "accuracy만 높이면 된다"는 문제가 아니다. ambiguous에서는 unknown을 고르는 것이 맞지만, disambiguated에서 unknown을 고르면 과도한 abstention이 된다.

논문에서 주로 쓰는 지표는 세 개다.

| Metric | 의미 | 높을수록 좋은가 |
|---|---|---|
| Acc_amb | ambiguous 예시에서 unknown을 맞힌 비율 | 높을수록 좋음 |
| Acc_dis | disambiguated 예시에서 근거 기반 정답을 맞힌 비율 | 높을수록 좋음 |
| FAR | False Abstention Rate. disambiguated 예시에서 잘못 unknown을 고른 비율 | 낮을수록 좋음 |

주의할 점은 residual ambiguous bias score다. 이 값은 ambiguous에서 unknown이 아닌 답을 낸 아주 적은 residual case만 대상으로 계산된다. 이번 결과에서는 ambiguous accuracy가 거의 1에 가까워 residual denominator가 너무 작다. 그래서 bias score 숫자만 강하게 주장하면 위험하다. 논문에서는 raw count와 limitation으로 설명하는 전략을 택했다.

## 3. 방법의 구조

논문의 방법은 크게 네 가지 variant로 이해하면 쉽다.

### 3.1 Primary-answer-only

모델의 원래 답을 그대로 쓰는 기준선이다. abstention 후처리를 하지 않는다.

이 결과는 disambiguated 쪽에서 이미 강하다.

- Acc_dis = 0.8798
- FAR = 0.0717

하지만 ambiguous에서는 모델이 unknown을 충분히 고르지 못한다.

- Acc_amb = 0.5596

따라서 base model은 disambiguated utility는 이미 괜찮지만, ambiguous에서 unsupported demographic guess를 많이 낸다.

### 3.2 Condition-only retention

가장 단순하지만 가장 강한 clean-split 방법이다.

절차는 다음과 같다.

1. 이 예시가 ambiguous인지 disambiguated인지 classifier가 예측한다.
2. predicted ambiguous이면 final answer를 unknown으로 바꾼다.
3. predicted disambiguated이면 base model의 primary answer를 유지한다.

결과:

- Acc_amb = 1.0000
- Acc_dis = 0.8789
- FAR = 0.0726

이 결과는 clean BBQ split에서 사실상 최상위 operating point다. 중요한 점은 oracle condition label을 test time에 쓰지 않았다는 것이다. classifier가 condition을 예측했을 뿐이다.

### 3.3 Seven-signal MoE predicted condition

7개의 신호를 사용해 answer-retention score를 예측하는 MoE variant다.

사용한 신호:

1. s1 evidence: 답을 지지하는 context evidence가 있는지
2. s2 counterfactual: demographic group swap 이후 답이 안정적인지
3. s3 confidence: 선택지 log-probability confidence
4. s4 consistency: repeated stochastic sampling에서 답이 일관적인지
5. s5 bias head: demographic token에 대한 bias-relevant attention head mass
6. s6 prompt sensitivity: vanilla, debiasing, CoT, counterfactual prompt 간 답이 안정적인지
7. s7 SAE feature: Llama-Scope SAE feature activation

결과:

- Acc_amb = 0.9946
- Acc_dis = 0.8732
- FAR = 0.0843

강한 결과이긴 하지만 condition-only보다 약간 낮다. 따라서 clean split에서는 "7-signal MoE가 성능 향상의 주원인"이라고 주장하면 안 된다. 논문의 정확한 해석은 "7-signal MoE를 감사했지만 clean split에서는 condition-only가 충분했다"이다.

### 3.4 Hybrid fallback

condition classifier가 확신할 때는 condition-only rule을 쓰고, classifier가 불확실할 때만 MoE retention score로 fallback하는 방식이다.

full label setting에서는 condition-only와 거의 비슷하다.

- Acc_amb = 0.9979
- Acc_dis = 0.8795
- FAR = 0.0723

하지만 low-label setting, 특히 condition label이 1-5%밖에 없을 때는 확실히 도움이 된다.

## 4. Main clean BBQ comparison

가장 중요한 main table의 의미는 다음과 같다.

| System | Acc_amb | Acc_dis | FAR | 해석 |
|---|---:|---:|---:|---|
| Composite | 0.6843 | 0.2855 | 0.2449 | prompt-only baseline. disambiguated utility가 낮다. |
| DeCAP | 0.8057 | 0.7238 | 0.2419 | Composite보다 강하지만 FAR가 높다. |
| SDR | 0.9584 | 0.1928 | 0.7858 | ambiguous는 잘 abstain하지만 disambiguated에서 과도하게 unknown을 낸다. |
| Primary answer only | 0.5596 | 0.8798 | 0.0717 | disambiguated는 좋지만 ambiguous가 약하다. |
| Condition-only retention | 1.0000 | 0.8789 | 0.0726 | clean split 최강 deployable audit. |
| Hybrid fallback | 0.9979 | 0.8795 | 0.0723 | full label에서는 condition-only와 거의 동일. |
| MoE single threshold | 0.9494 | 0.8413 | 0.1325 | condition split 없이 하나의 threshold만 쓰면 약해진다. |
| MoE predicted condition | 0.9946 | 0.8732 | 0.0843 | 강하지만 condition-only보다 clean split에서 약간 낮다. |
| MoE oracle condition | 0.9946 | 0.8738 | 0.0837 | predicted와 거의 동일. condition prediction이 매우 쉬움을 의미한다. |

핵심 해석:

- condition-only가 clean split에서 가장 좋은 이유는 BBQ의 condition boundary가 매우 잘 분리되기 때문이다.
- primary-answer-only가 disambiguated에서 이미 강하므로, 이 논문의 clean-split gain은 주로 ambiguous에서 unsupported answer를 unknown으로 바꾸는 데서 온다.
- SDR은 ambiguous abstention은 높지만 disambiguated utility가 크게 무너진다. 이것이 over-abstention의 전형적인 사례다.
- single-threshold MoE가 condition-aware MoE보다 약한 것은, BBQ에서는 ambiguous와 disambiguated에 같은 threshold를 쓰면 안 된다는 것을 보여준다.

## 5. Confidence-only baseline

이 실험은 "그냥 confidence 낮으면 unknown으로 바꾸면 되는 것 아닌가?"라는 질문에 답한다.

비교한 방법:

- Max-softmax threshold
- Unknown probability threshold
- Temperature-scaled unknown probability threshold
- Condition-only retention

결과:

| System | Acc_amb | Acc_dis | FAR |
|---|---:|---:|---:|
| Max-softmax | 0.8599 | 0.7361 | 0.2530 |
| Unknown prob. | 0.7952 | 0.7587 | 0.2166 |
| Temp.-scaled unk. | 0.8057 | 0.7569 | 0.2190 |
| Condition-only retention | 1.0000 | 0.8789 | 0.0726 |

해석:

- 단순 confidence threshold만으로는 condition-only를 이기지 못한다.
- 이는 clean-split gain이 단순 uncertainty cutoff가 아니라 ambiguous/disambiguated condition separation에서 온다는 증거다.

## 6. Low-label hybrid fallback

이 실험은 7-signal MoE의 가치가 어디에 있는지를 보여준다.

clean split에서 condition label이 충분하면 condition-only가 거의 완벽하다. 하지만 condition label이 아주 적으면 condition classifier가 약해진다. 이때 7-signal MoE fallback이 도움이 되는지 본다.

| Label fraction | System | Acc_amb | Acc_dis | FAR |
|---|---|---:|---:|---:|
| 1% | Condition-only | 0.9136 | 0.6786 | 0.2931 |
| 1% | Hybrid fallback | 0.9530 | 0.8247 | 0.1452 |
| 5% | Condition-only | 0.9645 | 0.8301 | 0.1280 |
| 5% | Hybrid fallback | 0.9744 | 0.8548 | 0.1048 |
| 10% | Condition-only | 0.9852 | 0.8587 | 0.0958 |
| 10% | Hybrid fallback | 0.9834 | 0.8726 | 0.0831 |
| 100% | Condition-only | 1.0000 | 0.8789 | 0.0726 |
| 100% | Hybrid fallback | 0.9979 | 0.8795 | 0.0723 |

가장 중요한 결과:

- 1% label에서 Acc_dis가 0.6786에서 0.8247로 크게 상승한다.
- FAR도 0.2931에서 0.1452로 크게 낮아진다.
- 5% label에서도 hybrid가 확실히 좋다.
- 10%와 100%에서는 차이가 작아진다.

해석:

- 7-signal MoE는 clean split의 main driver가 아니다.
- 하지만 condition supervision이 부족하거나 condition classifier가 불확실할 때 fallback으로 실용적 가치가 있다.
- 이것이 논문에서 7-signal/MoE를 살리는 가장 안전한 주장이다.

## 7. Generalization experiments

### 7.1 LOCO

LOCO는 Leave-One-Category-Out이다. 예를 들어 Age category를 통째로 빼고 학습한 뒤 Age에 test하는 방식이다. 이 실험은 category-specific pattern을 외운 것이 아닌지 확인하기 위해 필요하다.

결과:

| Experiment | Acc_amb | Acc_dis | FAR |
|---|---:|---:|---:|
| LOCO, predicted condition | 0.9214 | 0.8331 | 0.1161 |
| LOCO, single threshold | 0.8362 | 0.8013 | 0.1536 |

해석:

- clean split보다는 성능이 내려간다.
- 그래도 predicted-condition variant는 꽤 강하다.
- condition-specific threshold가 없는 single-threshold보다 좋다.
- 이 결과는 "category만 외운 것 아니냐"는 reviewer 공격을 줄여준다.

### 7.2 Open-BBQ transfer

Open-BBQ는 외부 BBQ-style benchmark로, 원본 BBQ split에만 과적합한 것이 아닌지 확인하는 실험이다.

결과:

- Acc_amb = 0.9915
- Acc_dis = 0.8358
- FAR = 0.1012

해석:

- ambiguous abstention은 매우 강하게 유지된다.
- disambiguated utility는 clean split보다 낮아지지만 여전히 의미 있는 수준이다.
- BBQ 원본 split에만 맞춘 결과가 아니라 외부 BBQ-style data에서도 어느 정도 작동한다는 근거가 된다.

### 7.3 Cross-LLM robustness

Llama-3.1-8B 외에 Qwen과 Mistral에서도 같은 경향이 나오는지 본다.

| Backbone | Acc_amb | Acc_dis | FAR |
|---|---:|---:|---:|
| Qwen-2.5-7B | 0.9895 | 0.8147 | 0.1672 |
| Mistral-7B-v0.3 | 0.9940 | 0.7798 | 0.1916 |

해석:

- ambiguous abstention은 다른 backbone에서도 강하다.
- 하지만 FAR가 Llama보다 높다.
- 따라서 "모든 모델에서 동일하게 최강"이라고 말하면 안 된다.
- 안전한 표현은 "qualitatively similar pattern, but with higher FAR on Qwen/Mistral"이다.
- Qwen/Mistral에서는 Llama-Scope SAE feature set이 호환되지 않아 s7을 zero/default로 둔 점도 limitation으로 설명해야 한다.

## 8. Risk-coverage and AURC

이 실험은 final accuracy가 아니라 retention score의 ranking quality를 본다.

즉, 어떤 score가 "어떤 primary answer를 유지하고 어떤 answer를 버려야 하는지" 잘 정렬하는지 확인한다.

| Ranking score | AURC | E-AURC |
|---|---:|---:|
| Chosen softmax | 0.1197 | 0.0760 |
| Unknown-prob. inverse | 0.2011 | 0.1574 |
| Condition probability | 0.1467 | 0.1030 |
| MoE retention | 0.0626 | 0.0189 |

해석:

- AURC와 E-AURC는 낮을수록 좋다.
- MoE retention score가 가장 낮다.
- 즉, clean split operating point에서는 condition-only가 가장 강하지만, MoE score 자체는 retention ranking signal로는 유용하다.
- 이것은 MoE를 완전히 무의미한 것으로 버리지 않고 "fallback/ranking audit에 유용"하다고 주장할 수 있는 근거다.

## 9. Low-threshold plateau

이 실험은 `tau_dis = 0.05`가 grid의 하한이라서 우연히 선택된 것이 아닌지 확인하기 위해 필요하다.

| tau_dis | Acc_amb | Acc_dis | FAR |
|---:|---:|---:|---:|
| 0.00 | 0.9946 | 0.8798 | 0.0717 |
| 0.01 | 0.9946 | 0.8789 | 0.0744 |
| 0.02 | 0.9946 | 0.8780 | 0.0771 |
| 0.03 | 0.9946 | 0.8768 | 0.0792 |
| 0.05 | 0.9946 | 0.8738 | 0.0837 |
| 0.10 | 0.9946 | 0.8696 | 0.0916 |

해석:

- tau_dis가 0.00에서 0.05 사이일 때 Acc_amb는 동일하다.
- Acc_dis와 FAR도 완만하게 변한다.
- 따라서 특정 threshold 하나에 fragile하게 의존한다고 보기 어렵다.
- 이 실험은 "validation grid 하한에서 optimum이 나왔으니 grid가 부족한 것 아닌가?"라는 의심을 방어한다.

## 10. Condition classifier ablation

이 논문에서 가장 중요한 audit 중 하나다. 이 실험은 condition classifier가 실제로 무엇을 보고 ambiguous/disambiguated를 맞히는지 확인한다.

| Feature set | Test accuracy |
|---|---:|
| Signals + embedding + category + primary | 0.9983 |
| Minus primary answer | 0.9982 |
| Signals + raw-text embedding | 0.9980 |
| Raw-text embedding only | 0.9961 |
| Signals only | 0.8334 |
| Primary answer only | 0.5440 |
| Category only | 0.5000 |

해석:

- primary answer를 빼도 성능이 거의 그대로다.
- category만 쓰면 random 수준이다.
- raw-text embedding만 써도 0.9961이다.
- 따라서 answer feature leakage가 핵심 원인이 아니다.
- BBQ의 ambiguous/disambiguated boundary 자체가 input representation에 강하게 들어있다는 뜻이다.

이 결과가 논문의 제목과 핵심 claim을 뒷받침한다.

## 11. LOCO condition-prediction audit

LOCO에서는 clean split보다 condition prediction이 어려워진다.

| Feature set | Held-out accuracy |
|---|---:|
| Signals + raw-text emb. + primary | 0.9261 |
| Signals + raw-text emb. | 0.9259 |
| Raw-text emb. only | 0.8909 |
| Signals only | 0.8125 |

해석:

- clean split의 0.9983보다는 낮다.
- held-out category에서는 condition boundary가 덜 쉽게 분리된다.
- 그래도 0.9261은 상당히 강하다.
- 이 결과는 clean split이 너무 쉬운 특성이 있음을 인정하면서도, 완전히 category memorization만은 아님을 보여준다.

## 12. KoBBQ condition-transfer audit

KoBBQ는 한국어 BBQ-style dataset이다. 이 실험은 cross-lingual transfer를 본다.

| Scope | Full features | Embedding only | Signals only |
|---|---:|---:|---:|
| English BBQ -> KoBBQ | 0.5000 | 0.5000 | 0.6534 |
| Within KoBBQ | 0.9995 | 0.9990 | 0.6847 |

해석:

- English BBQ에서 학습한 classifier를 KoBBQ에 바로 적용하면 random 수준이다.
- 하지만 KoBBQ 안에서 train/test하면 0.9995로 매우 잘 된다.
- 따라서 KoBBQ에도 condition boundary는 존재하지만, English embedding classifier가 Korean representation으로 직접 transfer되지 않는 것이다.
- 이 결과는 "방법이 모든 언어에 바로 일반화된다"는 주장을 막아준다.
- 동시에 "condition separability는 benchmark/language representation property"라는 논문의 핵심 해석을 강화한다.

## 13. Learned rejector audit

이 실험은 "MoE 대신 단순 logistic rejector를 학습하면 되지 않나?"라는 질문에 답한다.

| Rejector | Acc_amb | Acc_dis | FAR |
|---|---:|---:|---:|
| Signals only | 0.8711 | 0.7407 | 0.2437 |
| Raw-text emb. only | 0.9027 | 0.7913 | 0.1750 |
| Signals + raw-text emb. | 0.9295 | 0.7729 | 0.2093 |
| Condition-only retention | 1.0000 | 0.8789 | 0.0726 |

해석:

- generic learned rejector는 condition-only보다 약하다.
- 즉, clean split 성능은 단순히 "retention/rejection classifier를 하나 더 학습했기 때문"이 아니다.
- condition-aware abstention이 핵심이다.

## 14. Residual ambiguous-bias count

Residual ambiguous bias score는 불안정하므로 raw count로 봐야 한다.

| Seed | Unknown correct | Residual non-unknown | Stereo | Anti |
|---:|---:|---:|---:|---:|
| 42 | 663/664 | 1 | 0 | 1 |
| 123 | 664/664 | 0 | 0 | 0 |
| 456 | 661/664 | 3 | 0 | 3 |
| 789 | 659/664 | 5 | 5 | 0 |
| 999 | 655/664 | 9 | 3 | 6 |

해석:

- ambiguous에서 대부분 unknown을 맞힌다.
- residual non-unknown case가 seed당 0-9개뿐이다.
- 이 정도 denominator에서는 bias score가 크게 출렁인다.
- 그래서 논문은 "bias score도 우리가 최고"라고 주장하지 않는다.
- 안전한 주장은 "ambiguous residual cases are rare; residual bias score is numerically unstable"이다.

## 15. Signal masking audit

각 신호 하나를 mask했을 때 성능 변화가 얼마나 생기는지 보는 실험이다.

표의 delta는 full-model metric minus masked-model metric이다.

| Signal | Delta Acc_amb | Delta Acc_dis | Delta FAR | 해석 |
|---|---:|---:|---:|---|
| s1 evidence | -0.0015 | -0.0018 | +0.0036 | 영향 작음 |
| s2 counterfactual | -0.0012 | +0.0015 | +0.0012 | 영향 작음 |
| s3 confidence | -0.0012 | 0.0000 | +0.0024 | 영향 작음 |
| s4 consistency | -0.0003 | -0.0015 | +0.0033 | 영향 작음 |
| s5 bias head | -0.0012 | -0.0009 | +0.0009 | 영향 작음 |
| s6 prompt | +0.0009 | -0.0039 | +0.0087 | 상대적으로 영향이 크지만 여전히 작음 |
| s7 SAE | -0.0015 | -0.0006 | +0.0015 | SAE 단독 claim은 약함 |

해석:

- 개별 신호 하나를 제거해도 영향이 작다.
- s7 SAE feature가 핵심 원인이라고 주장할 수 없다.
- 신호들이 redundant하거나, clean split에서는 condition prediction이 대부분을 설명한다고 보는 것이 안전하다.

## 16. MoE signal-subset audit

신호를 전부 제거하거나 일부만 남겼을 때의 성능을 본다.

| MoE variant | Acc_amb | Acc_dis | FAR |
|---|---:|---:|---:|
| All signals zeroed | 1.0000 | 0.8789 | 0.0726 |
| s3 confidence only | 0.9985 | 0.8789 | 0.0726 |
| s1, s3, s4, s6 core | 0.9988 | 0.8789 | 0.0726 |
| Full seven-signal MoE | 0.9946 | 0.8732 | 0.0843 |

해석:

- all-zero가 condition-only와 동일하다.
- full seven-signal MoE가 clean split에서는 오히려 약간 낮다.
- 따라서 clean split에서 "7-signal이 필요하다"는 주장은 데이터와 맞지 않는다.
- 더 정확한 주장은 "signals are useful as low-label fallback and ranking audit, but not necessary on the clean main split"이다.

## 17. Rule-based explanation audit

이 논문은 단순히 답을 바꾸는 것에 그치지 않고, 왜 바꿨는지 rule-based explanation artifact를 남긴다.

5 seeds 전체 6,640 seed-level decisions에서의 결과:

| Decision label | Count | Share |
|---|---:|---:|
| Utility-preserving keep | 2899 | 0.4366 |
| Ambiguous abstention | 1858 | 0.2798 |
| Stereotyped raw answer blocked | 995 | 0.1498 |
| Anti-stereotyped unsupported answer blocked | 449 | 0.0676 |
| False abstention | 280 | 0.0422 |
| Wrong stereotyped keep | 76 | 0.0114 |
| Wrong anti-stereotyped keep | 65 | 0.0098 |
| Anti-stereotype slip | 10 | 0.0015 |
| Stereotype bias slip | 8 | 0.0012 |

해석:

- 많은 경우는 utility-preserving keep 또는 ambiguous abstention이다.
- stereotyped raw answer를 unknown으로 막은 사례가 995개 있다.
- residual stereotype slip은 8개로 매우 적다.
- 이 explanation은 causal mechanistic explanation이 아니다.
- 대신 decision behavior를 audit하기 쉽게 만드는 deterministic label이다.

즉, 이 부분은 "왜 편향인지 설명해주는 실용적 audit layer"로 어필할 수 있다. 다만 "모델 내부 인과 메커니즘을 완전히 설명한다"고 쓰면 안 된다.

## 18. Per-category coverage audit

category별로 over-abstention이 특정 그룹에 몰리는지 확인한다.

| Category | Coverage | Disambig. coverage | FAR |
|---|---:|---:|---:|
| Age | 0.4787 | 0.9573 | 0.0427 |
| Disability status | 0.4613 | 0.9227 | 0.0773 |
| Gender identity | 0.4667 | 0.9333 | 0.0667 |
| Nationality | 0.4867 | 0.9733 | 0.0267 |
| Physical appearance | 0.4107 | 0.8213 | 0.1787 |
| Race ethnicity | 0.4880 | 0.9760 | 0.0240 |
| Religion | 0.4387 | 0.8773 | 0.1227 |
| SES | 0.4853 | 0.9707 | 0.0293 |
| Sexual orientation | 0.4562 | 0.9125 | 0.0875 |

해석:

- Physical appearance와 Religion에서 FAR가 높다.
- Race, Nationality, SES는 FAR가 낮다.
- category-aware threshold로 보정할 수도 있지만, group-specific deployment policy가 되기 때문에 main method에는 넣지 않았다.
- 이 표는 aggregate FAR만 보지 않고 category-level over-abstention을 투명하게 보여주는 역할을 한다.

## 19. Historical robustness checks

최종 clean experiment 이전 결과들도 appendix에 정직하게 남겼다.

이전 8,864 saved-signal run:

- Acc_amb = 0.9977
- Acc_dis = 0.8736
- FAR = 0.0832

single full run:

- Acc_amb = 0.9993
- Acc_dis = 0.8748
- FAR = 0.0754

older Open-BBQ protocol:

- Acc_amb = 0.9527
- Acc_dis = 0.7939
- FAR = 0.1685

Harder transfer:

- ImplicitBBQ: Acc_amb = 0.8227, Acc_dis = 0.5464, FAR = 0.3208
- KoBBQ end-to-end: Acc_amb = 0.6557, Acc_dis = 0.6475, FAR = 0.2186

해석:

- 최종 protocol이 가장 깔끔하지만, 이전 결과와 harder transfer 결과도 숨기지 않았다.
- 이는 reviewer에게 신뢰를 주는 요소다.
- 동시에 limitation도 분명히 보여준다. 특히 KoBBQ와 ImplicitBBQ에서는 성능이 크게 낮아진다.

## 20. 논문에서 안전하게 주장할 수 있는 것

다음 주장은 안전하다.

1. BBQ clean split은 ambiguous/disambiguated condition boundary가 매우 잘 분리된다.
2. predicted condition만으로도 oracle condition과 거의 같은 성능을 얻는다.
3. condition-only retention은 clean BBQ split에서 ambiguous accuracy와 disambiguated utility 사이의 trade-off를 매우 잘 관리한다.
4. 7-signal MoE는 clean split main driver는 아니지만 low-label/uncertain-condition setting에서 fallback으로 유용하다.
5. MoE retention score는 risk-coverage ranking signal로 raw confidence보다 좋다.
6. rule-based explanation artifact는 runtime decision과 benchmark audit label을 분리하여 decision behavior를 더 투명하게 만든다.
7. LOCO, Open-BBQ, cross-LLM 결과는 원본 split에만 완전히 갇힌 결과는 아님을 보여준다.
8. KoBBQ와 harder transfer 결과는 generalization limitation을 보여준다.

## 21. 논문에서 피해야 하는 주장

다음 표현은 위험하다.

1. "7-signal MoE가 clean split 성능 향상의 주원인이다."
2. "SAE feature s7이 bias mitigation의 causal mechanism이다."
3. "우리 방법은 모든 bias benchmark에 일반적으로 통한다."
4. "ambiguous residual bias score도 우리가 가장 좋다."
5. "FairSteer 비교로 우리가 우월함을 증명했다."
6. "Self-debiasing 원논문을 완전히 공식 재현했다."
7. "Qwen/Mistral에서도 Llama와 똑같이 안정적이다."

더 안전한 표현은 다음과 같다.

- "We audit, rather than causally explain, the role of multi-signal retention."
- "The clean split is highly condition-separable."
- "The seven-signal layer is best understood as a low-label fallback."
- "Residual ambiguous-bias scores are numerically unstable due to a very small residual denominator."
- "Cross-LLM results show a similar qualitative pattern but higher FAR."

## 22. 한눈에 보는 최종 결론

이 논문은 "복잡한 debiasing mechanism을 새로 만들어서 bias를 완전히 해결했다"는 논문이 아니다. 오히려 더 정직한 audit paper다.

핵심 발견은 다음과 같다.

- BBQ 원본 split은 condition prediction만으로도 매우 잘 풀린다.
- 이 때문에 condition-aware selective abstention이 clean split에서 가장 강하다.
- 7-signal MoE는 clean split에서는 필요하지 않지만, label이 적거나 classifier가 불확실할 때 도움이 된다.
- rule-based explanation은 결과를 감사하고 실패 유형을 분해하는 데 유용하다.
- KoBBQ, ImplicitBBQ, cross-LLM 결과는 방법의 한계와 transfer 리스크를 솔직하게 보여준다.

따라서 IEEE Access 투고에서 가장 좋은 framing은 다음이다.

> This paper is a reproducible audit of BBQ condition separability and no-oracle selective abstention. It shows that condition-aware abstention, not a uniquely necessary multi-signal mechanism, explains the clean-split gain, while the multi-signal layer remains useful as a low-label fallback and audit tool.

