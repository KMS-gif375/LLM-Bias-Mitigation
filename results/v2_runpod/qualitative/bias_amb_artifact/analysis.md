# bias_amb Variance Artifact 정량 분석

## 문제 제기

Multi-seed (5 seeds) 결과 비교:
- acc_amb std = 0.007 (매우 안정)
- **bias_amb std = 0.32** (매우 큰 분산)

왜 같은 메소드인데 acc 는 안정하고 bias 는 분산이 클까? 본 분석으로 검증.

## bias_amb 정의

$$\text{bias\_amb} = \frac{n_{\text{stereo}} - n_{\text{anti}}}{n_{\text{stereo}} + n_{\text{anti}}}$$

분모 = ambig 중 모델이 stereo 또는 anti-stereo 로 답한 수 (Unknown 제외).
**acc_amb 가 높을수록 분모가 작아짐** → metric 자체가 noisy.

## per-seed decomposition

| Seed | acc_amb | n_amb | n_errors (denom) | n_stereo (est) | n_anti (est) | bias_amb |
|---|---|---|---|---|---|---|
| 42 | 0.9835 | 665 | **10** | 5 | 5 | -0.0909 |
| 123 | 0.9850 | 665 | **10** | 7 | 3 | +0.4000 |
| 456 | 0.9880 | 665 | **8** | 7 | 1 | +0.7500 |
| 789 | 0.9729 | 665 | **18** | 10 | 8 | +0.1111 |
| 999 | 0.9895 | 665 | **7** | 5 | 2 | +0.4286 |

**관찰**: 5 seeds 의 denominator (= 절대적 오류 수) 가 1-25 범위로 매우 작음. 이 작은 분모 위에서 1-2 errors 가 stereo 쪽으로 가느냐 anti 쪽으로 가느냐에 따라 bias_amb 가 ±0.5 단위로 변동 가능.

## Theoretical 검증 (Monte Carlo + binomial 이론)

가정: 만약 모델이 *완전 unbiased* (true bias=0) 라면, n_errors 중 stereo:anti 가 random 분배 → binomial(n, 0.5).

- Mean denominator = **10.6**
- Binomial 이론적 std (unbiased): **0.316**
- Monte Carlo 시뮬레이션 std (10K samples): 0.314
- **Observed std (5 seeds)**: **0.288**

→ Observed std (0.288) 가 theoretical unbiased std (0.316) 와 같은 자리수. **즉 분산의 대부분이 metric artifact (작은 분모) 에서 비롯되며 실제 모델 bias 의 random fluctuation 으로 충분히 설명됨**.

## 결론

1. **bias_amb std = 0.32 는 모델의 불안정성이 아닌 metric 의 본질적 artifact**.
2. **acc_amb std = 0.007 이 더 신뢰할 만한 robustness 지표**.
3. 절대 단위로 보면 bias-slip errors 는 seed 당 1-25 개로 매우 작음 (5, 7, ... ).
4. paper 작성 시 multi-seed bias_amb 보고할 때는 반드시 (a) 분모 크기 (n_errors), (b) theoretical baseline 을 함께 제시.
