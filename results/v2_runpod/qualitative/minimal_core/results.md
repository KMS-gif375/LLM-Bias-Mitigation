# Minimal-Core Signal Ablation (Q2)

MoE 를 다음 신호 subset 으로 학습 → minimal core 검증.

## Results (5 seeds 평균)

| Variant | n signals | val_loss | test_loss |
|---|---|---|---|
| full_7 ⭐ | 7 | 0.4095 ± 0.0228 | 0.3835 ± 0.0415 |
| core_4_s1346 | 4 | 0.4097 ± 0.0208 | 0.3779 ± 0.0269 |
| core5_plus_s5 | 5 | 0.4097 ± 0.0218 | 0.3785 ± 0.0266 |
| core5_plus_s7 | 5 | 0.4116 ± 0.0205 | 0.3812 ± 0.0312 |
| core5_plus_s2 | 5 | 0.4083 ± 0.0224 | 0.3799 ± 0.0336 |
| core6_plus_s5_s7 | 6 | 0.4123 ± 0.0202 | 0.3806 ± 0.0313 |

## 해석

- **core_4 (s1+s3+s4+s6) val_loss 0.4097 ≈ full_7 0.4095** → s2/s5/s7 정말 redundant.
- +s5/+s7 추가 시 향상 폭이 작으면 → 해당 신호 marginal.
- minimal core 의 경우 paper 에서 '4-signal core MoE' 로 streamlined version 제시 가능.
