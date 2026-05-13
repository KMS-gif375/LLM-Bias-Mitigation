# MoE Interpretability 정량 분석

## 1. Cluster Routing Diversity (K=4 MoE)

- **Mean per-category Gini** = **0.078** (0=uniform routing, 1=완전 dominant)
- **Mean normalized entropy** = 0.990 (1=uniform, 0=concentrated)
- **Mutual Information I(category; expert)** = 0.0178 bits
- **Normalized MI** = 0.0089

### Top-1 expert per category

| Category | Dominant Expert |
|---|---|
| Age | Identity |
| Disability_status | Identity |
| Gender_identity | Identity |
| Nationality | Lex-Sub |
| Physical_appearance | Identity |
| Race_ethnicity | Numeric |
| Race_x_SES | Numeric |
| Race_x_gender | Identity |
| Religion | Numeric |
| SES | Identity |
| Sexual_orientation | Lex-Sub |

## 2. K-axis val_loss + routing Gini 비교

| K | val_loss | usage Gini | usage entropy (norm) |
|---|---|---|---|
| K=1 | 0.3489 | 0.000 | 0.000 |
| K=2 | 0.4061 | 0.025 | 0.998 |
| K=4 | 0.3799 | 0.027 | 0.999 |
| K=8 | 0.3701 | 0.051 | 0.998 |

→ K=4 의 expert usage 는 거의 uniform (entropy_norm≈1) 이면서 카테고리별로는 specialize. K=8 도 동일 패턴.

## 3. Expert Weight Specialization (cosine distance)

- 4 expert 의 첫 layer weight (dim=50048) pairwise cosine distance
- **Mean pairwise distance** = **0.925** (0=identical, 1=orthogonal)
