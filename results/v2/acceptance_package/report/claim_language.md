# Claim Language

Strong claim:

> The proposed method preserves high ambiguous-context abstention accuracy while substantially improving disambiguated-context utility and reducing false abstention, without relying on oracle condition labels at test time.

Generalization claim:

> Leave-one-category-out and Open-BBQ transfer experiments indicate that the behavior is not explained solely by category memorization or by tuning to the original BBQ split.

SAE/s7 wording:

> The SAE-derived signal is included in the full signal set and explicitly audited through signal masking. Its isolated ablation effect is small, suggesting that the final behavior is driven by the combined decision mechanism rather than by a single SAE feature.

Avoid:

- Do not claim lowest ambiguous residual bias.
- Do not claim s7 is the main driver.
- Do not treat Fairsteer as a primary full-coverage baseline when overlap is small.
- Do not claim significant improvement over self-debiasing on ambiguous accuracy alone.
