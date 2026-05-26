# Confidence-Aware Multi-Signal Debiasing

LLM social-bias mitigation for BBQ-style question answering. The method keeps the model weights fixed, extracts multiple confidence signals, and applies condition-aware abstention so that ambiguous examples are answered conservatively while disambiguated examples keep utility.

This README is intentionally short. It is organized for paper review, reproduction, and figure reuse.

## Current Status

Latest reviewer-defense package: **2026-05-26**

Main safe claim:

> The proposed method preserves high ambiguous-context abstention accuracy while substantially improving disambiguated-context utility and reducing false abstention, without relying on oracle condition labels at test time.

What not to overclaim:

- Do not claim the best ambiguous residual bias score. The residual denominator is tiny when ambiguous accuracy is near-perfect, so `abs_bias_amb` is unstable.
- Do not claim that SAE feature `s7` is the main driver. It is included and audited, but its isolated ablation effect is small.
- Do not use FairSteer as a primary full-coverage baseline. Its matched-ID overlap is too small and should stay in the appendix.

## Key Results

### Main Clean BBQ

Llama-3.1-8B, clean acceptance package, 5 seeds, `n_test=1,328`.

| Variant | acc_amb | acc_dis | FAR | Role |
|---|---:|---:|---:|---|
| predicted-condition | **0.9946 ± 0.0054** | **0.8732 ± 0.0108** | **0.0843 ± 0.0193** | main no-oracle claim |
| oracle per-condition | 0.9946 ± 0.0054 | 0.8738 ± 0.0109 | 0.0837 ± 0.0194 | upper bound |
| single-threshold | 0.9494 ± 0.0126 | 0.8413 ± 0.0184 | 0.1325 ± 0.0240 | deployable fallback |

### Reviewer-Defense Experiments

| Experiment | Setting | Result | Why it matters |
|---|---|---|---|
| Clean LOCO | 9 held-out categories × 5 seeds | acc_amb **0.9214 ± 0.0421**, acc_dis **0.8331 ± 0.0793**, FAR **0.1161 ± 0.0551** | argues against category memorization |
| Open-BBQ fresh transfer | 11 categories, `n=3,300` | acc_amb **0.9915**, acc_dis **0.8358**, FAR **0.1012** | argues against original-BBQ split overfit |
| Cross-LLM | Qwen + Mistral, 5 seeds each | Qwen **0.9895/0.8147/FAR 0.1672**; Mistral **0.9940/0.7798/FAR 0.1916** | argues against Llama-only tuning |
| Threshold repetition | Llama/Qwen/Mistral × 15 runs | `tau_dis = 0.05`, std **0.000** | robust empirical pattern, but still grid-boundary limited |
| SAE/s7 audit | Open-BBQ signal extraction | `s7_bias_sae_feature_count=56` | confirms s7 signal path is active |

## Figures

Use the PDF files in `results/figures/` for the paper. The matching PNG files in `docs/figures/` are for README previews.

### Figure 1. Pipeline

![Pipeline](docs/figures/fig1_pipeline.png)

### Figure 3. MoE Aggregator

![MoE architecture](docs/figures/fig3_moe_architecture.png)

### Figure 4. Main Comparison

This figure intentionally focuses on `acc_amb`, `acc_dis`, and FAR. Residual ambiguous bias is better reported as raw counts/CI in the appendix.

![Main results](docs/figures/fig4_main_results.png)

### Figure 5. Gate Weights by Category

![Cluster routing](docs/figures/fig5_cluster_routing.png)

### Additional Diagnostic Figures

![Risk coverage](docs/figures/risk_coverage_curve.png)

![Bias heads](docs/figures/bias_heads_heatmap.png)

## Method Summary

The pipeline has four stages:

1. Run four prompt variants: vanilla, debiasing prompt, chain-of-thought, and counterfactual swap.
2. Extract seven confidence/bias signals:
   - `s1`: logit confidence
   - `s2`: multi-prompt consistency
   - `s3`: counterfactual stability
   - `s4`: evidence-quote consistency
   - `s5`: self-consistency
   - `s6`: bias-head attention
   - `s7`: SAE bias-feature activation
3. Aggregate signals with a small 4-expert MoE conditioned on the question embedding.
4. Apply threshold override. Low confidence is changed to the unknown answer.

The central empirical pattern is:

| Model family | Seeds | `tau_dis` |
|---|---:|---:|
| Llama-3.1-8B | 5 | 0.05 ± 0.000 |
| Qwen-2.5-7B | 5 | 0.05 ± 0.000 |
| Mistral-7B-v0.3 | 5 | 0.05 ± 0.000 |

Interpret this as a low-threshold saturated pattern on the current canonical grid, not as proof that `0.05` is the continuous optimum.

## Repository Layout

| Path | Purpose |
|---|---|
| `run_pipeline.py` | main BBQ pipeline entry point |
| `src/signals/` | signal extraction |
| `src/models/` | MoE aggregator and threshold override |
| `src/transfer/` | Open-BBQ / KoBBQ / transfer experiments |
| `src/analysis/` | multi-seed, ablation, qualitative, plotting utilities |
| `src/paper/figures.py` | paper figure generator |
| `scripts/run_clean_experiments.py` | clean main-suite runner |
| `scripts/run_loco_clean.py` | clean leave-one-category-out runner |
| `scripts/run_acceptance_package.py` | reviewer-defense package runner |
| `scripts/build_acceptance_report.py` | appendix/report table builder |
| `docs/figures/` | README PNG previews |
| `results/figures/` | paper-ready PDF/PNG figures |

Large generated predictions and run outputs are local artifacts. Do not treat every `results/` subdirectory as something that must be committed.

## Reproduction

### Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Required for gated Llama weights
echo "HF_TOKEN=hf_..." > .env
```

Recommended hardware:

| Task | Suggested hardware |
|---|---|
| Llama-3.1-8B inference | CUDA GPU 16GB+ or Apple Silicon 64GB |
| SAE feature extraction | CUDA GPU recommended |
| Clean LOCO / transfer package | H100 recommended |
| MoE training / report building | CPU is enough |

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

One-shot:

```bash
python scripts/run_acceptance_package.py
```

Or run the key pieces explicitly:

```bash
# Leave-one-category-out
python scripts/run_loco_clean.py \
  --seeds 42 123 456 789 999 \
  --out-dir results/v2/acceptance_package/loco

# Open-BBQ fresh transfer.
# --max-samples 300 means 300 examples/category, 11 categories, n=3,300 total.
python -m src.transfer.run_open_bbq \
  --max-samples 300 \
  --out-dir results/v2/acceptance_package/open_bbq \
  --force --model main

# Cross-LLM 5-seed summaries from existing signals
python -m src.analysis.multi_seed --version v2 --model qwen \
  --seeds 42,123,456,789,999 \
  --out-dir results/v2/cross_llm/qwen/multi_seed_5seed

python -m src.analysis.multi_seed --version v2 --model mistral \
  --seeds 42,123,456,789,999 \
  --out-dir results/v2/cross_llm/mistral/multi_seed_5seed

# Paper/appendix tables
python scripts/build_acceptance_report.py
```

### Regenerate Figures

```bash
# Main paper figures
python -m src.paper.figures --figs 1 3 4 5 --out-dir results/figures

# README copies
python -m src.paper.figures --figs 1 3 4 5 --out-dir docs/figures

# Diagnostic figures
python -m src.analysis.qualitative \
  --tasks bias_heads_heatmap risk_coverage \
  --out-dir results/figures

python -m src.analysis.qualitative \
  --tasks bias_heads_heatmap risk_coverage \
  --out-dir docs/figures
```

## Paper Writing Notes

Use these claims:

- The method preserves ambiguous abstention accuracy while improving disambiguated utility.
- Predicted-condition results are the deployable main setting.
- LOCO and Open-BBQ transfer reduce the risk that the method only memorizes BBQ category patterns.
- Cross-LLM results show that the behavior is not Llama-only.

Avoid these claims:

- "We achieve the lowest ambiguous bias score."
- "s7 is the reason the method works."
- "FairSteer proves superiority as a full baseline."
- "0.05 is the true continuous optimum."

## Licenses and Data

Datasets and models retain their original licenses:

- BBQ: NYU MLL, CC-BY-4.0
- Open-BBQ: CC-BY-4.0
- KoBBQ: CC-BY-SA-4.0
- Winogender: Rudinger et al., NAACL 2018
- Llama-3.1-8B: Meta Llama license
- Qwen-2.5-7B: Apache 2.0
- Mistral-7B-v0.3: Apache 2.0

## Citation

This repository is a research artifact for the paper draft:

```bibtex
@misc{confidence_aware_bias_mitigation_2026,
  title = {Confidence-Aware Multi-Signal Debiasing with Condition-Aware Abstention},
  author = {KMS},
  year = {2026},
  note = {Research artifact}
}
```
