# Reproducibility Commands

Environment:

```bash
python -m pip install -r requirements.txt
```

Main clean suite:

```bash
python scripts/run_clean_experiments.py \
  --seeds 42 123 456 789 999 \
  --out-dir results/v2/clean_experiments \
  --run-signal-ablation
```

Clean LOCO:

```bash
python scripts/run_loco_clean.py \
  --seeds 42 123 456 789 999 \
  --out-dir results/v2/acceptance_package/loco
```

Open-BBQ transfer:

```bash
python -m src.transfer.run_open_bbq \
  --max-samples 300 \
  --out-dir results/v2/acceptance_package/open_bbq \
  --force --model main
```

Cross-LLM 5-seed summaries from existing signals:

```bash
python -m src.analysis.multi_seed --version v2 --model qwen \
  --seeds 42,123,456,789,999 \
  --out-dir results/v2/cross_llm/qwen/multi_seed_5seed

python -m src.analysis.multi_seed --version v2 --model mistral \
  --seeds 42,123,456,789,999 \
  --out-dir results/v2/cross_llm/mistral/multi_seed_5seed
```

Build paper/appendix tables:

```bash
python scripts/build_acceptance_report.py
```
