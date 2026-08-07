# Reproducing the paper

All commands run from the repo root with the project venv. Seeds are fixed
(42, 123, 456, 789, 999). GPU-marked steps need the LLM (we used 1x H100,
bfloat16). The remaining steps operate on saved artifacts and are CPU-capable,
although scripts whose device remains `auto` may use an available accelerator.

## 0. Environment

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
# GPU steps additionally need: transformers==4.46.3 sae_lens accelerate
# and HF_TOKEN in .env (Llama-3.1-8B-Instruct license)
```

## 1. Obtain the exact release and replay assets

Tag `v1.1.0` is the immutable submission snapshot. Binary embedding tensors and
validation-best checkpoints are distributed separately on the GitHub release
because of their size. The archive contains repository-relative paths and can be
extracted directly at the repository root.

```bash
git clone --branch v1.1.0 --depth 1 \
  https://github.com/KMS-gif375/LLM-Bias-Mitigation.git
cd LLM-Bias-Mitigation

BASE=https://github.com/KMS-gif375/LLM-Bias-Mitigation/releases/download/v1.1.0
curl -fL "$BASE/CASA_replay_assets_v1.1.0.tar.gz" \
  -o CASA_replay_assets_v1.1.0.tar.gz
curl -fL "$BASE/SHA256SUMS.txt" -o SHA256SUMS.txt

# Verify the downloaded archive on Linux:
grep 'CASA_replay_assets_v1.1.0.tar.gz' SHA256SUMS.txt | sha256sum -c -
# macOS alternative:
# grep 'CASA_replay_assets_v1.1.0.tar.gz' SHA256SUMS.txt | shasum -a 256 -c -

tar -xzf CASA_replay_assets_v1.1.0.tar.gz

# Verify every extracted checkpoint and embedding on Linux:
sha256sum -c REPLAY_ASSET_SHA256SUMS.txt
# macOS alternative:
# shasum -a 256 -c REPLAY_ASSET_SHA256SUMS.txt
```

The release archive includes the main checkpoint at
`results/v2_runpod/moe/main/moe_best.pt`, the saved embedding tensors, and the
Qwen/Mistral validation-best checkpoints used by the documented artifact-only
replays. It contains no credentials or private raw data.

## 2. Signal extraction (GPU) — or reuse the released artifacts

Tracked JSONL signals and metrics for the full 8,864-example pool are included
under `results/v2/signals/main/`. Binary embedding tensors and checkpoints are
restored by step 1. The full pipeline that produced the signals is
`run_pipeline.py` (stage 1 inference + seven-signal extraction + MoE +
override). Regenerating them from scratch requires gated access to
Llama-3.1-8B-Instruct and a suitable GPU; artifact-only replays do not perform
new LLM inference.

## 3. Original diagnostic clean five-seed suite (CPU)

```bash
./venv/bin/python scripts/run_clean_experiments.py \
    --seeds 42 123 456 789 999 --run-signal-ablation \
    --out-dir results/v2/clean_experiments
# outputs: metrics_summary.csv, paired_tests.csv, aggregate_metrics.csv, summary.json
```

This command reconstructs the original diagnostic checkpoint family used by
the auxiliary paired, threshold, masking, ranking, and explanation audits. The
corrected-full main-table MoE and condition-only rows are produced in step 7.

## 4. Reviewer audit suites (CPU)

```bash
./venv/bin/python scripts/run_minor_revision_audits.py      # confidence-only, learned rejectors, risk-coverage, category coverage
# Fast reviewer controls and the single-prompt low-training-label audit
./venv/bin/python scripts/run_reviewer_extra_audits.py \
    --audits lexical embedding conformal single_prompt \
    --out-dir results/v2/reviewer_extra_audits_main
# NLI evidence audit (GPU)
./venv/bin/python scripts/run_reviewer_extra_audits.py \
    --audits nli --device cuda \
    --out-dir results/v2/reviewer_extra_audits_nli_full
# Condition-aware risk control, calibration, cross-backbone, and stress audits
./venv/bin/python scripts/run_reviewer_extra_audits.py \
    --audits condition_conformal category_calibration cross_llm paraphrase \
    --paraphrase-file data/bbq_stress/paraphrase_template_stress.jsonl \
    --out-dir results/v2/reviewer_extra_audits_followup
./venv/bin/python scripts/run_hybrid_abstention_audits.py   # low-training-label + uncertainty fallback (Table 4)
./venv/bin/python scripts/run_r2_audits.py                  # template-disjoint, rescue, overlap, s2 no-op, keeps, cond-only transfer
./venv/bin/python scripts/run_loco_clean.py                 # LOCO (Tables 6, 14, 15)
```

`run_hybrid_abstention_audits.py` reproduces the historical low-training-label
protocol, but not the exact five-seed table after random seeding was moved ahead
of model construction. Only the original seed-level summaries were retained;
the per-example decisions and historical low-label checkpoints were not. A
controlled seed-42 1% replay therefore differs slightly from the historical
row, and `results/v2/reviewer_audits/r2_audits/lowlabel_bootstrap_validity.json`
marks exact reconstruction as false. The fraction applies only to the condition
classifier's training labels: at 1%, that classifier uses 62 labeled training
items, while the fixed MoE was trained on all 6,208 training items and both
hybrid thresholds use the full 1,328-item labeled validation split. The test
split is untouched. Consequently,
`scripts/run_lowlabel_bootstrap.py` is only a proxy reconstruction using a
different embedding-only label draw and saved MoE scores; it does **not**
reproduce exact paired low-label p-values and is not part of the inferential
replication path.

The archived `2.31 s` latency number is a historical 11-generation schedule
proxy, not an end-to-end timing of the current 12-generation-plus-five-forward
extractor. Its fourth view used the legacy short prompt and its `s2` query did
not perform the actual demographic swap. The current benchmark script exposes
`--schedule legacy` to document that archived call schedule and `--schedule
current` for the revised counterfactual-instruction view; either remains a
generation-schedule proxy and must not be reported as full-pipeline latency.

The saved MPT generations can be audited without another GPU run. The release
currently stores raw text in `predictions.jsonl`; the reparsing script also
accepts a `raw_outputs.jsonl` file when present and never overwrites either raw
input or the legacy `final.json`:

```bash
./venv/bin/python scripts/reparse_mpt_baseline.py
# -> results/v2_runpod/baselines/mpt/strict_reparse_metrics.json
# -> results/v2_runpod/baselines/mpt/strict_reparsed_predictions.jsonl
```

## 5. Explanations and bias-risk artifacts (CPU)

```bash
./venv/bin/python scripts/generate_rule_based_explanations.py
#   -> results/v2/rule_explanations/ours_predicted_condition/{explanations.jsonl, bias_risk_summary.csv}
./venv/bin/python scripts/show_bias_risk_explanation.py     # print representative rationales
```

## 6. Stress / transfer (data: CPU; evaluation: GPU)

```bash
./venv/bin/python scripts/generate_paraphrase_stress_bbq.py # -> data/bbq_stress/paraphrase_template_stress.jsonl
./venv/bin/python -m src.transfer.run_open_bbq              # raw oracle-routed Open-BBQ feature transfer (GPU)
./venv/bin/python scripts/run_transfer_condition_audits.py  # KoBBQ condition-transfer audits (CPU)
./venv/bin/python -m src.analysis.multi_seed --version v2 --model qwen \
    --seeds 42,123,456,789,999 \
    --out-dir results/v2/cross_llm/qwen/multi_seed_5seed
./venv/bin/python -m src.analysis.multi_seed --version v2 --model mistral \
    --seeds 42,123,456,789,999 \
    --out-dir results/v2/cross_llm/mistral/multi_seed_5seed
```

The raw transfer runners select their condition-specific threshold from the
target dataset's gold `context_condition`; their payloads therefore record
`routing_mode=oracle_target_condition`. The paper's no-oracle Open-BBQ and
KoBBQ rows are reconstructed separately by `run_transfer_routing_unify.py` and
`recompute_kobbq_deduplicated_routing.py`.

```bash
# Replay Open-BBQ with the checkpoint distributed in v1.0-ieee-access.
./venv/bin/python scripts/run_transfer_routing_unify.py \
    --moe-checkpoint results/v2_runpod/moe/main/moe_best.pt \
    --out-dir results/v2/reviewer_audits/routing_unify_published
# -> report.json (records checkpoint SHA-256 6e63661c...) and
#    open_bbq_cluster_routing.csv
```

The retained ImplicitBBQ artifact is a legacy single-threshold run: both
condition thresholds equal 0.5, so the stored gold condition is
decision-irrelevant. Its complete 2,640 generated texts and embeddings were not
retained, and neither a new predicted-condition comparison nor an exact
end-to-end replay is possible from the released files.

The cross-LLM commands reuse already extracted backbone-specific signals, then
train a backbone-specific MoE for 30 epochs, restore the validation-best
checkpoint, choose thresholds on validation, and evaluate the test split using
gold BBQ conditions for routing. They skip expensive signal extraction, but
they are not inference-only summaries or no-oracle cross-backbone transfers.

The current runner restores the validation-best checkpoint before selecting
validation thresholds and evaluating test data. To audit previously saved
Qwen/Mistral checkpoints without retraining, run:

```bash
./venv/bin/python scripts/recompute_cross_llm_best_checkpoints.py
# -> results/v2/reviewer_audits/cross_llm_best_checkpoint/summary.json

# Reweight the archived KoBBQ run after removing repeated IDs (no LLM calls)
./venv/bin/python scripts/recompute_kobbq_deduplicated_routing.py
# -> results/v2/reviewer_audits/kobbq_deduplicated_routing_published/report.json
./venv/bin/python scripts/run_transfer_condition_audits.py \
    --out-dir results/v2/reviewer_audits/kobbq_deduplicated_condition
# -> results/v2/reviewer_audits/kobbq_deduplicated_condition/

# Direct English-BBQ-to-KoBBQ encoder sensitivity after deduplicating KoBBQ IDs
./venv/bin/python scripts/run_multilingual_condition.py \
    --encoders sentence-transformers/all-MiniLM-L6-v2 \
               sentence-transformers/LaBSE \
               intfloat/multilingual-e5-base \
    --ko-per-cat 150 \
    --out results/v2/reviewer_audits/multilingual_condition_deduplicated
# -> results/v2/reviewer_audits/multilingual_condition_deduplicated/summary.csv
```

## 7. Post-audit corrections (v1.0.2--v1.1.0)

```bash
# corrected stereotype-direction recomputation from saved predictions (CPU)
./venv/bin/python scripts/recompute_bias_direction.py
#   -> results/v2/reviewer_audits/bias_direction_fix/report.json

# corrected s5/s7 re-extraction (GPU, ~6 min on H100) — artifacts already released
./venv/bin/python scripts/extract_corrected_s5s7.py
#   -> results/v2/signals/corrected_s5s7/
# rerun of the clean suite on corrected signals (CPU)
./venv/bin/python scripts/run_clean_experiments.py --model corrected_s5s7 \
    --seeds 42 123 456 789 999 --run-signal-ablation \
    --out-dir results/v2/clean_experiments_corrected_s5s7

# fully corrected retrain (Limitations item i: Eq. 7 labels rebuilt, 11.2% change; CPU)
./venv/bin/python scripts/build_corrected_full_signals.py
#   -> results/v2/signals/corrected_full/
./venv/bin/python scripts/run_clean_experiments.py --model corrected_full \
    --seeds 42 123 456 789 999 --out-dir results/v2/clean_experiments_corrected_full
#   -> 0.9937+-0.0073 / 0.8753+-0.0098 / 0.0822+-0.0157

# Artifact-only replay of the corrected full-feature condition-only row.
# Uses saved predicted_condition and primary_answer fields; ambiguous
# predictions map to the released item's unknown option, and disambiguated
# predictions retain primary_answer. No model or MoE retraining is required.
./venv/bin/python scripts/recompute_condition_only_predicted_metrics.py
#   -> results/v2/clean_experiments_corrected_full/condition_only_predicted_metrics.csv

# transfer routing unification (Table 6 predicted-condition rows; CPU)
./venv/bin/python scripts/run_transfer_routing_unify.py
#   -> results/v2/reviewer_audits/routing_unify_published/report.json

# Recompute the published routing interpretability summary from the archived
# Open-BBQ routing CSV and the public MoE checkpoint.
./venv/bin/python -m src.analysis.moe_interpretability \
    --routing-csv results/v2/reviewer_audits/routing_unify_published/open_bbq_cluster_routing.csv \
    --moe-ckpt results/v2_runpod/moe/main/moe_best.pt \
    --out-dir results/v2/reviewer_audits/routing_unify_published/interpretability

# template-disjoint replication splits (CPU, deterministic)
./venv/bin/python scripts/export_template_disjoint_splits.py
#   -> data/splits/template_disjoint/seed_{42,123,456,789,999}.json
```

The released corrected-full checkpoints predate the fix that seeds every random
generator before model construction. They are retained as observed artifacts;
the command above reproduces the documented protocol with the corrected code,
but bit-identical regeneration of those historical weights is not claimed.

In the condition-aware risk-control artifact, fields named
`global_*_at_fallback_tau` describe the fallback threshold in isolation; they
are not coverage/risk measurements of the complete condition-partitioned
policy. The manuscript reports only the complete-policy BBQ metrics.

## 8. Pinning

* Original runs: tag `v1.0-ieee-access` (commit `9d272e4`), Zenodo DOI
  `10.5281/zenodo.20621246`; MoE checkpoint `moe_best.pt` (SHA-256 `6e63661c…`)
  attached to that GitHub release — this is the main-run checkpoint used by the
  transfer/qualitative pipelines; clean-suite per-seed checkpoints are
  regenerated by step 2.
* Post-audit corrections: tag `v1.0.1` (matcher/converter/parser fixes),
  tag `v1.0.2` (adds corrected s5/s7 extraction code + artifacts, routing
  reconstruction, split export), and tag `v1.0.3` (adds the corrected-full
  label-merge script and this section's commands; annotates stale
  pre-correction baseline summaries — see
  `results/v2_runpod/baselines/DEPRECATED_PARSERS_NOTE.md`).
* Submission artifact: tag `v1.1.0` at the
  [GitHub release](https://github.com/KMS-gif375/LLM-Bias-Mitigation/releases/tag/v1.1.0).
  The immutable Zenodo version DOI is
  [10.5281/zenodo.21839822](https://doi.org/10.5281/zenodo.21839822), while
  [10.5281/zenodo.20621245](https://doi.org/10.5281/zenodo.20621245) is the
  version-independent concept DOI. This version adds the validation-best
  cross-LLM and KoBBQ deduplication recomputations, strict MPT reparse,
  low-training-label validity audit, corrected template-disjoint splits, and
  the final paper source/PDF. The GitHub release attaches the final PDF, LaTeX
  source, `CASA_replay_assets_v1.1.0.tar.gz`, and `SHA256SUMS.txt`; the replay
  archive restores the ignored model and embedding files required by the
  artifact-only commands above.
