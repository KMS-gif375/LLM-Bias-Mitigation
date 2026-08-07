# Results provenance map

This directory contains both the final corrected analyses and historical diagnostic outputs. Use the paths below as the canonical entry points; do not select a result merely because its directory name is newer or because a raw runner produced it.

## Canonical paper results

| Analysis | Canonical path | Notes |
|---|---|---|
| Corrected clean five-seed main results | `v2/clean_experiments_corrected_full/` | Main corrected condition-only and MoE rows |
| Corrected signal reconstruction | `v2/signals/corrected_full/` | Inputs used by the corrected clean suite |
| Original diagnostic clean suite | `v2/clean_experiments/` | Source for scoped threshold, ranking, masking, and explanation audits |
| Confidence and learned-rejector controls | `v2/minor_revision_audits/` | Post-hoc controls on the original diagnostic pipeline |
| Lexical, encoder, conformal, and single-prompt audits | `v2/reviewer_extra_audits_main/` | Reviewer-requested clean-split controls |
| Condition-aware risk control and deterministic stress | `v2/reviewer_extra_audits_followup/` | Follow-up controls and stress data |
| NLI evidence audit | `v2/reviewer_extra_audits_nli_full/` | Separate evidence-checking audit |
| LOCO and Open-BBQ package | `v2/acceptance_package/` | Category-holdout and related-dataset replay |
| Published-checkpoint routing replay | `v2/reviewer_audits/routing_unify_published/` | Uses checkpoint SHA-256 beginning `6e63661c` |
| Deduplicated KoBBQ routing | `v2/reviewer_audits/kobbq_deduplicated_routing_published/` | Canonical archived KoBBQ routing result |
| Deduplicated KoBBQ condition audit | `v2/reviewer_audits/kobbq_deduplicated_condition/` | Condition-transfer audit after ID deduplication |
| Validation-best cross-LLM recomputation | `v2/reviewer_audits/cross_llm_best_checkpoint/` | Qwen and Mistral audit provenance |
| Rule-based runtime traces | `v2/rule_explanations/ours_predicted_condition/` | Deployable traces plus separate benchmark audit labels |

## Historical and auxiliary paths

- `v2_runpod/` contains archived GPU-run outputs and raw transfer artifacts. Some scripts intentionally read these paths for provenance; they are not automatically preferred over the canonical paths above.
- `v2/clean_experiments_corrected_s5s7/` is an intermediate correction audit. The full corrected suite supersedes it for the main table.
- `v2/multi_seed/`, `v2/multi_seed_clean/`, and the historical clean suites document earlier checkpoint families and should not be mixed with the corrected main rows.
- Files explicitly marked `DEPRECATED`, `legacy`, `pre_parser`, or `historical` are retained only to explain protocol evolution.
- Smoke and mini runs are development checks, not paper results.

Large binary embeddings and validation-best checkpoints are intentionally excluded from Git and distributed in `CASA_replay_assets_v1.1.0.tar.gz` on the [v1.1.0 GitHub release](https://github.com/KMS-gif375/LLM-Bias-Mitigation/releases/tag/v1.1.0). See [REPRODUCING.md](../REPRODUCING.md) for checksum verification and extraction instructions.

The immutable submission snapshot is tag `v1.1.0` with Zenodo DOI [`10.5281/zenodo.21839822`](https://doi.org/10.5281/zenodo.21839822).
