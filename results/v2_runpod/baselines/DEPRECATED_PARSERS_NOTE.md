# Stale summary JSONs — pre-correction parsers

The `final.json` / summary metrics in `mpt/` and `composite/` under this
directory were computed with **pre-correction answer parsers** and are kept
only as raw provenance. They do NOT match the manuscript:

| Artifact | Stale value | Manuscript (Table 2) | Why |
|---|---|---|---|
| `mpt/final.json` | acc_amb 0.3003 / acc_dis 0.3423 | 0.9399 / 0.2778 / FAR 0.6381 | the original parser took an early answer-like letter from free-form text; the strict re-parse accepts only a final complete `Answer: (X)` line (`scripts/reparse_mpt_baseline.py`) |
| `composite/final.json` and clean-suite composite rows (0.6843) | acc_amb ~0.68 | 0.7181 ± 0.0234 | pre-word-boundary `extract_letter` misread "Answer: Based on …" as (B); corrected parser in `src/baselines/composite_prompting.py` |

The manuscript rows are regenerated from the **raw responses** in
`*/predictions.jsonl` via the corrected parsers — see
`scripts/recompute_bias_direction.py` (composite section) and
`scripts/reparse_mpt_baseline.py`. 1,318/8,864 (14.9%) Composite responses remain
unparseable under the corrected parser and score as wrong (standard
convention). Under the strict MPT convention, 76/1,332 responses (5.7%) lack a
complete final answer and score as wrong; the exact outputs are
`mpt/strict_reparse_metrics.json` and `mpt/strict_reparsed_predictions.jsonl`.
DeCAP and SDR have no recorded parse failures.

Note: the legacy matched-subset `mpt` rows (n≈196 per seed) in older clean-suite
summaries used a protocol-mismatched subset and are superseded by the
full-test-split MPT row reported in the manuscript.
