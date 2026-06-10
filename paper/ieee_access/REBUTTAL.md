# Response to Reviewer Comments

**Manuscript:** Condition-Aware Selective Abstention for Bias Benchmark Question Answering: A Multi-Signal Audit of BBQ Condition Separability

We thank the reviewer for the careful and accurate reading of the paper and for the constructive, well-scoped requests. We are glad the framing (a frozen-model, inference-time, no-oracle condition-aware selective-abstention policy) and the central finding (the original BBQ split is condition-separable from raw text, so a condition-only policy already reaches near-ceiling ambiguous accuracy while preserving disambiguated utility, and the seven-signal MoE adds little on the clean split but is useful as a low-label / uncertain-condition fallback) were understood as intended, and that the audit/diagnostic framing — rather than a debiasing-mechanism claim — was read as a strength.

All seven points are addressed below. None required new large-scale experiments: the responses are (a) implementation-detail clarifications, (b) one new quantitative table, (c) targeted expansions of existing text, and (d) an explicit artifact release. New/changed locations are noted per item.

---

## R1 — Implementation details of $s_1$ (evidence) and $s_2$ (counterfactual)
**Added: Appendix "Signal Implementation Details and Inference Cost" (`app:signal_impl`).**

- **$s_1$ evidence.** Under a verbatim-extraction system prompt the model quotes the exact contiguous supporting span or outputs `NONE` (greedy decoding, $T{=}0$, $\le96$ tokens). The reply is parsed (triple-quoted span → quoted substring → first line) and scored after normalization (lowercasing, whitespace collapse, punctuation removal): **1.0** if the normalized quote is a substring of the normalized context, **0.5** if the unigram token-overlap ratio $|q\cap c|/|q|\ge0.5$, else **0.0**. So $s_1$ is a quote-**grounding** proxy, not entailment; its main **failure mode is over-quoting** (a whole-context span yields $s_1{=}1.0$ even on an ambiguous item with no real evidence — shown at the token level in Fig. 5), which is why the decision relies on the condition/MoE layer, not $s_1$ alone.
- **$s_2$ counterfactual.** The two non-`unknown` answer entities are swapped in the context (case-insensitive, **word-bounded** regex with a placeholder for a safe 3-way exchange; options, `answer_info` tags, and gold label swapped consistently) and the frozen model is re-queried; $s_2{=}1.0$ iff the post-swap answer maps to the same group **tag** (i.e., the answer follows the group, not the surface position). We now state the three implementation facts the reviewer asked about: **(i) multi-word handling** — full strings are matched, so names swap reliably, but when the option string differs from the in-context surface form ("the Malian friend" vs. "my Malian friend") the bounded match can fail and $s_2$ reduces to a same-context re-query; **(ii) semantic coherence** — the swap is a *syntactic* exchange (structure/length preserved by construction), with no explicit coherence check, relying on BBQ's two-group symmetry; **(iii)** if an item lacks exactly two non-`unknown` groups, $s_2$ is undefined and defaults to 1.0.

## R2 — Stress-set construction (templates, removed-vs-retained cues, examples, human validation)
**Expanded the stress paragraph in `app:condition_audit` (Table `tab:followup_stress`).**

The set is built deterministically from BBQ in **three variants** (entity-masked, light-rewrite, combined): non-`unknown` demographic entities → "Person A/B", plus **14 word-bounded template rewrites** (e.g., `I saw→There were`, `last week→recently`, `Who was→Which person was`) applied to context/question only. **Removed cues:** surface demonstratives/temporal/template phrasings and the demographic entity surface forms. **Retained:** the disambiguating evidence sentence (not paraphrased) and all metadata (condition label, answers, category). **Scale:** stratified sample of 2,700 instances (≤150 per category–condition pair, seed 42) × 3 variants = **8,100** stressed examples; condition accuracy drops 0.9961→0.9373. **Human validation: none** (deterministic by design; we state it is *not* a human-validated paraphrase benchmark and flag a human-verified adversarial set as future work). Code: `scripts/generate_paraphrase_stress_bbq.py`; data: `data/bbq_stress/paraphrase_template_stress.jsonl` (released).

## R3 — Per-category FAR calibration (Physical appearance / Religion disparity)
**Expanded the calibration discussion near Table 26 (`tab:category_calibration_followup`).**

We now report that the validation-only category-aware audit reduces aggregate FAR only marginally (**0.0726→0.0717**) while lowering condition accuracy (**0.9983→0.9928**); i.e., per-category thresholds barely help and slightly hurt generalization. We therefore frame per-category/per-template calibration — which would target the **Physical appearance (FAR 0.179)** and **Religion (FAR 0.123)** gap — as a deployment option that must be tied to an **explicit fairness objective**, not enabled by default. Full per-category FAR is in Table 30; the validation-only sweep is `src/analysis/threshold_sweep.py:per_category_threshold`.

## R4 — Inference cost
**Added a cost table (`tab:cost`) in `app:signal_impl`.**

Per example, the **full pipeline = 11 LLM generations** (4 prompt views reused for $s_6$ + $s_1$ + $s_2$ + $5$ self-consistency samples for $s_4$, $T{=}0.7$); $s_3,s_5,s_6,s_7$ add **0** generations (log-probs / attention / SAE forward passes). The **condition-only policy = 1 generation + 1 sentence-embedding pass** (`all-MiniLM-L6-v2`). So the signal layer costs ~an order of magnitude more generation for **no clean-split gain**, which is exactly why it is reserved for the low-label / uncertain-condition regime — the **hybrid break-even** the reviewer asked about.

## R5 — Human-centered study of the explanation layer
**Strengthened Limitation 7.**

This is outside the frozen-LLM inference-time scope and is already noted as a limitation; we now add an explicit **future-work commitment**: a small **auditor pilot** measuring whether the signal-based bias-risk rationales improve error triage and trust calibration.

## R6 — Pure lexical condition classifier for low-resource deployment
**Added a sentence after the lexical/encoder audit (Table `tab:lexical_encoder_audit`).**

Because word/character n-gram classifiers nearly match the neural encoder (0.9968 condition accuracy), we now note that a **pure-lexical condition classifier is a viable low-resource deployment default** that removes the sentence-embedding dependency entirely.

## R7 — Artifact release
**Added a `Data Availability` section.**

All code/artifacts are at `https://github.com/KMS-gif375/LLM-Bias-Mitigation`. We now explicitly release and name the reviewer-requested items: **attention-head list** `results/bias_heads.json` (20 ranked heads), **SAE feature indices** `results/v2_runpod/sae_layers/features_layer15.json` (56 indices, Llama-Scope layer 15), **MoE checkpoint** `results/v2_runpod/moe/main/moe_best.pt`, **stress file** `data/bbq_stress/paraphrase_template_stress.jsonl`, and the per-token signal scripts. The **condition classifier** is a balanced logistic-regression model fully specified by the released code + features + fixed seeds (reproducible via `scripts/run_clean_experiments.py`).

---

### Summary of manuscript changes
| Item | Change | Location |
|---|---|---|
| R1, R4 | New appendix: $s_1$/$s_2$ implementation + cost table | `app:signal_impl`, `tab:cost` |
| R2 | Stress-set construction expanded (variants, 14 rewrites, scale, no human-val) | `app:condition_audit` |
| R3 | Category-calibration numbers + per-template guidance | near `tab:category_calibration_followup` |
| R5 | Auditor-pilot future work | Limitation 7 |
| R6 | Pure-lexical low-resource default | after `tab:lexical_encoder_audit` |
| R7 | Data Availability section + named artifacts | before Funding |

All numeric changes were verified against the source code and saved result files. No claims were altered; the additions clarify implementation, quantify cost, and release artifacts, consistent with the paper's audit framing.

---

# Addendum — full reviewer report (Questions 1–7 + weaknesses)

Two new analyses were run (real numbers below); two are provided as ready-to-run scripts; the rest are clarifications/related-work.

## Q1 — Comparison to Multi-Persona Thinking (MPT) **(new result)**
Added MPT \cite{chen2026mpt} to related work and **ran an MPT-style single-pass reimplementation on the clean test split (1,332 examples, H100, deterministic)**; the row is added to Table 2 (`tab:main`). Result: **Acc_amb 0.9565, Acc_dis 0.3003, FAR 0.6667** — the multi-persona prompt reaches ambiguous accuracy close to the condition-aware policies, but only by abstaining on two thirds of disambiguated items (the same over-abstention failure mode as the self-debiasing-style baseline, in milder form; cf. condition-only FAR 0.0726 at Acc_dis 0.8789). This directly supports the paper's decomposition: perspective-diverse prompting buys ambiguous safety through indiscriminate abstention, while condition-aware abstention preserves disambiguated utility. We disclose that this is an unofficial single-prompt reimplementation of the multi-persona idea, not the authors' full iterative protocol. Script: `scripts/run_mpt_baseline.py` (`--mode single|panel`); raw predictions released under `results/v2_runpod/baselines/mpt/`.

## Q2 — Comparison to ASPIRE-style learned self-evaluation **(new result)**
We added ASPIRE \cite{chen2023aspire} to related work and a **learned self-evaluation row to Table 7** (`tab:aurc`). A logistic self-eval trained on the same saved signals to predict correctness gives **AURC 0.1053, E-AURC 0.0616, AUROC 0.826**, while the **MoE retention score ranks better (AURC 0.0626, AUROC 0.945)**; both beat raw softmax (AURC 0.1197). Script: `scripts/run_aspire_selfeval.py` (CPU; reproduces the existing Table 7 rows exactly).

## Q3 — Quantitative compute/latency
New cost table `tab:cost`: full pipeline = **11 generations/example** (4 prompt views + s1 + s2 + 5 self-consistency), condition-only = **1 generation + 1 embedding**; s3/s5/s6/s7 add no generations. Added wall-clock anchors (~0.06 s/example inference-only; ~8 s/example for a multi-call CoT baseline).

## Q4 — Condition-classifier features and residual delta
Reported: the deployed classifier is **embedding-only (0.9961)**; adding all non-text features (signals + primary + category) raises accuracy only to **0.9983 (+0.22 pt)**; signals-only 0.833, category-only 0.500. So non-text features contribute almost nothing — the text embedding alone is near-ceiling (source: `condition_classifier_ablation_summary.csv`).

## Q5 — Human-validated paraphrase / adversarial
Out of frozen-LLM scope; kept as an explicit limitation with a future-work commitment (human-verified adversarial BBQ).

## Q6 — Fairness-aware calibration (group-wise equalized FAR)
Added: per-category calibration gives negligible aggregate gain (FAR 0.0726→0.0717) and lowers condition accuracy (0.9983→0.9928); we name **equalized FAR across groups** as the natural objective and note its Acc_amb trade-off (1.0000→0.9946), left to a fairness-objective deployment study.

## Q7 — Multilingual encoders on KoBBQ **(new result)**
We tested multilingual encoders for English→KoBBQ condition transfer (no Korean retraining). **LaBSE 0.977 and multilingual-E5 0.885 vs all-MiniLM 0.493** (near chance); within-KoBBQ ≈ 1.0 for all. So multilingual encoders **restore** cross-lingual condition separability, confirming the structural property is language-robust given a multilingual encoder. Script: `scripts/run_multilingual_condition.py` (CPU).

## Weaknesses
- **Novelty reliance on condition separability (W1):** strengthened the Novelty/Scope framing — the diagnostic/negative result is itself the contribution (benchmark-artifact audit genre); the MoE is retained for the low-label/uncertain-condition regime, not the clean split.
- **Symbol reuse, TeX (W5):** defined $\tilde{y}_i$ (final post-override answer) alongside $\hat{y}_i$/$\hat{c}_i$; no other artifacts found.
- **Missing related work (W7/W8):** added MPT, ASPIRE, and the OOD/selective-classification survey \cite{lu2025ood}, situating our policy in the training-agnostic, frozen-model family.

**New artifacts/scripts:** `scripts/run_mpt_baseline.py`, `scripts/run_aspire_selfeval.py`, `scripts/run_multilingual_condition.py`; new audit outputs under `results/v2/reviewer_audits/{aspire_selfeval,multilingual_condition}/`.
