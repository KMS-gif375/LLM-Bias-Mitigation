# Reviewer Audit Report

All results are computed from saved signals, embeddings, and predictions; no LLM inference is run.

## Condition Classifier Feature Ablation
| Feature set | Test acc. |
|---|---:|
| signals+embedding+category+primary | 0.9983 +/- 0.0011 |
| minus_primary | 0.9982 +/- 0.0014 |
| signals+embedding | 0.9980 +/- 0.0011 |
| embedding_only | 0.9961 +/- 0.0023 |
| signals_only | 0.8334 +/- 0.0078 |
| primary_only | 0.5440 +/- 0.0093 |
| category_only | 0.5000 +/- 0.0000 |

## Simple Policy Baselines
| System | Acc_amb | Acc_dis | FAR |
|---|---:|---:|---:|
| condition_only_embedding | 0.9994 +/- 0.0008 | 0.8774 +/- 0.0082 | 0.0741 +/- 0.0083 |
| condition_only_full_features | 1.0000 +/- 0.0000 | 0.8789 +/- 0.0070 | 0.0726 +/- 0.0067 |
| condition_only_signals_embedding | 0.9994 +/- 0.0008 | 0.8789 +/- 0.0079 | 0.0726 +/- 0.0067 |
| primary_answer_only | 0.5596 +/- 0.0152 | 0.8798 +/- 0.0076 | 0.0717 +/- 0.0069 |
| s3_only_predicted_condition | 1.0000 +/- 0.0000 | 0.8789 +/- 0.0070 | 0.0726 +/- 0.0067 |

## LOCO Condition Prediction
| Feature set | Held-out acc. |
|---|---:|
| embedding_only | 0.8909 +/- 0.0320 |
| signals+embedding | 0.9259 +/- 0.0268 |
| signals+embedding+primary | 0.9261 +/- 0.0273 |
| signals_only | 0.8125 +/- 0.0667 |

## Low-threshold Plateau
| tau_dis | Acc_amb | Acc_dis | FAR |
|---:|---:|---:|---:|
| 0.00 | 0.9946 +/- 0.0054 | 0.8798 +/- 0.0076 | 0.0717 +/- 0.0069 |
| 0.01 | 0.9946 +/- 0.0054 | 0.8789 +/- 0.0087 | 0.0744 +/- 0.0106 |
| 0.02 | 0.9946 +/- 0.0054 | 0.8780 +/- 0.0101 | 0.0771 +/- 0.0139 |
| 0.03 | 0.9946 +/- 0.0054 | 0.8768 +/- 0.0098 | 0.0792 +/- 0.0152 |
| 0.05 | 0.9946 +/- 0.0054 | 0.8738 +/- 0.0109 | 0.0837 +/- 0.0194 |
| 0.10 | 0.9946 +/- 0.0054 | 0.8696 +/- 0.0132 | 0.0916 +/- 0.0215 |

## MoE Signal Subsets
| Variant | Signals kept | Acc_amb | Acc_dis | FAR |
|---|---|---:|---:|---:|
| all_zero_moe | <none> | 1.0000 +/- 0.0000 | 0.8789 +/- 0.0070 | 0.0726 +/- 0.0067 |
| core4_s1346_moe | s1_evidence,s3_confidence,s4_consistency,s6_prompt_sensitivity | 0.9988 +/- 0.0027 | 0.8789 +/- 0.0070 | 0.0726 +/- 0.0067 |
| s3_only_moe | s3_confidence | 0.9985 +/- 0.0015 | 0.8789 +/- 0.0070 | 0.0726 +/- 0.0067 |
