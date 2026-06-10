# Hybrid Abstention Audit Report

All runs use saved signals/embeddings only; no LLM inference is run.

## Clean BBQ

| System | runs | Acc_amb | Acc_dis | FAR | cond. acc. |
|---|---:|---:|---:|---:|---:|
| hybrid_uncertain_signal_fallback | 1 | 1.0000+/-0.0000 | 0.8765+/-0.0000 | 0.0798+/-0.0000 | 0.9970+/-0.0000 |
| primary_answer | 1 | 0.5617+/-0.0000 | 0.8780+/-0.0000 | 0.0783+/-0.0000 | - |
| seven_signal_moe_predicted | 1 | 1.0000+/-0.0000 | 0.8765+/-0.0000 | 0.0798+/-0.0000 | - |
| simple_condition_only | 1 | 1.0000+/-0.0000 | 0.8765+/-0.0000 | 0.0798+/-0.0000 | 0.9970+/-0.0000 |

## Low-label Condition Classifier

| frac | system | runs | Acc_amb | Acc_dis | FAR | cond. acc. |
|---:|---|---:|---:|---:|---:|---:|
| 0.1 | hybrid_uncertain_signal_fallback | 1 | 0.9849+/-0.0000 | 0.8599+/-0.0000 | 0.0994+/-0.0000 | 0.9593+/-0.0000 |
| 0.1 | simple_condition_only | 1 | 0.9864+/-0.0000 | 0.8449+/-0.0000 | 0.1145+/-0.0000 | 0.9593+/-0.0000 |
