import json
from collections import Counter

with open("data/raw/Age.jsonl") as f:
    lines = [json.loads(l) for l in f]

labels = Counter(item["label"] for item in lines)
print("=== Age 전체 정답 위치 분포 ===")
for k in sorted(labels): print(f"  label={k}: {labels[k]}  ({labels[k]/len(lines)*100:.1f}%)")

print(f"\nans2 값: {set(item['ans2'] for item in lines)}")

ambig = [item for item in lines if item["context_condition"] == "ambig"]
disambig = [item for item in lines if item["context_condition"] == "disambig"]

print(f"\n=== ambig 정답 분포 (n={len(ambig)}) ===")
for k, v in sorted(Counter(item["label"] for item in ambig).items()): print(f"  label={k}: {v}")

print(f"\n=== disambig 정답 분포 (n={len(disambig)}) ===")
for k, v in sorted(Counter(item["label"] for item in disambig).items()): print(f"  label={k}: {v}")
