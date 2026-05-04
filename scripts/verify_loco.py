"""
LOCO (Leave-One-Category-Out) 평가 검증 스크립트.

기존 quick-test 결과(stage1.jsonl, signals.jsonl, embeddings.pt)를 활용하여
LOCO 7-fold를 다시 돌리고 held_acc != 0 인지 확인.

사용:
    source venv/bin/activate
    python scripts/verify_loco.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import yaml
import torch


def main() -> int:
    with open(ROOT / "configs" / "default.yaml") as f:
        config = yaml.safe_load(f)
    categories = config["data"]["categories"]
    signals_dir = ROOT / "results" / "signals" / "main"

    # 1. signal records + embeddings 모으기
    print("[1] records + embeddings 로드")
    records: list[dict] = []
    embeddings: dict = {}
    for cat in categories:
        sig_path = signals_dir / f"{cat}_signals.jsonl"
        emb_path = signals_dir / f"{cat}_embeddings.pt"
        if not sig_path.exists() or not emb_path.exists():
            print(f"   [skip] {cat}: 신호/임베딩 파일 없음")
            continue
        with open(sig_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    rec.setdefault("category", cat)
                    records.append(rec)
        embeddings.update(torch.load(emb_path, map_location="cpu", weights_only=True))
    print(f"   records={len(records)}  embeddings={len(embeddings)}")
    if not records:
        print("❌ stage1/signals 데이터 없음 — smoke test 먼저")
        return 2

    # 2. instances_by_id (parquet 자동 파싱)
    from run_pipeline import _load_items
    instances_by_id: dict = {}
    for cat in categories:
        for it in _load_items(config, cat, n_per_cat=300):
            it.setdefault("category", cat)
            instances_by_id[it["example_id"]] = it
    print(f"   BBQ instances pool: {len(instances_by_id)}")

    # 3. SignalsDataset.example_id 보존 검증
    from src.models.trainer import SignalsDataset, TrainConfig
    ds = SignalsDataset(records, embeddings)
    sample = ds[0]
    assert "example_id" in sample, "❌ SignalsDataset이 example_id를 저장하지 않음"
    print(f"\n[2] SignalsDataset example_id 보존 OK: {sample['example_id']}")

    # 4. LOCO 실행
    print("\n[3] LOCO 7-fold 실행")
    from src.ablation.loco_ablation import run_loco_ablation
    train_config = TrainConfig(
        epochs=2, batch_size=8, lr=1e-3, val_every=1,
        device="auto", seed=42, save_dir=None,
    )
    summary = run_loco_ablation(
        all_records=records,
        embeddings=embeddings,
        instances_by_id=instances_by_id,
        categories=tuple(categories),
        embed_dim=next(iter(embeddings.values())).shape[-1],
        train_config=train_config,
        threshold=0.5,
        save_dir=str(ROOT / "results" / "ablation" / "main" / "loco_verify"),
    )

    # 5. 결과 검증
    print("\n[4] Fold별 held_acc:")
    nonzero_count = 0
    for cat, fold in summary.per_fold.items():
        zero = (fold.held_out_acc_amb == 0.0 and fold.held_out_acc_dis == 0.0)
        marker = "❌" if zero else "✅"
        if not zero:
            nonzero_count += 1
        print(f"   {marker} {cat:25s}  acc_amb={fold.held_out_acc_amb:.4f}  "
              f"acc_dis={fold.held_out_acc_dis:.4f}  n={fold.n_held_out}")

    agg = summary.aggregate()
    print(f"\n[5] Aggregate (7-fold mean):")
    for k, v in agg.items():
        print(f"   {k}: {v:.4f}")

    if nonzero_count == 0:
        print("\n❌ 모든 fold가 0.0 — example_id 매칭이 여전히 실패")
        return 1

    print(f"\n🎉 {nonzero_count}/{len(summary.per_fold)} fold가 nonzero")
    print(f"   결과: results/ablation/main/loco_verify/loco_ablation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
