"""
Bias-head 식별 + 캐싱 + s5 신호 동작 검증 스크립트.

- 기존 results/signals/main/{cat}_stage1.jsonl이 있어야 함 (smoke test 결과 활용).
- 식별 결과는 results/bias_heads.json에 저장됨.
- 검증: s5 신호값이 0.0이 아닌 실제 attention 값을 반환하는지.

사용:
    source venv/bin/activate
    python scripts/verify_bias_heads.py
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


def _load_stage1(signals_dir: Path, categories: list[str]) -> list[dict]:
    """카테고리별 stage1.jsonl을 모아 리스트로 반환."""
    out: list[dict] = []
    for cat in categories:
        path = signals_dir / f"{cat}_stage1.jsonl"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    rec.setdefault("category", cat)
                    out.append(rec)
    return out


def main() -> int:
    with open(ROOT / "configs" / "default.yaml") as f:
        config = yaml.safe_load(f)

    categories = config["data"]["categories"]
    signals_dir = ROOT / "results" / "signals" / "main"
    stage1 = _load_stage1(signals_dir, categories)
    if not stage1:
        print("❌ stage1 결과 없음 — 먼저 `python run_pipeline.py --stage inference --quick-test`")
        return 2
    print(f"[1] stage1 records: {len(stage1)}개")

    # BBQ items 로드 (parquet에서)
    from run_pipeline import _load_items
    items: list[dict] = []
    for cat in categories:
        items.extend(_load_items(config, cat, n_per_cat=50))
        for it in items[-50:]:
            it.setdefault("category", cat)
    items_by_id = {it["example_id"]: it for it in items}
    print(f"    BBQ items in pool: {len(items_by_id)}개")

    # 매칭되는 stage1 record만 사용
    matched = [r for r in stage1 if r["example_id"] in items_by_id]
    train_items = [items_by_id[r["example_id"]] for r in matched]
    print(f"    matched train items: {len(train_items)}개")

    # LLM 로드
    print(f"\n[2] Llama-3.1-8B 로드 (MPS, bfloat16)")
    from src.utils.llm_utils import LLMWrapper
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    llm = LLMWrapper(
        model_name=config["models"]["main"]["name"],
        dtype="bfloat16",
        device=device,
    )

    # Bias-head 식별
    print(f"\n[3] Bias-head 식별 (contrastive, max 20 samples)")
    from src.signals.bias_head import identify_bias_heads, load_bias_heads
    from src.signals.prompts import PROMPT_BUILDERS

    save_path = ROOT / "results" / "bias_heads.json"
    head_indices = identify_bias_heads(
        bbq_train_data=train_items,
        stage1_results=matched,
        llm=llm,
        prompt_builder=PROMPT_BUILDERS["vanilla"],
        primary_prompt="vanilla",
        n_top=20,
        save_path=save_path,
        max_samples=20,
    )
    print(f"\n[4] Top-20 bias heads: {head_indices}")
    assert head_indices, "❌ bias-head 식별 실패 (빈 리스트)"
    assert len(head_indices) <= 20

    # 캐시 round-trip
    loaded = load_bias_heads(save_path)
    assert loaded == head_indices, "캐시 round-trip 실패"
    print(f"   캐시 round-trip OK ({save_path})")

    # 실제 s5 추출
    print(f"\n[5] 1 instance에서 s5 추출 (식별된 head 사용)")
    from src.signals.bias_head import compute_bias_head_activation
    test_item = train_items[0]
    s5 = compute_bias_head_activation(
        item=test_item,
        llm=llm,
        prompt_builder=PROMPT_BUILDERS["vanilla"],
        head_indices=head_indices,
    )
    print(f"   s5 = {s5:.6f}")
    assert s5 > 0.0, f"❌ s5가 0이거나 음수: {s5}"
    print("\n" + "=" * 60)
    print("🎉 ALL CHECKS PASSED")
    print(f"   bias_heads.json: {save_path}")
    print(f"   s5 nonzero: {s5:.6f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
