"""
SAE 로드 + 1 BBQ instance에서 feature activation 추출 검증 스크립트.

사용:
    source venv/bin/activate
    python scripts/verify_sae.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 import path에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import yaml
import torch


def main() -> int:
    # 1. Config 로드
    with open(ROOT / "configs" / "default.yaml") as f:
        config = yaml.safe_load(f)
    sae_cfg = config["sae"]["llama"]
    print(f"[config] release={sae_cfg['release']}  sae_id={sae_cfg['sae_id']}  layer={sae_cfg['layer']}")

    # 2. SAE 로드 — sae_lens 직접 호출 (제일 단순한 검증)
    print("\n[1] SAE.from_pretrained 직접 호출")
    from sae_lens import SAE
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"   device: {device}")
    sae, _, _ = SAE.from_pretrained(
        release=sae_cfg["release"],
        sae_id=sae_cfg["sae_id"],
        device=device,
    )
    print("✅ SAE 로드 성공!")
    print(f"   feature dim (d_sae): {sae.cfg.d_sae}")
    print(f"   input dim (d_in)   : {sae.cfg.d_in}")
    print(f"   dtype              : {sae.cfg.dtype}")

    # 3. SAEWrapper 경유로도 로드 검증
    print("\n[2] SAEWrapper 경유 로드")
    from src.signals.sae_feature import SAEWrapper
    wrapper = SAEWrapper(
        release=sae_cfg["release"],
        sae_id=sae_cfg["sae_id"],
        layer=int(sae_cfg["layer"]),
        device=device,
    )
    wrapper._load()
    print("✅ SAEWrapper 로드 성공")

    # 4. 1 BBQ instance에서 feature activation 추출
    print("\n[3] 1 BBQ instance feature activation 추출")
    from src.utils.llm_utils import LLMWrapper
    from src.signals.sae_feature import compute_sae_signal
    from src.signals.prompts import PROMPT_BUILDERS
    from run_pipeline import _load_items  # parquet 자동 파싱 활용

    print("   Llama-3.1-8B 로드 중 (캐시되어 있으면 빠름)...")
    llm = LLMWrapper(
        model_name=config["models"]["main"]["name"],
        dtype=config["models"]["main"].get("dtype", "bfloat16"),
        device=device,
    )

    items = _load_items(config, "Age", n_per_cat=1)
    if not items:
        print("❌ BBQ Age 카테고리 instance 없음 — data 준비 먼저 (python -m src.utils.data_loader --all)")
        return 2
    item = items[0]
    item.setdefault("category", "Age")
    print(f"   instance example_id={item.get('example_id')} category={item.get('category')}")
    print(f"   context (preview)  : {item.get('context', '')[:80]}...")

    # SAEWrapper의 hidden state hook 검증
    score = compute_sae_signal(
        item=item,
        llm=llm,
        sae=wrapper,
        prompt_builder=PROMPT_BUILDERS["vanilla"],
        bias_feature_indices=[],   # 빈 리스트 → top-k 평균 사용
        top_k=int(sae_cfg.get("feature_top_k", 50)),
    )
    print(f"\n✅ s7 SAE feature activation = {score}")
    assert score is not None, "SAE feature score is None"
    assert isinstance(score, float), f"Expected float, got {type(score).__name__}"
    print("   shape/dtype 정상 (단일 float)")

    print("\n" + "=" * 60)
    print("🎉 ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
