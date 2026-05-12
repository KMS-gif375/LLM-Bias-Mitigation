"""
SAE bias-feature qualitative case study.

각 식별된 bias SAE feature에 대해 가장 강하게 활성화되는 BBQ test-split
instance를 찾아 case study 자료로 정리.

산출물:
    results/v2_runpod/qualitative/sae_features/
      ├── per_feature_activations.jsonl   (인스턴스 × 56 bias features)
      ├── top_activating_examples.json    (feature → top-5 instances)
      └── case_study.md                   (논문용 압축본)

CLI:
    python -m src.analysis.sae_qualitative --layer 15 --top-k 5 --max-instances 1332
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("sae_qualitative")


def _load_test_split(parquet_path: Path):
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    return df


def _format_user_message(row) -> tuple[str, str]:
    """vanilla prompt 구성 (system, user)."""
    system = (
        "Answer the following multiple-choice question. "
        "Respond with only the letter (A, B, or C) of the correct option."
    )
    user = (
        f"Context: {row['context']}\n\n"
        f"Question: {row['question']}\n\n"
        f"A. {row['ans0']}\n"
        f"B. {row['ans1']}\n"
        f"C. {row['ans2']}\n\n"
        f"Answer:"
    )
    return system, user


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=5, help="feature당 최대 instance 수")
    parser.add_argument("--max-instances", type=int, default=1332)
    parser.add_argument("--out-dir", type=str,
                        default="results/v2_runpod/qualitative/sae_features")
    parser.add_argument("--features-file", type=str,
                        default="results/v2_runpod/sae_layers/features_layer15.json")
    parser.add_argument("--bias-feature-limit", type=int, default=56,
                        help="처리할 bias feature 수 상한 (메모리 절약용)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1. bias features 로드 -----
    feat_data = json.loads(Path(args.features_file).read_text())
    bias_features = feat_data.get("bias_features", [])[:args.bias_feature_limit]
    if not bias_features:
        logger.error("bias_features 비어있음 — features_layer*.json 확인 필요")
        return 1
    logger.info(f"Loaded {len(bias_features)} bias features from {args.features_file}")

    # ----- 2. test split 로드 -----
    df = _load_test_split(Path("data/sampled_v2/test.parquet"))
    df = df.head(args.max_instances)
    logger.info(f"Loaded {len(df)} test instances (categories: {df['category'].nunique()})")

    # ----- 3. LLM + SAE 로드 -----
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    # .env 로드 (HF_TOKEN)
    import os
    from dotenv import load_dotenv
    load_dotenv()

    from src.utils.llm_utils import LLMWrapper
    from src.signals.sae_feature import SAEWrapper

    logger.info("Loading Llama-3.1-8B-Instruct ...")
    llm = LLMWrapper(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        device=args.device,
        dtype="bfloat16",
        hf_token=os.environ.get("HF_TOKEN"),
    )

    # SAE는 "auto" 미지원 — 실제 device로 변환
    sae_device = str(getattr(llm, "device", "cpu"))
    logger.info(f"Loading Llama-Scope SAE (layer={args.layer}, device={sae_device}) ...")
    sae = SAEWrapper(
        release="llama_scope_lxr_8x",
        sae_id=f"l{args.layer}r_8x",
        layer=args.layer,
        device=sae_device,
    )
    sae._load()

    # ----- 4. 인스턴스별 forward → SAE encode → 56 features 활성도 저장 -----
    out_jsonl = out_dir / "per_feature_activations.jsonl"
    feat_idx_to_position = {f: i for i, f in enumerate(bias_features)}

    n_done = 0
    with open(out_jsonl, "w") as f:
        for _, row in df.iterrows():
            system, user = _format_user_message(row)
            try:
                out = llm.generate(
                    user_message=user,
                    system_message=system,
                    max_new_tokens=1,
                    return_hidden_states=True,
                    hidden_layer=args.layer,
                )
                if out.hidden_states is None:
                    continue
                feats = sae.get_feature_activations(out.hidden_states)
                # bias_features 인덱스만 추출
                bias_acts = {int(fid): float(feats[fid].item()) for fid in bias_features}
            except Exception as e:
                logger.warning(f"  instance {row['example_id']} 실패: {e}")
                continue

            rec = {
                "example_id": int(row["example_id"]),
                "question_index": str(row.get("question_index", "")),
                "category": str(row["category"]),
                "context_condition": str(row["context_condition"]),
                "label": int(row["label"]),
                "context": str(row["context"]),
                "question": str(row["question"]),
                "ans0": str(row["ans0"]),
                "ans1": str(row["ans1"]),
                "ans2": str(row["ans2"]),
                "bias_feature_activations": bias_acts,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 50 == 0:
                logger.info(f"  진행 {n_done}/{len(df)}")

    logger.info(f"[저장] {out_jsonl} ({n_done} records)")

    # ----- 5. Offline: feature별 top-k 인스턴스 추출 -----
    logger.info("Computing top-activating examples per feature ...")
    records: list[dict] = []
    with open(out_jsonl) as f:
        for line in f:
            records.append(json.loads(line))

    top_examples = {}
    for fid in bias_features:
        scored = []
        for r in records:
            act = r["bias_feature_activations"].get(str(fid))
            if act is None:
                act = r["bias_feature_activations"].get(fid)
            if act is not None:
                scored.append((act, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_examples[str(fid)] = [
            {
                "activation": s[0],
                "example_id": s[1]["example_id"],
                "category": s[1]["category"],
                "condition": s[1]["context_condition"],
                "context": s[1]["context"][:300],
                "question": s[1]["question"],
                "ans0": s[1]["ans0"],
                "ans1": s[1]["ans1"],
                "ans2": s[1]["ans2"],
                "label": s[1]["label"],
            }
            for s in scored[:args.top_k]
        ]

    out_top = out_dir / "top_activating_examples.json"
    out_top.write_text(json.dumps(top_examples, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    logger.info(f"[저장] {out_top}")

    # ----- 6. case study markdown 자동 생성 -----
    # 정렬: feature별 top-1 activation 기준 내림차순
    feature_top1 = [
        (fid, top_examples[fid][0]["activation"] if top_examples[fid] else 0.0)
        for fid in [str(f) for f in bias_features]
    ]
    feature_top1.sort(key=lambda x: x[1], reverse=True)
    selected = feature_top1[:10]  # 상위 10 features

    md_lines = [
        "# SAE Bias-Feature Qualitative Case Study",
        "",
        f"**Layer**: {args.layer} (Llama-Scope `l{args.layer}r_8x`)",
        f"**Total bias features**: {len(bias_features)}",
        f"**Total test instances**: {n_done}",
        "",
        "## Top-10 Bias Features by Max Activation",
        "",
        "각 feature가 가장 강하게 활성화되는 1개 BBQ test 인스턴스를 보입니다.",
        "feature ID는 SAE의 32K hidden dim 중 하나를 가리키며, 우리가 사전에 식별한 56개 bias feature 중 일부입니다.",
        "",
    ]
    for fid, top1_act in selected:
        ex = top_examples[fid][0]
        md_lines.extend([
            f"### Feature #{fid} — top activation = {top1_act:.3f}",
            "",
            f"- **Category**: {ex['category']}",
            f"- **Condition**: {ex['condition']}",
            f"- **Context**: {ex['context']}",
            f"- **Question**: {ex['question']}",
            f"- **Options**: A. {ex['ans0']} / B. {ex['ans1']} / C. {ex['ans2']}",
            f"- **Ground truth**: {chr(65 + ex['label'])}",
            "",
        ])

    case_md = out_dir / "case_study.md"
    case_md.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info(f"[저장] {case_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
