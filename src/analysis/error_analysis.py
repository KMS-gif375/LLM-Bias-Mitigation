"""
Error analysis — Ours (MoE + per-cond τ) 의 failure case 분석.

Test split 인스턴스 (또는 전체) 에 대해:
    1. signals JSONL 로드
    2. question embedding 생성 (sentence-transformers)
    3. MoE checkpoint forward → p_score
    4. per-condition τ (amb=0.95, dis=0.05) override
    5. ground truth 와 비교 → 4 type 으로 분류
        A. ambig + Ours = stereotype  → bias-slip
        B. ambig + Ours = anti-stereo  → over-correction (반대 방향)
        C. disambig + Ours = Unknown   → over-abstention
        D. disambig + Ours = wrong specific → wrong-keep
    6. type 별 5 sample → 총 ~20 case
    7. README markdown 자동 생성

CLI:
    python -m src.analysis.error_analysis --max-cases 20
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("error_analysis")


def _load_signals(signals_dir: Path) -> list[dict]:
    """results/v2_runpod/signals/main/*_signals.jsonl 전부 로드."""
    records = []
    for f in sorted(signals_dir.glob("*_signals.jsonl")):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _load_bbq_items(split: str = "all") -> dict:
    """
    BBQ items lookup.

    Args:
        split: "all" (test+train) | "test" (test only, leak-free for MoE).
    """
    import pandas as pd
    items = {}
    parquets = ["data/sampled_v2/test.parquet"]
    if split == "all":
        parquets.append("data/sampled_v2/train.parquet")
    for parquet in parquets:
        p = Path(parquet)
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        for _, row in df.iterrows():
            key = f"{row['category']}::{row['example_id']}"
            items[key] = row.to_dict()
    return items


def _find_unknown_idx(item: dict) -> int:
    """ans0/1/2 중 'Unknown' 류 옵션 index. BBQ 모든 표현 커버."""
    keywords = [
        "unknown", "cannot", "not enough", "can't", "undetermined",
        "not be determined", "not known", "not answerable",
        "no answer", "no enough", "unclear",
    ]
    for i, k in enumerate(["ans0", "ans1", "ans2"]):
        opt = (item.get(k) or "").lower().strip()
        if any(kw in opt for kw in keywords):
            return i
    return -1


def _is_stereotype_answer(item: dict, pred: int) -> bool:
    """예측이 stereotyped_groups 에 해당하는지."""
    try:
        meta = item.get("additional_metadata")
        if isinstance(meta, str):
            meta = json.loads(meta)
        stereo_groups = meta.get("stereotyped_groups", []) if meta else []
        ans_info = item.get("answer_info")
        if isinstance(ans_info, str):
            ans_info = json.loads(ans_info)
        pred_ans_key = f"ans{pred}"
        ans_meta = ans_info.get(pred_ans_key, []) if ans_info else []
        if isinstance(ans_meta, list) and ans_meta:
            # ans_meta[0] 은 표면형, ans_meta[1] 은 정규형 — stereotype 그룹과 일치 검사
            for tag in ans_meta:
                if any(sg.lower() in str(tag).lower() for sg in stereo_groups):
                    return True
        return False
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals-dir", type=str,
                        default="results/v2_runpod/signals/main")
    parser.add_argument("--moe-ckpt", type=str,
                        default="results/v2_runpod/moe/main/moe_best.pt")
    parser.add_argument("--tau-amb", type=float, default=0.95)
    parser.add_argument("--tau-dis", type=float, default=0.05)
    parser.add_argument("--max-cases", type=int, default=20,
                        help="총 출력 case 수 (type 별 균등 분배)")
    parser.add_argument("--out-dir", type=str,
                        default="results/v2_runpod/qualitative/error_analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "all"],
                        help="'test' = test parquet 만 (leak-free, default); 'all' = train+test")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    from dotenv import load_dotenv
    load_dotenv()

    # ----- 1. signals 로드 -----
    signals = _load_signals(Path(args.signals_dir))
    logger.info(f"Loaded {len(signals)} signal records")

    # ----- 2. BBQ items 로드 (context/question/answers) -----
    items_by_key = _load_bbq_items(split=args.split)
    logger.info(f"Loaded {len(items_by_key)} BBQ items (split={args.split})")

    # ----- 3. MoE checkpoint 로드 -----
    saved = torch.load(args.moe_ckpt, map_location="cpu", weights_only=True)
    cfg = saved.get("model_config", {})
    from src.models.moe_aggregator import MoEAggregator
    model = MoEAggregator(
        signal_dim=int(cfg.get("signal_dim", 7)),
        embed_dim=int(cfg.get("embed_dim", 384)),
        num_experts=int(cfg.get("num_experts", 4)),
        gating_hidden=int(cfg.get("gating_hidden", 64)),
        expert_hidden=int(cfg.get("expert_hidden", 128)),
    )
    model.load_state_dict(saved.get("model_state_dict", saved), strict=False)
    model.eval()
    logger.info(f"Loaded MoE: embed_dim={cfg.get('embed_dim')}, experts={cfg.get('num_experts')}")

    # ----- 4. question embedding (sentence-transformers) — CPU 강제 (MoE도 CPU) -----
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    logger.info("Loaded sentence-transformer all-MiniLM-L6-v2 on CPU")

    # ----- 5. 각 signal record forward → 분류 -----
    from src.models.override import apply_per_condition_override

    failures = {"A_bias_slip": [], "B_over_correct": [], "C_over_abstain": [], "D_wrong_keep": []}
    total_correct = 0
    total_evaluated = 0

    for sig in signals:
        key = f"{sig['category']}::{sig['example_id']}"
        item = items_by_key.get(key)
        if item is None:
            continue
        # signals → tensor
        s = sig.get("signals", {})
        sig_vec = torch.tensor([
            s.get("s1_evidence", 0.0),
            s.get("s2_counterfactual", 0.0),
            s.get("s3_confidence", 0.0),
            s.get("s4_consistency", 0.0),
            s.get("s5_bias_head", 0.0),
            s.get("s6_prompt_sensitivity", 0.0),
            s.get("s7_sae_feature", 0.0),
        ], dtype=torch.float32).unsqueeze(0)
        # question embedding
        text = f"{item.get('context','')} {item.get('question','')}"
        emb = embed_model.encode(text, convert_to_tensor=True).unsqueeze(0).float()

        with torch.no_grad():
            out = model(sig_vec, emb)
        # MoEOutput.p — (batch,) sigmoid confidence
        p_score = float(out.p.item())

        # per-cond override
        primary = sig.get("primary_answer", -1)
        result = apply_per_condition_override(
            primary_answer=int(primary),
            p_score=p_score,
            item={"context_condition": sig.get("context_condition", ""), **item},
            thresholds={"ambig": args.tau_amb, "disambig": args.tau_dis},
        )
        final = result["final_answer"]

        gold = int(sig.get("label", -1))
        total_evaluated += 1
        if final == gold:
            total_correct += 1
            continue

        # 분류
        unknown_idx = _find_unknown_idx(item)
        cond = sig.get("context_condition", "")

        case = {
            "example_id": sig["example_id"],
            "category": sig["category"],
            "condition": cond,
            "context": str(item.get("context", ""))[:400],
            "question": str(item.get("question", "")),
            "ans0": str(item.get("ans0", "")),
            "ans1": str(item.get("ans1", "")),
            "ans2": str(item.get("ans2", "")),
            "gold": gold,
            "primary_pred": int(primary),
            "final_pred": int(final),
            "p_score": round(p_score, 4),
            "tau_used": result.get("threshold_used"),
            "signals": s,
        }

        if cond == "ambig":
            # gold should be unknown_idx
            if final == unknown_idx:
                continue  # actually correct (rare edge)
            # final is NOT unknown but should be
            if _is_stereotype_answer(item, final):
                failures["A_bias_slip"].append(case)
            else:
                failures["B_over_correct"].append(case)
        elif cond == "disambig":
            # gold is specific answer
            if final == unknown_idx:
                failures["C_over_abstain"].append(case)
            else:
                failures["D_wrong_keep"].append(case)

    n_eval = total_evaluated
    n_corr = total_correct
    n_failures = sum(len(v) for v in failures.values())
    logger.info(f"Eval: {n_corr}/{n_eval} correct ({n_corr/n_eval*100:.2f}%)")
    logger.info(f"Failures by type: A_bias_slip={len(failures['A_bias_slip'])}, "
                f"B_over_correct={len(failures['B_over_correct'])}, "
                f"C_over_abstain={len(failures['C_over_abstain'])}, "
                f"D_wrong_keep={len(failures['D_wrong_keep'])}")

    # ----- 6. type 별 sample -----
    per_type = max(1, args.max_cases // 4)
    sampled = {}
    for k, v in failures.items():
        random.shuffle(v)
        sampled[k] = v[:per_type]

    # 저장
    out_json = out_dir / "failure_cases.json"
    out_json.write_text(json.dumps({
        "tau_amb": args.tau_amb,
        "tau_dis": args.tau_dis,
        "total_evaluated": n_eval,
        "total_correct": n_corr,
        "accuracy": n_corr / n_eval if n_eval else 0,
        "failure_counts": {k: len(v) for k, v in failures.items()},
        "sampled_cases": sampled,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info(f"[저장] {out_json}")

    # ----- 7. markdown 자동 생성 -----
    md = [
        "# Error Analysis — Failure Cases of Ours (MoE + per-cond τ)",
        "",
        f"**Split**: {args.split} ({'test parquet only — leak-free' if args.split == 'test' else 'train+test combined — includes train (MoE 가 본 데이터)'})",
        f"**Total evaluated**: {n_eval} instances",
        f"**Correct**: {n_corr} ({n_corr/n_eval*100:.2f}%)",
        f"**Thresholds used**: τ_amb={args.tau_amb}, τ_dis={args.tau_dis}",
        "",
        "## Failure type counts",
        "",
        "| Type | Description | Count |",
        "|---|---|---|",
        f"| **A. Bias-slip** | ambig + Ours chose stereotype | {len(failures['A_bias_slip'])} |",
        f"| **B. Over-correction** | ambig + Ours chose anti-stereo | {len(failures['B_over_correct'])} |",
        f"| **C. Over-abstention** | disambig + Ours chose Unknown | {len(failures['C_over_abstain'])} |",
        f"| **D. Wrong-keep** | disambig + Ours chose wrong specific | {len(failures['D_wrong_keep'])} |",
        "",
        f"**Sampled per type**: up to {per_type} representative cases",
        "",
    ]
    type_labels = {
        "A_bias_slip": "A. Bias-slip — ambig context + Ours produced stereotypical answer",
        "B_over_correct": "B. Over-correction — ambig context + Ours produced anti-stereotypical answer (less concerning, but still error)",
        "C_over_abstain": "C. Over-abstention — disambig context + Ours abstained to Unknown",
        "D_wrong_keep": "D. Wrong-keep — disambig context + Ours kept wrong specific answer",
    }
    for k, label in type_labels.items():
        cases = sampled.get(k, [])
        if not cases:
            continue
        md.append(f"## {label}")
        md.append("")
        for i, c in enumerate(cases, 1):
            md.extend([
                f"### Case {k[0]}{i} — {c['category']} (example_id={c['example_id']})",
                "",
                f"- **Condition**: {c['condition']}",
                f"- **Context**: {c['context']}",
                f"- **Question**: {c['question']}",
                f"- **Options**: A. {c['ans0']} / B. {c['ans1']} / C. {c['ans2']}",
                f"- **Gold**: {chr(65+c['gold'])} ({[c['ans0'], c['ans1'], c['ans2']][c['gold']]})",
                f"- **Primary (LLM raw)**: {chr(65+c['primary_pred']) if c['primary_pred']>=0 else '-'}",
                f"- **Final (after τ)**: {chr(65+c['final_pred']) if c['final_pred']>=0 else '-'}",
                f"- **p_score**: {c['p_score']:.3f} (τ used = {c['tau_used']})",
                f"- **Signals**: s1={c['signals']['s1_evidence']:.2f} s2={c['signals']['s2_counterfactual']:.2f} "
                f"s3={c['signals']['s3_confidence']:.2f} s4={c['signals']['s4_consistency']:.2f} "
                f"s5={c['signals']['s5_bias_head']:.2f} s6={c['signals']['s6_prompt_sensitivity']:.2f} "
                f"s7={c['signals']['s7_sae_feature']:.2f}",
                "",
            ])

    out_md = out_dir / "failure_cases.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    logger.info(f"[저장] {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
