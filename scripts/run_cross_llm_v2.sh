#!/usr/bin/env bash
# Cross-LLM 풀 파이프라인 (Gemma-2-9B-It 또는 Qwen-2.5-7B-Instruct).
#
# 사용:
#   MODEL=gemma bash scripts/run_cross_llm_v2.sh
#   MODEL=qwen  bash scripts/run_cross_llm_v2.sh
#
# 환경변수:
#   MODEL: gemma | qwen  (필수)
#   VERSION: v2 | smoke | mini  (기본 v2)
#   TRANSFER_MAX: transfer 평가 cap (기본 300)
#
# 예상 시간 (H100):
#   Gemma (9B): Stage 1 ~75min + Stage 2 ~85min + 3-5 ~10min = ~3h
#   Qwen (7B):  Stage 1 ~60min + Stage 2 ~70min + 3-5 ~10min = ~2.5h
#
# 결과: results/v2/cross_llm/{MODEL}/...

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ -f "venv/bin/activate" ]] && source venv/bin/activate

MODEL="${MODEL:?MODEL env var 필수 (gemma 또는 qwen)}"
VERSION="${VERSION:-v2}"
TRANSFER_MAX="${TRANSFER_MAX:-300}"

if [[ "$MODEL" != "gemma" && "$MODEL" != "qwen" && "$MODEL" != "mistral" ]]; then
  echo "ERROR: MODEL must be 'gemma', 'qwen', or 'mistral', got '$MODEL'"
  exit 2
fi

OUT_BASE=$(python3 -c "
mapping = {'v2': 'results/v2', 'smoke': 'results/smoke_e2e', 'mini': 'results/v2_mini'}
print(mapping['$VERSION'])
")
OUT="$OUT_BASE/cross_llm/$MODEL"
LOG="logs/cross_llm_${MODEL}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")" "$OUT"

step() {
  echo ""
  echo "================================================================"
  echo " [$1] $2"
  echo " ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "================================================================"
}

step "INIT" "Cross-LLM ${MODEL} ${VERSION} 시작"
echo " model        : $MODEL"
echo " version      : $VERSION"
echo " out_dir      : $OUT"
echo " transfer_max : $TRANSFER_MAX"

# ----- Phase 1: Data + Stage 1 + Stage 2 + Bias-head ID -----
step "1/9" "Data sampling 확인 (공유)"
ls data/sampled_v2/ 2>&1 | head -3

step "2/9" "Stage 1: 4-Prompt Inference (${MODEL})"
python run_pipeline.py --version "$VERSION" --model "$MODEL" --stage inference

step "3/9" "Stage 2: 7-Signal Extraction (${MODEL}, bias-head auto-identify)"
python run_pipeline.py --version "$VERSION" --model "$MODEL" --stage signal_extraction

# ----- Phase 2: MoE 학습 + per-condition τ + 평가 + ablation -----
step "4/9" "Multi-seed MoE (3 seeds, ${MODEL})"
python -m src.analysis.multi_seed --seeds 42,123,456 --version "$VERSION" --model "$MODEL" \
  --out-dir "$OUT/multi_seed"

step "5/9" "Stage 3+4 + Ablation (${MODEL})"
python run_pipeline.py --version "$VERSION" --model "$MODEL" --stage moe_training evaluation ablation

step "6/9" "Threshold sensitivity sweep"
python -m src.analysis.threshold_sweep \
  --thresholds 0.3,0.4,0.5,0.6,0.7 --no-plot \
  --out-dir "$OUT/thresholds" --model "$MODEL" 2>&1 | tail -5 || true

# ----- Phase 3: Transfer (Open-BBQ + KoBBQ) -----
step "7/9" "Transfer: Open-BBQ ($TRANSFER_MAX/cat)"
if [[ ! -d "data/open_bbq" ]]; then
  python -m src.data.prepare_open_bbq --auto 2>&1 | tail -3 || true
fi
python -m src.transfer.run_open_bbq --max-samples "$TRANSFER_MAX" \
  --out-dir "$OUT/transfer/open_bbq" --force --model "$MODEL" 2>&1 | tail -3 || true

step "8/9" "Transfer: KoBBQ ($TRANSFER_MAX/cat)"
python -m src.transfer.run_kobbq --max-samples "$TRANSFER_MAX" \
  --out-dir "$OUT/transfer/kobbq" --force --model "$MODEL" 2>&1 | tail -3 || true

# ----- Phase 4: Comparison + Verify -----
step "9/9" "결과 요약 + verify"
python3 << EOF
import json
from pathlib import Path
print()
print("="*72)
print(f" Cross-LLM ${MODEL} 결과 요약")
print("="*72)
for src, label in [
    ("$OUT/evaluation/main/final.json", "Main eval (per-cond τ)"),
    ("$OUT/multi_seed/summary.json", "Multi-seed 3 seeds"),
]:
    p = Path(src)
    if p.exists():
        try:
            d = json.load(open(p))
            if "metrics_per_condition" in d:
                m = d["metrics_per_condition"]
                tau = d.get("thresholds_per_condition", {})
                print(f"\n[{label}] τ_amb={tau.get('ambig')} τ_dis={tau.get('disambig')}")
                print(f"  acc_amb={m.get('accuracy_amb'):.4f} acc_dis={m.get('accuracy_dis'):.4f}")
                print(f"  far={m.get('false_abstention_rate'):.4f} bias_amb={m.get('bias_score_amb')}")
            elif "aggregate" in d:
                print(f"\n[{label}]")
                for k in ["accuracy_amb", "accuracy_dis", "false_abstention_rate"]:
                    v = d["aggregate"].get(k, {})
                    if v.get("n", 0) > 0:
                        print(f"  {k:25s}: {v['mean']:.4f} ± {v.get('std', 0):.4f}")
        except Exception as e:
            print(f"  ERR: {e}")
    else:
        print(f"\n[{label}] FILE MISSING: $src")
EOF

step "DONE" "End: $(date)"
echo " 결과: $OUT"
echo " 로그: $LOG"
