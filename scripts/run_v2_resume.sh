#!/usr/bin/env bash
# v2 풀런 Stage 11-22 resume 스크립트.
#
# Stage 1-10은 이미 results/v2/에 완료 (signals, MoE, eval, multi-seed 5 seeds, ablation).
# Stage 11-14 baselines는 v1 default로 잘못 돌고 있어 kill됨 → 여기서 v2로 재실행.
#
# 진입 시각: 2026-05-08 04:25 KST 경
# 예상 시간: ~70h (baselines 42h + sae 15min + transfer 30h + figs 5min)
# ETA: 2026-05-11 02:00 KST 경
#
# 사용:
#   nohup caffeinate -i bash scripts/run_v2_resume.sh > logs/v2_resume.log 2>&1 &

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ -f "venv/bin/activate" ]] && source venv/bin/activate

OUT="results/v2"
LOG="logs/v2_resume_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

TRANSFER_MAX="${TRANSFER_MAX:-300}"

step() {
  echo ""
  echo "================================================================"
  echo " [$1] $2"
  echo " ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "================================================================"
}

step "INIT" "Resume start: $(date)"
echo " out_dir      : $OUT"
echo " transfer_max : $TRANSFER_MAX/cat"
echo " skip stages  : 1-10 (already completed in results/v2/)"

# ----- Phase 3: Baselines (1000/cat full = 9000 each, --version v2) -----
step "11/22" "Composite Prompting baseline (full 1000/cat, ~16h)"
python -m src.baselines.composite_prompting --version v2 --max-samples 1000 \
  --out-dir "$OUT/baselines/composite" --force 2>&1 | tail -3 || true

step "12/22" "Self-Debiasing baseline (full 1000/cat, ~4h)"
python -m src.baselines.self_debiasing --version v2 --max-samples 1000 \
  --out-dir "$OUT/baselines/self_debiasing" --force 2>&1 | tail -3 || true

step "13/22" "DeCAP baseline (full 1000/cat, faithful 3-pass, ~16.5h)"
python -m src.baselines.decap --version v2 --max-samples 1000 \
  --out-dir "$OUT/baselines/decap" --force 2>&1 | tail -3 || true

step "14/22" "FairSteer baseline (full 1000/cat, 2-stage CAA, ~5.7h)"
python -m src.baselines.fairsteer --version v2 --max-samples 1000 --train-samples 300 \
  --out-dir "$OUT/baselines/fairsteer" --force 2>&1 | tail -3 || true

# ----- Phase 4: SAE Layer comparison -----
step "15/22" "SAE layer comparison (12, 15, 18) — 100 samples cap"
python -m src.analysis.sae_layer_comparison \
  --version v2 --layers 12,15,18 --max-samples 100 \
  --out-dir "$OUT/sae_layers" 2>&1 | tail -5 || echo "  [warn] SAE layer comparison 실패"

# ----- Phase 5: Cross-LLM (옵션) -----
if [[ "${RUN_CROSS_LLM:-0}" == "1" ]]; then
  step "16/22" "Cross-LLM Gemma-2-9B (300 samples)"
  python -m src.cross_llm.run_cross_llm --model gemma --max-samples 300 \
    --version v2 --out-dir "$OUT/cross_llm/gemma" 2>&1 | tail -3 || true

  step "17/22" "Cross-LLM Qwen-2.5-7B (300 samples)"
  python -m src.cross_llm.run_cross_llm --model qwen --max-samples 300 \
    --version v2 --out-dir "$OUT/cross_llm/qwen" 2>&1 | tail -3 || true
else
  step "16-17/22" "Cross-LLM SKIP (RUN_CROSS_LLM=1 로 활성화)"
fi

# ----- Phase 6: Transfer -----
step "18/22" "ImplicitBBQ-style 자체 생성 ($TRANSFER_MAX/cat) + 평가"
python -m src.data.generate_implicit_bbq --version v2 --max-samples "$TRANSFER_MAX" \
  --out-dir "data/implicit_bbq_v2_${TRANSFER_MAX}" 2>&1 | tail -3 || true
python -m src.transfer.run_implicit_bbq \
  --data-dir "data/implicit_bbq_v2_${TRANSFER_MAX}" --max-samples "$TRANSFER_MAX" \
  --out-dir "$OUT/transfer/implicit_bbq" --force 2>&1 | tail -5 || true

step "19/22" "Open-BBQ 변환 + 평가 ($TRANSFER_MAX/cat)"
if [[ ! -d "data/open_bbq" ]]; then
  python -m src.data.prepare_open_bbq --auto 2>&1 | tail -3 || true
fi
python -m src.transfer.run_open_bbq --max-samples "$TRANSFER_MAX" \
  --out-dir "$OUT/transfer/open_bbq" --force 2>&1 | tail -3 || true

step "20/22" "KoBBQ cross-lingual ($TRANSFER_MAX/cat)"
python -m src.transfer.run_kobbq --max-samples "$TRANSFER_MAX" \
  --out-dir "$OUT/transfer/kobbq" --force 2>&1 | tail -3 || true

# ----- Phase 7: Statistics + Figures -----
step "21/22" "Qualitative analysis"
python -m src.analysis.qualitative \
  --tasks bias_heads_heatmap risk_coverage \
  --out-dir "$OUT/qualitative" 2>&1 | tail -3 || true

step "22/22" "Paper figures (1, 3, 4, 5)"
python -m src.paper.figures --figs 1 3 4 5 \
  --out-dir "$OUT/figures" 2>&1 | tail -3 || true

# ----- Verify -----
step "VERIFY" "결과 파일 자동 검증"
python scripts/verify_smoke_e2e.py --out-dir "$OUT" || true

step "DONE" "End: $(date)"
echo " 결과: $OUT"
