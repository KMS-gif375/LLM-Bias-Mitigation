#!/usr/bin/env bash
# v2 풀런 (9 카테고리 × 1000 = 9000 instances)
#
# 페이퍼 메인 실험. 모든 stage full sample.
# 결과: results/v2/
#
# 예상 시간 (Mac M4 Pro, mini 기준 10x 외삽 + 일부 cap):
#   Stage 1 (4-prompt × 9000) : ~13h
#   Stage 2 (signal × 9000)   : ~17h
#   Multi-seed (3)            : ~5min
#   MoE+Eval+Abl              : ~5min
#   Threshold sweep           : ~30s
#   Baselines × 4 (1000/cat)  : ~42h
#     - Composite: 16h
#     - Self-Debiasing: 4h
#     - DeCAP: 16.5h
#     - FairSteer: 5.7h
#   SAE × 3 layers (100 samp) : ~15min (mini와 동일 cap)
#   Transfer × 3 (300/cat cap): ~30h (full 1000/cat은 75h)
#     - ImplicitBBQ: 14h (gen+eval)
#     - Open-BBQ: 8h
#     - KoBBQ: 8h
#   Total                     : ~100-110h (4-5일)
#
# Transfer는 300/cat로 cap (300×9=2700 인스턴스, 통계적으로 충분).
# 확장: 환경변수 TRANSFER_MAX=1000 으로 늘릴 수 있음.
#
# 사용:
#   bash scripts/run_v2.sh
#   caffeinate -i bash scripts/run_v2.sh    # Mac sleep 방지 (강력 권장)
#   nohup caffeinate -i bash scripts/run_v2.sh > logs/v2_run.log 2>&1 &
#

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ -f "venv/bin/activate" ]] && source venv/bin/activate

OUT="results/v2"
LOG="logs/v2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

TRANSFER_MAX="${TRANSFER_MAX:-300}"

step() {
  echo ""
  echo "================================================================"
  echo " [$1] $2"
  echo " ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "================================================================"
}

step "INIT" "Start: $(date)"
echo " out_dir      : $OUT"
echo " log          : $LOG"
echo " transfer_max : $TRANSFER_MAX/cat"

# ----- Phase 1: Data + Signal Extraction -----
step "1/22" "Data sampling (9×1000=9000 instances, version=v2)"
python -m src.utils.data_loader --version v2 --sample

step "2/22" "Stage 1: 4-Prompt Inference (9000 × 4 = 36000 calls, ~13h)"
python run_pipeline.py --version v2 --stage inference

step "3/22" "Stage 2: 7-Signal Extraction (s1~s7, ~17h)"
python run_pipeline.py --version v2 --stage signal_extraction

step "4/22" "Bias-head 식별 (contrastive)"
python scripts/verify_bias_heads.py 2>&1 | tail -5 || echo "  [warn] bias-head 식별 실패"

step "5/22" "SAE feature 식별은 SAE Layer comparison에서 통합 수행"

# ----- Phase 2: MoE + Threshold + Ablation -----
step "6/22" "Multi-seed MoE (5 seeds, v2 version)"
python -m src.analysis.multi_seed --seeds 42,123,456,789,999 --version v2 \
  --out-dir "$OUT/multi_seed"

step "7/22" "Stage 3+4 + Ablation (run_pipeline 통합, per-condition threshold 포함)"
python run_pipeline.py --version v2 --stage moe_training evaluation ablation

step "8/22" "Threshold sensitivity sweep"
python -m src.analysis.threshold_sweep \
  --thresholds 0.3,0.4,0.5,0.6,0.7 --no-plot \
  --out-dir "$OUT/thresholds"

step "9/22" "Cluster ablation은 Stage 5에 포함됨"
ls "$OUT/ablation/main/cluster/" 2>&1 | head -3 || true

step "10/22" "LOCO ablation은 Stage 5에 포함됨"
ls "$OUT/ablation/main/loco/" 2>&1 | head -3 || true

# ----- Phase 3: Baselines (1000/cat full = 9000 each) -----
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

# ----- Phase 4: SAE Layer comparison (3 layers × 100 samples cap) -----
step "15/22" "SAE layer comparison (12, 15, 18) — 100 samples cap"
python -m src.analysis.sae_layer_comparison \
  --version v2 --layers 12,15,18 --max-samples 100 \
  --out-dir "$OUT/sae_layers" 2>&1 | tail -5 || echo "  [warn] SAE layer comparison 실패"

# ----- Phase 5: Cross-LLM (옵션, default skip ~30GB DL) -----
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

# ----- Phase 6: Transfer (capped at TRANSFER_MAX/cat) -----
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
step "21/22" "Qualitative analysis (bias_heads + risk_coverage)"
python -m src.analysis.qualitative \
  --tasks bias_heads_heatmap risk_coverage \
  --out-dir "$OUT/qualitative" 2>&1 | tail -3 || true

step "22/22" "Paper figures (1, 3, 4, 5)"
python -m src.paper.figures --figs 1 3 4 5 \
  --out-dir "$OUT/figures" 2>&1 | tail -3 || true

# ----- 최종 검증 -----
step "VERIFY" "결과 파일 자동 검증"
python scripts/verify_smoke_e2e.py --out-dir "$OUT" || true

step "DONE" "End: $(date)"
echo " 결과: $OUT"
echo " 로그: $LOG"
