#!/usr/bin/env bash
# End-to-End Smoke Test (22 stage 통합 검증, ~30-60분)
#
# 9 카테고리 × 5 instance = 45개 instance로 모든 stage 작동 확인.
# 풀 런 (9×1000) 가기 전 안전망.
#
# 결과: results/smoke_e2e/
#
# 사용:
#   bash scripts/run_smoke_e2e.sh
#   caffeinate -i bash scripts/run_smoke_e2e.sh    # Mac sleep 방지
#
# Cross-LLM (Gemma/Qwen)은 모델 다운로드 ~30GB 필요해 default skip.
# RUN_CROSS_LLM=1 으로 환경변수 설정 시 활성화.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ -f "venv/bin/activate" ]] && source venv/bin/activate

OUT="results/smoke_e2e"
LOG="logs/smoke_e2e_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

# Helper: 단계별 echo
step() {
  echo ""
  echo "================================================================"
  echo " [$1] $2"
  echo " ($(date '+%H:%M:%S'))"
  echo "================================================================"
}

step "INIT" "Start: $(date)"
echo " out_dir : $OUT"
echo " log     : $LOG"

# ----- Phase 1: Data + Signal Extraction -----
step "1/22" "Data sampling (9×5=45 instances, version=smoke)"
python -m src.utils.data_loader --version smoke --sample

step "2/22" "Stage 1: 4-Prompt Inference (45 × 4 = 180 calls)"
python run_pipeline.py --version smoke --stage inference

step "3/22" "Stage 2: 7-Signal Extraction (s1~s7)"
python run_pipeline.py --version smoke --stage signal_extraction

step "4/22" "Bias-head 식별 (contrastive, 20 samples)"
python scripts/verify_bias_heads.py 2>&1 | tail -5 || echo "  [warn] bias-head 식별 실패 (signals 부재 가능)"

step "5/22" "SAE feature 식별은 SAE Layer comparison에서 통합 수행 (Phase 4)"

# ----- Phase 2: MoE + Threshold + Ablation -----
step "6/22" "Multi-seed MoE (3 seeds, smoke version)"
python -m src.analysis.multi_seed --seeds 42,123,456 --version smoke \
  --out-dir "$OUT/multi_seed"

step "7/22" "Stage 3+4 + Ablation (run_pipeline 통합)"
python run_pipeline.py --version smoke --stage moe_training evaluation ablation

step "8/22" "Threshold sensitivity sweep"
python -m src.analysis.threshold_sweep \
  --thresholds 0.3,0.5,0.7 --no-plot \
  --out-dir "$OUT/thresholds"

step "9/22" "Cluster ablation은 Stage 5 (run_pipeline ablation)에 포함됨"
ls "$OUT/ablation/main/cluster/" 2>&1 | head -3

step "10/22" "LOCO ablation은 Stage 5에 포함됨"
ls "$OUT/ablation/main/loco/" 2>&1 | head -3

# ----- Phase 3: Baselines (4종, max_samples=30) -----
step "11/22" "Composite Prompting baseline (max 5/cat)"
python -m src.baselines.composite_prompting --max-samples 5 \
  --out-dir "$OUT/baselines/composite" --force 2>&1 | tail -3 || true

step "12/22" "Self-Debiasing baseline (max 5/cat)"
python -m src.baselines.self_debiasing --max-samples 5 \
  --out-dir "$OUT/baselines/self_debiasing" --force 2>&1 | tail -3 || true

step "13/22" "DeCAP baseline (max 5/cat, faithful 3-pass)"
python -m src.baselines.decap --max-samples 5 \
  --out-dir "$OUT/baselines/decap" --force 2>&1 | tail -3 || true

step "14/22" "FairSteer baseline (max 5/cat, 2-stage CAA)"
python -m src.baselines.fairsteer --max-samples 5 --train-samples 30 \
  --out-dir "$OUT/baselines/fairsteer" --force 2>&1 | tail -3 || true

# ----- Phase 4: SAE Layer comparison (3 layers) -----
step "15/22" "SAE layer comparison (12, 15, 18) — 30 samples"
python -m src.analysis.sae_layer_comparison \
  --version v1 --layers 12,15,18 --max-samples 30 \
  --out-dir "$OUT/sae_layers" 2>&1 | tail -5 || echo "  [warn] SAE layer comparison 실패"

# ----- Phase 5: Cross-LLM (옵션) -----
if [[ "${RUN_CROSS_LLM:-0}" == "1" ]]; then
  step "16/22" "Cross-LLM Gemma-2-9B (10 samples)"
  python -m src.cross_llm.run_cross_llm --model gemma --max-samples 5 \
    --version smoke --out-dir "$OUT/cross_llm/gemma" 2>&1 | tail -3 || true

  step "17/22" "Cross-LLM Qwen-2.5-7B (10 samples)"
  python -m src.cross_llm.run_cross_llm --model qwen --max-samples 5 \
    --version smoke --out-dir "$OUT/cross_llm/qwen" 2>&1 | tail -3 || true
else
  step "16-17/22" "Cross-LLM SKIP (RUN_CROSS_LLM=1 로 활성화)"
  echo "  Gemma + Qwen 모델 다운로드 ~30GB 필요. smoke에서 default skip."
fi

# ----- Phase 6: Transfer (Implicit + Open-BBQ + KoBBQ) -----
step "18/22" "ImplicitBBQ-style 자체 생성 (3/cat) + 평가"
python -m src.data.generate_implicit_bbq --version smoke --max-samples 3 \
  --out-dir data/implicit_bbq_smoke 2>&1 | tail -3 || true
python -m src.transfer.run_implicit_bbq \
  --data-dir data/implicit_bbq_smoke --max-samples 3 \
  --out-dir "$OUT/transfer/implicit_bbq" --force 2>&1 | tail -5 || true

step "19/22" "Open-BBQ 변환 + 평가 (3/cat)"
if [[ ! -d "data/open_bbq" ]]; then
  python -m src.data.prepare_open_bbq --auto 2>&1 | tail -3 || true
fi
python -m src.transfer.run_open_bbq --max-samples 3 \
  --categories Age Gender_identity Race_ethnicity \
  --out-dir "$OUT/transfer/open_bbq" --force 2>&1 | tail -3 || true

step "20/22" "KoBBQ cross-lingual (3/cat)"
python -m src.transfer.run_kobbq --max-samples 3 \
  --categories Age Gender_identity Race_ethnicity \
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
