#!/usr/bin/env bash
# v2 풀 런 (9 카테고리 × 1000 = 9000 instances) — 진짜 페이퍼용
#
# v2_mini로 pre-flight 통과 후 실행.
#
# 결과: results/v2/
#
# 예상 시간 (Mac M4 Pro, FULL):
#   Phase 1 (data + Stage 1·2 + bias-head):   ~85-95h  ← 가장 긴 부분
#     - Stage 1 (4-prompt × 9000)             ~7-8h
#     - Stage 2 (7-signal × 9000)             ~80-90h
#   Phase 2 (multi_seed 5 + MoE+Eval+Abl):    ~1h
#   Phase 3 (4 baselines × 1000/cat)          ~6-8h
#   Phase 4 (SAE layer × 3 × 200 samples)     ~1.5h
#   Phase 6 (3 transfer × 200/cat)            ~3-4h
#   Phase 7 (qualitative + figures)           ~30min
#   Total                                     ~95-110h
#
# 사용:
#   bash scripts/run_v2_full.sh
#   caffeinate -i bash scripts/run_v2_full.sh    # Mac sleep 방지

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ -f "venv/bin/activate" ]] && source venv/bin/activate

OUT="results/v2"
LOG="logs/v2_full_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

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
step "1/22" "Data sampling (9 cat × 1000 = 9000 instances, version=v2)"
python -m src.utils.data_loader --version v2 --sample

step "2/22" "Stage 1: 4-Prompt Inference (9000 × 4 = 36000 calls, ~7-8h)"
python run_pipeline.py --version v2 --stage inference --skip-existing

step "3/22" "Stage 2: 7-Signal Extraction (s1~s7 on 9000 items, ~80-90h)"
python run_pipeline.py --version v2 --stage signal_extraction --skip-existing

step "4/22" "Bias-head 식별 (contrastive, results/bias_heads.json 자동 생성)"
python scripts/verify_bias_heads.py 2>&1 | tail -5 || echo "  [warn] bias-head 식별 실패"

step "5/22" "SAE feature 식별은 Phase 4 SAE Layer comparison에서 통합 수행"

# ----- Phase 2: MoE + Threshold + Ablation -----
step "6/22" "Multi-seed MoE (5 seeds for robust CI)"
python -m src.analysis.multi_seed --seeds 42,123,456,789,999 --version v2 \
  --out-dir "$OUT/multi_seed"

step "7/22" "Stage 3+4 + Ablation (run_pipeline 통합)"
python run_pipeline.py --version v2 --stage moe_training evaluation ablation

step "8/22" "Threshold sensitivity sweep"
python -m src.analysis.threshold_sweep \
  --thresholds 0.3,0.4,0.5,0.6,0.7 --no-plot \
  --out-dir "$OUT/thresholds"

step "9/22" "Cluster ablation은 Stage 5에 포함됨"
ls "$OUT/ablation/main/cluster/" 2>&1 | head -3 || true

step "10/22" "LOCO ablation은 Stage 5에 포함됨"
ls "$OUT/ablation/main/loco/" 2>&1 | head -3 || true

# ----- Phase 3: Baselines (max_samples=1000/cat = 9000 each) -----
step "11/22" "Composite Prompting baseline (1000/cat)"
python -m src.baselines.composite_prompting --max-samples 1000 \
  --out-dir "$OUT/baselines/composite" --force 2>&1 | tail -3 || true

step "12/22" "Self-Debiasing baseline (1000/cat)"
python -m src.baselines.self_debiasing --max-samples 1000 \
  --out-dir "$OUT/baselines/self_debiasing" --force 2>&1 | tail -3 || true

step "13/22" "DeCAP baseline (1000/cat, faithful 3-pass)"
python -m src.baselines.decap --max-samples 1000 \
  --out-dir "$OUT/baselines/decap" --force 2>&1 | tail -3 || true

step "14/22" "FairSteer baseline (1000/cat, 2-stage CAA, fp64 norm fix)"
python -m src.baselines.fairsteer --max-samples 1000 --train-samples 300 \
  --out-dir "$OUT/baselines/fairsteer" --force 2>&1 | tail -3 || true

# ----- Phase 4: SAE Layer comparison (3 layers) -----
step "15/22" "SAE layer comparison (12, 15, 18) — 200 samples each"
python -m src.analysis.sae_layer_comparison \
  --version v2 --layers 12,15,18 --max-samples 200 \
  --out-dir "$OUT/sae_layers" 2>&1 | tail -5 || echo "  [warn] SAE layer comparison 실패"

# ----- Phase 5: Cross-LLM (옵션, 풀에서는 권장) -----
if [[ "${RUN_CROSS_LLM:-0}" == "1" ]]; then
  step "16/22" "Cross-LLM Gemma-2-9B (200 samples)"
  python -m src.cross_llm.run_cross_llm --model gemma --max-samples 200 \
    --version v2 --out-dir "$OUT/cross_llm/gemma" 2>&1 | tail -3 || true

  step "17/22" "Cross-LLM Qwen-2.5-7B (200 samples)"
  python -m src.cross_llm.run_cross_llm --model qwen --max-samples 200 \
    --version v2 --out-dir "$OUT/cross_llm/qwen" 2>&1 | tail -3 || true
else
  step "16-17/22" "Cross-LLM SKIP (RUN_CROSS_LLM=1 로 활성화)"
fi

# ----- Phase 6: Transfer (Implicit + Open-BBQ + KoBBQ, 200/cat) -----
step "18/22" "ImplicitBBQ-style 자체 생성 (200/cat) + 평가"
python -m src.data.generate_implicit_bbq --version v2 --max-samples 200 \
  --out-dir data/implicit_bbq_v2 2>&1 | tail -3 || true
python -m src.transfer.run_implicit_bbq \
  --data-dir data/implicit_bbq_v2 --max-samples 200 \
  --out-dir "$OUT/transfer/implicit_bbq" --force 2>&1 | tail -5 || true

step "19/22" "Open-BBQ 변환 + 평가 (200/cat, 9 main + 2 intersectional)"
if [[ ! -d "data/open_bbq" ]]; then
  python -m src.data.prepare_open_bbq --auto 2>&1 | tail -3 || true
fi
python -m src.transfer.run_open_bbq --max-samples 200 \
  --out-dir "$OUT/transfer/open_bbq" --force 2>&1 | tail -3 || true

step "20/22" "KoBBQ cross-lingual (200/cat)"
python -m src.transfer.run_kobbq --max-samples 200 \
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
