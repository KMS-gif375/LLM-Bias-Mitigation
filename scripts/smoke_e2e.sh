#!/usr/bin/env bash
# End-to-end smoke test for all major modules.
# Exercises the full pipeline at minimal scale (~10-15 min total on Mac M4 Pro).
#
# Modules covered:
#   1. Data sampling (v1 + v2 verify)
#   2. Multi-seed MoE (uses existing v1 signals)
#   3. SAE layer comparison (1 layer × small sample)
#   4. Threshold sensitivity (existing data)
#   5. Qualitative analysis (bias_heads + risk_coverage)
#   6. Paper figures (1, 3, 4, 5)
#   7. ImplicitBBQ-style: generate (3/cat) + evaluate
#   8. Open-BBQ: convert + evaluate (3/cat)
#   9. KoBBQ: HF download + evaluate (3/cat)
#
# Sequential to avoid GPU contention. Each module logs to stdout.
# Exit on first error (set -e).

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Activate venv
if [[ -f "venv/bin/activate" ]]; then
  source venv/bin/activate
fi

echo "================================================================"
echo " END-TO-END SMOKE TEST (~10-15 min)"
echo " Started: $(date)"
echo "================================================================"

SMOKE_OUT="results/_smoke_e2e"
rm -rf "$SMOKE_OUT" "data/implicit_smoke_e2e"
mkdir -p "$SMOKE_OUT"

# ----- 1. Data sampling (verify v1 + v2 exist) -----
echo ""
echo "[1/9] Data sampling verification"
ls -lh data/sampled/ data/sampled_v2/ 2>&1 | head -10

# ----- 2. Multi-seed MoE (uses v1 signals) -----
echo ""
echo "[2/9] Multi-seed MoE (2 seeds, v1)"
python -m src.analysis.multi_seed --seeds 42,123 --version v1 \
  --out-dir "$SMOKE_OUT/multi_seed" 2>&1 | tail -5

# ----- 3. SAE layer comparison (1 layer × 20 inst) -----
echo ""
echo "[3/9] SAE layer comparison (layer 15, 20 samples)"
python -m src.analysis.sae_layer_comparison --version v1 --layers 15 \
  --max-samples 20 --out-dir "$SMOKE_OUT/sae_layers" 2>&1 | tail -5

# ----- 4. Threshold sensitivity sweep -----
echo ""
echo "[4/9] Threshold sensitivity (existing v1 data)"
python -m src.analysis.threshold_sweep --no-plot \
  --thresholds 0.3,0.5,0.7 \
  --out-dir "$SMOKE_OUT/thresholds" 2>&1 | tail -5

# ----- 5. Qualitative analysis -----
echo ""
echo "[5/9] Qualitative analysis"
python -m src.analysis.qualitative \
  --tasks bias_heads_heatmap risk_coverage \
  --out-dir "$SMOKE_OUT/qualitative" 2>&1 | tail -5

# ----- 6. Paper figures -----
echo ""
echo "[6/9] Paper figures (1, 3, 4, 5)"
python -m src.paper.figures --figs 1 3 4 5 \
  --out-dir "$SMOKE_OUT/figures" 2>&1 | tail -5

# ----- 7. ImplicitBBQ-style: generate + evaluate -----
echo ""
echo "[7/9] ImplicitBBQ-style: generate (3/cat) + evaluate"
python -m src.data.generate_implicit_bbq --version v1 --max-samples 3 \
  --out-dir data/implicit_smoke_e2e 2>&1 | tail -3
python -m src.transfer.run_implicit_bbq \
  --data-dir data/implicit_smoke_e2e --max-samples 3 \
  --out-dir "$SMOKE_OUT/transfer/implicit" 2>&1 | tail -5

# ----- 8. Open-BBQ: convert + evaluate -----
echo ""
echo "[8/9] Open-BBQ: convert + evaluate (3/cat)"
if [[ ! -d "data/open_bbq" ]]; then
  python -m src.data.prepare_open_bbq --auto 2>&1 | tail -3
fi
python -m src.transfer.run_open_bbq --max-samples 3 \
  --categories Age Gender_identity Race_ethnicity \
  --out-dir "$SMOKE_OUT/transfer/open_bbq" 2>&1 | tail -5

# ----- 9. KoBBQ: HF download + evaluate -----
echo ""
echo "[9/9] KoBBQ: cross-lingual transfer (3/cat)"
python -m src.transfer.run_kobbq --max-samples 3 \
  --categories Age Gender_identity Race_ethnicity \
  --out-dir "$SMOKE_OUT/transfer/kobbq" 2>&1 | tail -5

# ----- Summary -----
echo ""
echo "================================================================"
echo " SMOKE TEST COMPLETED: $(date)"
echo "================================================================"
echo ""
echo "Outputs under $SMOKE_OUT:"
find "$SMOKE_OUT" -type f \( -name "*.json" -o -name "*.csv" -o -name "*.pdf" \) | head -25
echo ""
echo "OK"
