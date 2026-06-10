#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# token_signal_runpod.sh  —  run the per-token signal extraction on a CUDA box.
#
# This is the ONLY GPU step for the mechanistic qualitative figures. It is
# INFERENCE (not training): a couple of forward passes on Llama-3.1-8B-Instruct
# to dump per-token s5 attention + s7 SAE activations (+ s1/s2) to small JSONs.
# Rendering the figures is done locally with scripts/token_signal_plot.py (no GPU).
#
# PREREQS on the RunPod box (cwd = repo root):
#   - repo present  (git clone https://github.com/KMS-gif375/LLM-Bias-Mitigation.git)
#     PLUS scripts/token_signal_extract.py + token_signal_plot.py copied in
#     (they are new — scp them or include in the archive).
#   - .env with HF_TOKEN=hf_...   (gated Llama-3.1-8B-Instruct)
#   - artifacts (~10MB, from the v2_runpod archive / scp):
#       results/bias_heads.json
#       results/v2_runpod/sae_layers/features_layer15.json
#       results/v2_runpod/moe/main/moe_best.pt
#       results/v2_runpod/signals/main/*_signals.jsonl
#       data/sampled_v2/test.parquet
#
# USAGE:
#   bash scripts/token_signal_runpod.sh            # run the 2 default examples
#   SETUP=1 bash scripts/token_signal_runpod.sh    # also create venv + pip install
#   PAIRS="312:Nationality 3751:Race_ethnicity 225:Age" bash scripts/token_signal_runpod.sh
#
# After it finishes, fetch the JSONs to your Mac and plot:
#   rsync -avz root@<RUNPOD_IP>:~/LLM-Bias-Mitigation/results/v2_runpod/qualitative/token_signals/ \
#        ./results/v2_runpod/qualitative/token_signals/
#   ./venv/bin/python scripts/token_signal_plot.py \
#        --json results/v2_runpod/qualitative/token_signals/nationality_312.json \
#        --out  results/v2_runpod/qualitative/token_signals/nationality_312.pdf
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-./venv/bin/python}"
OUT="results/v2_runpod/qualitative/token_signals"
PAIRS="${PAIRS:-312:Nationality 3751:Race_ethnicity}"
PROMPT_STYLE="${PROMPT_STYLE:-vanilla}"

# -- optional environment setup --------------------------------------------
if [[ "${SETUP:-0}" == "1" ]]; then
  echo "[setup] venv + deps"
  [[ -d venv ]] || python3 -m venv venv
  ./venv/bin/pip install -q --upgrade pip
  ./venv/bin/pip install -q -r requirements.txt
  # extras the extract/plot need (no-op if already in requirements.txt)
  ./venv/bin/pip install -q sae_lens sentence-transformers python-dotenv matplotlib
fi

echo "[gpu] check"
$PY -c "import torch;print('  cuda:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "[files] check artifacts"
for f in results/bias_heads.json \
         results/v2_runpod/sae_layers/features_layer15.json \
         results/v2_runpod/moe/main/moe_best.pt \
         results/v2_runpod/signals/main \
         data/sampled_v2/test.parquet \
         scripts/token_signal_extract.py; do
  [[ -e "$f" ]] && echo "  ✓ $f" || { echo "  ✗ MISSING: $f"; exit 1; }
done

mkdir -p "$OUT"
echo "[run] extracting per-token signals for: $PAIRS  (prompt-style=$PROMPT_STYLE)"
for pair in $PAIRS; do
  eid="${pair%%:*}"; cat="${pair##*:}"
  name="$(echo "$cat" | tr '[:upper:]' '[:lower:]')_${eid}"
  echo "  -> $cat::$eid  ($name.json)"
  $PY scripts/token_signal_extract.py \
      --example-id "$eid" --category "$cat" --prompt-style "$PROMPT_STYLE" \
      --out "$OUT/$name.json"
done

echo ""
echo "[done] JSONs written to: $OUT"
ls -la "$OUT"/*.json
echo ""
echo "Fetch them to your Mac, then render with scripts/token_signal_plot.py (no GPU)."
