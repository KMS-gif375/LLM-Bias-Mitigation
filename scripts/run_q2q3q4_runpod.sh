#!/usr/bin/env bash
# Q2 + Q3 (Winogender + StereoSet) + Q4 RunPod 통합 실행.
#
# 사전 조건:
#   - RunPod H100 SXM 80GB pod
#   - 코드 clone 완료
#   - .env 에 HF_TOKEN 있음
#   - Llama-3.1-8B-Instruct license 승인됨
#   - results/v2_runpod/ 동기화 완료 (signals, moe checkpoint)
#
# 실행:
#   bash scripts/run_q2q3q4_runpod.sh
#
# 예상 시간: 5-8h on H100
# 예상 비용: $15-25 ($3/h)

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ -f "venv/bin/activate" ]] && source venv/bin/activate

LOG_DIR="logs/q2q3q4_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

step() {
  echo ""
  echo "================================================================"
  echo " [$1] $2"
  echo " ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "================================================================"
}

# ------------------------------------------------------------------
# 1. Audit (실험 시작 전 누설 검증)
# ------------------------------------------------------------------
step "PRE-AUDIT" "Q2/Q3/Q4 코드 + split 누설 검증"
python -m src.utils.audit_q2q3q4 2>&1 | tee "$LOG_DIR/audit_pre.log"

# ------------------------------------------------------------------
# 2. Q2 — Minimal-core MoE ablation (가장 빠름, 30분)
# ------------------------------------------------------------------
step "Q2" "Minimal-core MoE ablation (5 seeds × 6 variants)"
python -m src.analysis.minimal_core_ablation \
    --signals-dir results/v2_runpod/signals/main \
    --seeds 42,123,456,789,999 \
    --epochs 20 --batch-size 64 \
    --out-dir results/v2_runpod/qualitative/minimal_core \
    2>&1 | tee "$LOG_DIR/q2_minimal_core.log"

# ------------------------------------------------------------------
# 3. Q3a — Winogender transfer (~1.5h)
# ------------------------------------------------------------------
step "Q3a" "Winogender zero-shot transfer (n=720)"
python -m src.transfer.run_winogender \
    --model main \
    --tau-amb 0.95 --tau-dis 0.05 \
    --out-dir results/v2_runpod/transfer/winogender \
    --force 2>&1 | tee "$LOG_DIR/q3a_winogender.log"

# ------------------------------------------------------------------
# 4. Q3b — StereoSet transfer (~3-4h)
# ------------------------------------------------------------------
step "Q3b" "StereoSet zero-shot transfer (n=2123, validation split)"
python -m src.transfer.run_stereoset \
    --model main \
    --tau-amb 0.95 --tau-dis 0.05 \
    --out-dir results/v2_runpod/transfer/stereoset \
    --force 2>&1 | tee "$LOG_DIR/q3b_stereoset.log"

# ------------------------------------------------------------------
# 5. Q4 — Self-Debiasing-Reprompting (이미 결과 있으면 skip)
# ------------------------------------------------------------------
step "Q4" "Self-Debiasing-Reprompting (Gallegos NAACL 2025) — 이미 완료된 경우 skip"
if [[ -f "results/v2_runpod/baselines/self_debiasing/final.json" ]]; then
    echo "✅ 이미 완료: results/v2_runpod/baselines/self_debiasing/final.json"
    echo "재실행하려면 --force"
else
    python -m src.baselines.self_debiasing \
        --eval --output-dir results/v2_runpod/baselines/self_debiasing \
        2>&1 | tee "$LOG_DIR/q4_self_debias.log"
fi

# ------------------------------------------------------------------
# 6. POST-AUDIT — 결과 무결성 검증
# ------------------------------------------------------------------
step "POST-AUDIT" "결과 파일 존재 + 일관성 검증"
python -m src.utils.audit_q2q3q4 2>&1 | tee "$LOG_DIR/audit_post.log"

# ------------------------------------------------------------------
# 7. 결과 요약
# ------------------------------------------------------------------
step "SUMMARY" "Q2/Q3/Q4 결과 요약"
python3 << 'PYEOF'
import json
from pathlib import Path

print("\n" + "="*72)
print(" Q2/Q3/Q4 결과 요약")
print("="*72)

# Q2
p = Path("results/v2_runpod/qualitative/minimal_core/results.json")
if p.exists():
    d = json.load(open(p))
    print("\n[Q2] Minimal-core ablation:")
    print(f"{'Variant':25s} | {'n_sig':5s} | {'val_loss':18s} | {'test_loss':18s}")
    print("-"*78)
    for vname, a in d.get("aggregate", {}).items():
        print(f"{vname:25s} | {a['n_signals_kept']:5d} | "
              f"{a['val_loss_mean']:.4f} ± {a['val_loss_std']:.4f} | "
              f"{a['test_loss_mean']:.4f} ± {a['test_loss_std']:.4f}")

# Q3a
p = Path("results/v2_runpod/transfer/winogender/overall_metrics.json")
if p.exists():
    d = json.load(open(p))
    o = d.get("overall", d)
    print(f"\n[Q3a] Winogender:")
    print(f"  n_total={o.get('n_total')} (ambig={o.get('n_ambig')}, dis={o.get('n_disambig')})")
    print(f"  acc_amb={o.get('accuracy_amb',0):.4f} acc_dis={o.get('accuracy_dis',0):.4f}")
    print(f"  far={o.get('false_abstention_rate',0):.4f}")

# Q3b
p = Path("results/v2_runpod/transfer/stereoset/overall_metrics.json")
if p.exists():
    d = json.load(open(p))
    o = d.get("overall", d)
    print(f"\n[Q3b] StereoSet (intrasentence):")
    print(f"  n={o.get('n_total')}, acc_unknown={o.get('accuracy_amb',0):.4f}")
    print(f"  bias_amb={o.get('bias_score_amb',0):+.4f}")
    print(f"  StereoSet LMS={o.get('stereoset_lms',0):.4f} SS={o.get('stereoset_ss',0):.4f} iCAT={o.get('stereoset_icat',0):.4f}")

# Q4
p = Path("results/v2_runpod/baselines/self_debiasing/final.json")
if p.exists():
    d = json.load(open(p))
    o = d.get("overall", d)
    print(f"\n[Q4] Self-Debiasing-Reprompting (Gallegos NAACL 2025):")
    print(f"  n={o.get('n_total')}, acc_amb={o.get('accuracy_amb',0):.4f}, "
          f"acc_dis={o.get('accuracy_dis',0):.4f}")
    print(f"  bias_amb={o.get('bias_score_amb',0):+.4f}, far={o.get('false_abstention_rate',0):.4f}")

print("\n" + "="*72)
print(f" 결과 로그: {Path('$LOG_DIR').resolve() if False else 'logs/q2q3q4_*'}")
print("="*72)
PYEOF

echo ""
echo "✅ Q2/Q3/Q4 RunPod 실행 완료. 결과 회수 후 README 업데이트."
