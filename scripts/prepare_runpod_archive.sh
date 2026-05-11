#!/usr/bin/env bash
# RunPod 마이그레이션용 아카이브 생성.
#
# 포함:
#   - data/sampled_v2/         (parquet split, 1.2MB)
#   - results/v2/signals/      (signals + embeddings, 23MB)
#   - results/v2/moe/          (MoE checkpoint, 1.7MB)
#   - results/v2/evaluation/   (Stage 4 결과)
#   - results/v2/ablation/     (Stage 5 결과)
#   - results/v2/multi_seed/   (5 seeds, 8.8MB)
#   - results/v2/baselines/{composite,self_debiasing}/  (이미 완료된 baseline)
#   - results/bias_heads.json
#   - data/open_bbq/           (있으면)
#
# 출력: v2_runpod_$(date).tar.gz (~50MB)
#
# 사용:
#   bash scripts/prepare_runpod_archive.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="v2_runpod_$(date +%Y%m%d_%H%M%S).tar.gz"

# 포함할 파일들 (존재하는 것만)
FILES=()
[[ -d "data/sampled_v2" ]] && FILES+=("data/sampled_v2")
[[ -d "results/v2/signals" ]] && FILES+=("results/v2/signals")
[[ -d "results/v2/moe" ]] && FILES+=("results/v2/moe")
[[ -d "results/v2/evaluation" ]] && FILES+=("results/v2/evaluation")
[[ -d "results/v2/ablation" ]] && FILES+=("results/v2/ablation")
[[ -d "results/v2/multi_seed" ]] && FILES+=("results/v2/multi_seed")
[[ -d "results/v2/baselines/composite" ]] && FILES+=("results/v2/baselines/composite")
[[ -d "results/v2/baselines/self_debiasing" ]] && FILES+=("results/v2/baselines/self_debiasing")
[[ -f "results/bias_heads.json" ]] && FILES+=("results/bias_heads.json")
[[ -d "data/open_bbq" ]] && FILES+=("data/open_bbq")
[[ -f ".env" ]] && FILES+=(".env")

echo "Files to archive:"
for f in "${FILES[@]}"; do
  echo "  - $f"
done
echo ""

tar -czf "$OUT" "${FILES[@]}"

echo ""
echo "================================================================"
echo " 아카이브 생성 완료: $OUT"
echo " 크기: $(du -sh "$OUT" | cut -f1)"
echo "================================================================"
echo ""
echo "다음 단계:"
echo "  1. RunPod 인스턴스 SSH IP 확인 (대시보드)"
echo "  2. scp $OUT root@<IP>:~/"
echo "  3. RunPod에서: tar -xzf $OUT && bash scripts/runpod_setup.sh"
