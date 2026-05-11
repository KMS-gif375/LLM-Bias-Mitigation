#!/usr/bin/env bash
# Cross-LLM (Gemma/Qwen) RunPod 마이그레이션용 최소 아카이브.
#
# 포함:
#   - data/sampled_v2/   (BBQ data, ~1.2MB)
#   - data/open_bbq/     (있으면 transfer용)
#   - .env               (HF_TOKEN — Gemma gated 필요)
#
# Cross-LLM은 자체 Stage 1-5를 실행하므로 Llama 결과 불필요.
# 코드는 GitHub clone으로 받음.
#
# macOS resource fork (._*) 자동 제외 → utf-8 corruption 방지.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="cross_llm_runpod_$(date +%Y%m%d_%H%M%S).tar.gz"

FILES=()
[[ -d "data/sampled_v2" ]] && FILES+=("data/sampled_v2")
[[ -d "data/open_bbq" ]] && FILES+=("data/open_bbq")
[[ -f ".env" ]] && FILES+=(".env")

echo "Files to archive:"
for f in "${FILES[@]}"; do
  echo "  - $f"
done
echo ""

# COPYFILE_DISABLE: macOS extended attributes 제외
# --exclude='._*': resource forks 명시적 제외
COPYFILE_DISABLE=1 tar --exclude='._*' --exclude='.DS_Store' -czf "$OUT" "${FILES[@]}"

echo ""
echo "================================================================"
echo " Cross-LLM 아카이브 생성 완료: $OUT"
echo " 크기: $(du -sh "$OUT" | cut -f1)"
echo "================================================================"
echo ""
echo "다음 단계 (RunPod 2 인스턴스 spin up 후):"
echo ""
echo "1. 각 인스턴스에 업로드:"
echo "   scp -P PORT_GEMMA -i ~/.ssh/id_ed25519 $OUT root@IP_GEMMA:~/"
echo "   scp -P PORT_QWEN  -i ~/.ssh/id_ed25519 $OUT root@IP_QWEN:~/"
echo ""
echo "2. 각 인스턴스에서:"
echo "   ssh root@IP_X -p PORT -i ~/.ssh/id_ed25519"
echo "   git clone https://github.com/KMS-gif375/LLM-Bias-Mitigation.git"
echo "   cd LLM-Bias-Mitigation"
echo "   mv ~/$OUT ."
echo "   tar -xzf $OUT"
echo "   bash scripts/runpod_cross_llm_setup.sh"
echo "   # MODEL=gemma (또는 qwen) 자동 결정. 또는 명시:"
echo "   #   MODEL=gemma bash scripts/runpod_cross_llm_setup.sh"
