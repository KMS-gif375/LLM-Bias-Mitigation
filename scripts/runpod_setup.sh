#!/usr/bin/env bash
# RunPod에서 환경 셋업 + Stage 13-22 실행.
#
# 사전 준비 (RunPod 인스턴스에서):
#   1. 인스턴스 부팅 (H100 PCIe + PyTorch 2.4 image 권장)
#   2. SSH 접속
#   3. 코드 clone:
#        git clone https://github.com/KMS-gif375/LLM-Bias-Mitigation.git
#        cd LLM-Bias-Mitigation
#   4. 아카이브 업로드 (로컬에서):
#        scp v2_runpod_*.tar.gz root@<RUNPOD_IP>:~/LLM-Bias-Mitigation/
#   5. 아카이브 추출:
#        tar -xzf v2_runpod_*.tar.gz
#   6. HF_TOKEN 설정 (.env 없으면):
#        echo "HF_TOKEN=hf_xxx" > .env
#
# 실행:
#   bash scripts/runpod_setup.sh
#
# RunPod CUDA 환경 자동 감지 → device="auto"가 cuda 잡음.

set -e

echo "================================================================"
echo " RunPod 환경 셋업"
echo " $(date)"
echo "================================================================"

# 1. Python venv
if [[ ! -d "venv" ]]; then
  echo ""
  echo "[1/4] Python venv 생성"
  python3 -m venv venv
fi
source venv/bin/activate

# 2. CUDA PyTorch + 의존성 (이미 PyTorch image면 빠름)
echo ""
echo "[2/4] 의존성 설치"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt 2>&1 | tail -5

# 3. CUDA 확인
echo ""
echo "[3/4] CUDA / GPU 확인"
python3 -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# 4. 필수 파일 확인
echo ""
echo "[4/4] 필수 파일 확인"
for f in data/sampled_v2 results/v2/signals results/v2/moe results/bias_heads.json; do
  if [[ -e "$f" ]]; then
    echo "  ✓ $f"
  else
    echo "  ✗ $f 누락 — 아카이브 추출 확인 필요"
    exit 1
  fi
done

if [[ ! -f ".env" ]]; then
  echo ""
  echo "  ⚠ .env 없음. HF_TOKEN 설정 필요:"
  echo "     echo 'HF_TOKEN=hf_xxx' > .env"
  exit 1
fi

echo ""
echo "================================================================"
echo " 셋업 완료. Stage 13-22 실행 시작 (~5-9h on H100)"
echo "================================================================"
echo ""

# Stage 13-22 실행 (nohup으로 백그라운드 + 로그)
LOG="logs/runpod_resume_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
nohup bash scripts/run_v2_resume2.sh > "$LOG" 2>&1 &
PID=$!

echo "백그라운드 실행 시작:"
echo "  PID  : $PID"
echo "  Log  : $LOG"
echo ""
echo "진행 모니터링:"
echo "  tail -F $LOG"
echo "  grep '/22\\]' $LOG"
echo ""
echo "완료 후 결과 다운로드 (로컬에서):"
echo "  rsync -avz root@<RUNPOD_IP>:~/LLM-Bias-Mitigation/results/v2/ ./results/v2_runpod/"
