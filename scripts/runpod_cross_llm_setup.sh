#!/usr/bin/env bash
# RunPod Cross-LLM 자동 셋업 + 실행.
#
# 사용:
#   MODEL=gemma bash scripts/runpod_cross_llm_setup.sh
#   MODEL=qwen  bash scripts/runpod_cross_llm_setup.sh
#
# MODEL 미설정 시 hostname/IP 기반 추론 (안 되면 에러).
#
# 사전 조건:
#   - 코드 clone 완료
#   - cross_llm_runpod_*.tar.gz 추출 완료 (data/sampled_v2 + .env)
#   - HF_TOKEN이 .env에 있고 Gemma-2-9b-it license 승인됨

set -e

MODEL="${MODEL:-}"
if [[ -z "$MODEL" ]]; then
  echo "ERROR: MODEL env var 미설정. MODEL=gemma 또는 MODEL=qwen 로 실행."
  exit 2
fi

if [[ "$MODEL" != "gemma" && "$MODEL" != "qwen" && "$MODEL" != "mistral" ]]; then
  echo "ERROR: MODEL must be 'gemma', 'qwen', or 'mistral', got '$MODEL'"
  exit 2
fi

echo "================================================================"
echo " RunPod Cross-LLM 셋업 + 실행"
echo " MODEL: $MODEL"
echo " $(date)"
echo "================================================================"

# 1. Python venv
if [[ ! -d "venv" ]]; then
  echo ""
  echo "[1/5] Python venv 생성"
  python3 -m venv venv
fi
source venv/bin/activate

# 2. 의존성 설치
echo ""
echo "[2/5] 의존성 설치"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt 2>&1 | tail -5

# 3. CUDA + 디스크 확인
echo ""
echo "[3/5] 환경 검증"
python3 -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU Memory: {mem:.1f} GB')
    if mem < 24:
        print(f'  ⚠ GPU Memory 24GB 미만 — $MODEL 로드 시 OOM 가능')
"
echo ""
echo "  Disk usage:"
df -h /root | tail -1
free_gb=$(df -BG /root | tail -1 | awk '{print $4}' | tr -d 'G')
if [[ "$free_gb" -lt 30 ]]; then
  echo "  ⚠ 디스크 여유 30GB 미만 — Gemma+Qwen 다운로드 ~33GB 필요"
fi

# 4. HF 토큰 확인
echo ""
echo "[4/5] HuggingFace 토큰 확인"
if [[ ! -f ".env" ]]; then
  echo "  ❌ .env 없음. HF_TOKEN 필요."
  exit 1
fi
source .env
if [[ -z "$HF_TOKEN" ]]; then
  echo "  ❌ HF_TOKEN env var empty"
  exit 1
fi
echo "  HF_TOKEN: ${HF_TOKEN:0:10}..."

# 데이터 확인
if [[ ! -d "data/sampled_v2" ]]; then
  echo "  ❌ data/sampled_v2 없음. 아카이브 추출 확인."
  exit 1
fi
echo "  data/sampled_v2: $(ls data/sampled_v2/ | wc -l) files"

# 5. Cross-LLM 파이프라인 실행
echo ""
echo "[5/5] Cross-LLM 파이프라인 시작 ($MODEL)"
echo ""

LOG="logs/runpod_cross_llm_${MODEL}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
nohup bash -c "MODEL=$MODEL bash scripts/run_cross_llm_v2.sh" > "$LOG" 2>&1 &
PID=$!

echo "백그라운드 실행 시작:"
echo "  PID  : $PID"
echo "  Log  : $LOG"
echo ""
echo "진행 모니터링:"
echo "  tail -F $LOG"
echo "  grep '/9\\]' $LOG    # stage 진행"
echo "  nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv"
echo ""
echo "예상 시간:"
echo "  Gemma: ~3-3.5h on H100"
echo "  Qwen:  ~2.5-3h on H100"
echo ""
echo "완료 후 결과 회수 (로컬에서):"
echo "  scp -P PORT -i KEY -r root@IP:~/LLM-Bias-Mitigation/results/v2/cross_llm/$MODEL/ ./results/v2/cross_llm/"
