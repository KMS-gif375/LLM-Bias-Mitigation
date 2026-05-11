# RunPod Migration Guide (Stage 13-22 v2)

Mac M4 Pro에서 Stage 1-12까지 완료한 v2 풀런을 RunPod GPU로 옮겨 Stage 13-22 (~5-9h)를 마칩니다.

## 사전 결과 (보존)
- Stages 1-10: signals, MoE checkpoint, evaluation (per-cond), ablation
- Stage 6 (multi-seed): 5 seeds 완료
- Stage 11 Composite v2: acc_amb=0.682, n=8864
- Stage 12 Self-Debiasing v2: acc_amb=0.958, far=0.783

## Step 1 (로컬, Mac): 아카이브 생성

```bash
bash scripts/prepare_runpod_archive.sh
```

→ `v2_runpod_YYYYMMDD_HHMMSS.tar.gz` (~50MB) 생성.

## Step 2: RunPod 인스턴스 spin up

**권장**:
- GPU: **H100 PCIe** ($1.99/h) — 메모리 80GB, 8B+SAE 여유
- Template: **PyTorch 2.4** or **CUDA 12.1** (최신)
- Disk: 50GB (Llama-3.1-8B HF cache ~16GB + 결과 + 의존성)

**대안** (비용 우선):
- A100 80GB PCIe ($1.19/h) — ~9h
- L40S 48GB ($0.86/h) — ~13h

24GB GPU (4090/3090)는 LLM + SAE peak ~20GB라 OOM 위험.

## Step 3 (RunPod): 코드 clone + 아카이브 추출

```bash
git clone https://github.com/KMS-gif375/LLM-Bias-Mitigation.git
cd LLM-Bias-Mitigation

# 로컬에서 RunPod로 아카이브 업로드 (로컬 터미널에서):
# scp v2_runpod_*.tar.gz root@<RUNPOD_IP>:~/LLM-Bias-Mitigation/

# RunPod에서 추출:
tar -xzf v2_runpod_*.tar.gz
```

## Step 4 (RunPod): HF_TOKEN 설정

```bash
echo "HF_TOKEN=hf_xxx" > .env
```

(로컬 .env가 archive에 포함되었으면 skip)

## Step 5 (RunPod): 셋업 + 실행

```bash
bash scripts/runpod_setup.sh
```

자동으로:
1. Python venv + 의존성 설치
2. CUDA 환경 검증
3. 필수 파일 체크
4. `scripts/run_v2_resume2.sh`를 nohup 백그라운드로 실행

→ PID, log 경로 출력.

## Step 6: 모니터링

```bash
tail -F logs/runpod_resume_*.log
grep "/22\]" logs/runpod_resume_*.log    # stage 진행
ps -ef | grep src.baselines              # 활성 프로세스
nvidia-smi                                # GPU 사용
```

## Step 7: 완료 후 결과 회수 (로컬에서)

```bash
rsync -avz root@<RUNPOD_IP>:~/LLM-Bias-Mitigation/results/v2/ ./results/v2_runpod/
```

또는 RunPod의 결과를 그대로 두고 다음 단계 (paper figures, 분석) 진행.

## 비용 추정

| Item | 시간 | 시급 | 합계 |
|---|---|---|---|
| Setup + 의존성 + HF cache | ~30min | $1.99 | $1 |
| Stage 13-22 실행 (H100 PCIe) | ~5h | $1.99 | $10 |
| 결과 다운로드 | ~3min | $1.99 | ~$0 |
| **총** | **~5.5h** | | **~$11** |

vs Mac M4 Pro 잔여 ~52h.

## 주의사항

- **HF_TOKEN**: Llama-3.1-8B-Instruct gated repo, 권한 필요
- **첫 실행**: HF에서 Llama 가중치 다운로드 (~16GB, ~10-20분)
- **SAE Lens**: 첫 실행 시 Llama-Scope SAE 다운로드 (~3GB, ~3-5min)
- **인터럽트 대응**: RunPod이 갑자기 종료될 수 있으니, 중간 결과는 results/v2/baselines/* 에 stage별로 저장됨 → 다시 시작하면 끝난 baseline은 skip
- **device 자동 감지**: `device: "auto"`가 CUDA 잡음, 코드 수정 불필요
