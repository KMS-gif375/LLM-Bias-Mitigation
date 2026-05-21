"""
Signal s7: SAE Feature Activation

Sparse Autoencoder (SAE)의 특정 feature가 demographic 정보 처리에
관여하는지 측정합니다. 사전에 식별된 "bias-related SAE feature"의
activation 강도를 계산합니다.

지원 SAE:
    - Llama-Scope (Llama-3.1용, Fudan University): fnlp/Llama3_1-8B-*
    - Gemma Scope (Gemma-2용, DeepMind): google/gemma-scope-9b-it-res

Qwen은 SAE가 없으므로 이 신호는 None을 반환합니다 (6-signal version).

NOTE: 본 모듈은 골격이며, sae_lens 라이브러리 통합은 Mac에서 직접
      검증한 후 완성하는 것이 권장됩니다.
"""

from typing import Optional  # SAE 미지원 모델 대응

import torch  # SAE forward pass

from src.utils.llm_utils import LLMWrapper


class SAEWrapper:
    """
    SAE 로딩 및 feature activation 계산 wrapper.

    SAELens v6+ API에서는 (release, sae_id) 페어를 사용하며, release는 사전
    등록된 이름(예: "llama_scope_lxr_8x"), sae_id는 그 release 안의 layer별
    식별자(예: "l15r_8x")이다.

    sae_lens 라이브러리가 설치되어 있어야 합니다.
    """

    def __init__(
        self,
        release: str,    # SAELens release 식별자
        sae_id: str,     # release 내 SAE 식별자 (layer마다 다름)
        layer: int,      # hook 대상 layer 인덱스
        device: str = "auto",
    ) -> None:
        """
        Args:
            release: SAELens 사전 등록 release 이름 (예: "llama_scope_lxr_8x").
            sae_id: release 안의 SAE 식별자 (예: "l15r_8x").
            layer: hook 시 사용할 layer 인덱스 (sae_id와 일관 유지).
            device: device 문자열.
        """
        # 단순 metadata 저장 (실제 로드는 lazy)
        self.release = release
        self.sae_id = sae_id
        self.layer = layer
        self.device = device
        self._sae = None  # lazy load — 메모리 절약 (SAE는 ~수 GB)

    def _load(self) -> None:
        """SAE를 lazy하게 로드합니다."""
        if self._sae is not None:
            return  # 이미 로드된 경우 skip
        try:
            from sae_lens import SAE  # optional dependency
        except ImportError as e:
            raise RuntimeError(
                "sae_lens 미설치. `pip install sae-lens`"
            ) from e

        # release + sae_id 페어로 사전학습 SAE 다운로드 (HF cache 활용)
        result = SAE.from_pretrained(
            release=self.release,
            sae_id=self.sae_id,
            device=self.device,
        )
        # SAELens 버전에 따라 단일 SAE 객체 또는 (sae, cfg, sparsity) 튜플 반환
        # → 버전 호환성을 위해 두 경우 모두 처리
        self._sae = result[0] if isinstance(result, tuple) else result
        self._sae.eval()  # inference 모드 (BatchNorm/Dropout 비활성화)

    @torch.inference_mode()
    def get_feature_activations(
        self,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden state를 SAE에 통과시켜 feature activation을 얻습니다.

        Args:
            hidden_state: (batch, seq, d_model) 또는 (d_model,) 텐서.

        Returns:
            (n_features,) feature activation 벡터.
        """
        self._load()
        # 입력 차원 정규화 → 마지막 토큰의 hidden state만 사용
        if hidden_state.dim() == 3:
            # (batch=1, seq, d) → 마지막 토큰만
            hidden_state = hidden_state[0, -1, :]
        elif hidden_state.dim() == 2:
            # (seq, d) → 마지막 토큰만
            hidden_state = hidden_state[-1, :]

        with torch.no_grad():
            # SAE.encode: hidden → sparse feature activation (n_features=32,768 for Llama-Scope)
            # unsqueeze(0): (d,) → (1, d) for batch dim
            features = self._sae.encode(hidden_state.unsqueeze(0))
        # batch dim 제거 → (n_features,)
        return features.squeeze(0)


def compute_sae_signal(
    item: dict,
    llm: LLMWrapper,
    sae: Optional[SAEWrapper],
    prompt_builder,
    bias_feature_indices: list[int],
    top_k: int = 50,        # bias feature 미식별 시 fallback의 k
) -> Optional[float]:
    """
    SAE bias-related feature activation을 측정합니다.

    Args:
        item: BBQ instance.
        llm: LLMWrapper.
        sae: SAEWrapper (None이면 Qwen 등 SAE 미지원 모델 -> None 반환).
        prompt_builder: prompt 빌더.
        bias_feature_indices: 사전 식별된 bias 관련 feature 인덱스.
        top_k: 상위 k개 활성 feature를 고려.

    Returns:
        평균 activation 값 (0~) 또는 SAE 미지원 시 None.
    """
    # SAE 미지원 (Qwen, Mistral 등) → None 반환 (MoE에서 6-signal로 처리)
    if sae is None:
        return None

    # Prompt 생성 (s5와 동일 builder 사용)
    system_msg, user_msg = prompt_builder(item)

    # LLM forward pass with hidden state extraction
    # → max_new_tokens=1: 답 생성은 불필요, hidden state만 필요
    out = llm.generate(
        user_message=user_msg,
        system_message=system_msg,
        max_new_tokens=1,
        return_hidden_states=True,  # hidden state 반환 요청
        hidden_layer=sae.layer,      # SAE가 학습된 layer만 추출 (메모리 절약)
    )

    if out.hidden_states is None:
        # hidden state 추출 실패 (모델 미지원 등)
        return None

    # SAE encode → (n_features,) sparse activation
    activations = sae.get_feature_activations(out.hidden_states)

    if not bias_feature_indices:
        # 식별된 feature 없으면 top-k feature의 평균 activation 사용
        # → 단점: bias와 무관한 일반 feature일 수 있음 (warning은 extract_all.py에서)
        topk_vals, _ = activations.topk(min(top_k, activations.shape[-1]))
        return topk_vals.mean().item()

    # 식별된 bias feature들의 activation만 추출 후 평균
    # → 강한 활성화 = 모델이 bias-related feature 사용 = 편향 신호 ↑
    selected = activations[bias_feature_indices]
    return selected.mean().item()
