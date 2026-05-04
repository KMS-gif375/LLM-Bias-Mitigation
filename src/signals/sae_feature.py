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

from typing import Optional

import torch

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
        release: str,
        sae_id: str,
        layer: int,
        device: str = "auto",
    ) -> None:
        """
        Args:
            release: SAELens 사전 등록 release 이름 (예: "llama_scope_lxr_8x").
            sae_id: release 안의 SAE 식별자 (예: "l15r_8x").
            layer: hook 시 사용할 layer 인덱스 (sae_id와 일관 유지).
            device: device 문자열.
        """
        self.release = release
        self.sae_id = sae_id
        self.layer = layer
        self.device = device
        self._sae = None  # lazy load

    def _load(self) -> None:
        """SAE를 lazy하게 로드합니다."""
        if self._sae is not None:
            return
        try:
            from sae_lens import SAE
        except ImportError as e:
            raise RuntimeError(
                "sae_lens 미설치. `pip install sae-lens`"
            ) from e

        result = SAE.from_pretrained(
            release=self.release,
            sae_id=self.sae_id,
            device=self.device,
        )
        # SAELens 버전에 따라 단일 SAE 객체 또는 (sae, cfg, sparsity) 튜플 반환
        self._sae = result[0] if isinstance(result, tuple) else result
        self._sae.eval()

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
        if hidden_state.dim() == 3:
            # 마지막 토큰만 사용
            hidden_state = hidden_state[0, -1, :]
        elif hidden_state.dim() == 2:
            hidden_state = hidden_state[-1, :]

        with torch.no_grad():
            features = self._sae.encode(hidden_state.unsqueeze(0))
        return features.squeeze(0)


def compute_sae_signal(
    item: dict,
    llm: LLMWrapper,
    sae: Optional[SAEWrapper],
    prompt_builder,
    bias_feature_indices: list[int],
    top_k: int = 50,
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
    if sae is None:
        return None

    system_msg, user_msg = prompt_builder(item)

    out = llm.generate(
        user_message=user_msg,
        system_message=system_msg,
        max_new_tokens=1,
        return_hidden_states=True,
        hidden_layer=sae.layer,
    )

    if out.hidden_states is None:
        return None

    activations = sae.get_feature_activations(out.hidden_states)

    if not bias_feature_indices:
        # 식별된 feature 없으면 top-k feature의 평균 activation 사용
        topk_vals, _ = activations.topk(min(top_k, activations.shape[-1]))
        return topk_vals.mean().item()

    selected = activations[bias_feature_indices]
    return selected.mean().item()
