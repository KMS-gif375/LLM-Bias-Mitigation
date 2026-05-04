"""
LLM 로딩 및 추론 유틸리티 (HuggingFace transformers 기반).

Mac M4 Pro의 MPS와 RunPod의 CUDA 모두에서 작동하도록 device를 자동 선택합니다.
Llama-3.1, Gemma-2, Qwen-2.5의 chat template를 통일된 인터페이스로 처리합니다.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def select_device(prefer: str = "auto") -> torch.device:
    """
    사용 가능한 device를 선택합니다.

    Args:
        prefer: "auto", "mps", "cuda", "cpu" 중 하나.

    Returns:
        torch.device 객체.
    """
    if prefer != "auto":
        return torch.device(prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class GenerationOutput:
    """LLM 생성 결과를 담는 컨테이너."""

    text: str                              # 디코딩된 응답 텍스트
    input_ids: torch.Tensor                # 입력 token ids
    output_ids: torch.Tensor               # 출력 token ids (입력 포함)
    logits: Optional[torch.Tensor] = None  # 첫 번째 출력 토큰의 logit (s3용)
    hidden_states: Optional[tuple] = None  # layer별 hidden states (s5, s7용)


class LLMWrapper:
    """
    HuggingFace 모델 로더 및 추론 wrapper.

    chat_template를 자동 적용하고, hidden states / logits을 옵션으로 반환합니다.
    SAE/bias-head 신호 추출을 위해 hidden states를 캐싱할 수 있습니다.
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        device: str = "auto",
        hf_token: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name: HuggingFace 모델 ID.
            dtype: "bfloat16", "float16", "float32" 중 하나.
            device: "auto", "mps", "cuda", "cpu".
            hf_token: HuggingFace 토큰 (gated 모델용).
        """
        self.model_name = model_name
        self.device = select_device(device)
        self.torch_dtype = getattr(torch, dtype)

        print(f"[LLM 로드] {model_name} -> {self.device} ({dtype})")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=self.torch_dtype,
            device_map=str(self.device) if self.device.type != "mps" else None,
        )
        if self.device.type == "mps":
            self.model = self.model.to(self.device)
        self.model.eval()

    def build_chat_prompt(
        self,
        user_message: str,
        system_message: Optional[str] = None,
    ) -> str:
        """
        모델별 chat template를 적용하여 최종 프롬프트 문자열을 만듭니다.

        Args:
            user_message: 사용자 메시지.
            system_message: 시스템 메시지 (없으면 생략).

        Returns:
            chat template 적용된 프롬프트.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        return_logits: bool = False,
        return_hidden_states: bool = False,
        hidden_layer: Optional[int] = None,
    ) -> GenerationOutput:
        """
        LLM에 단일 질문을 보내고 응답을 받습니다.

        Args:
            user_message: 사용자 메시지.
            system_message: 시스템 메시지.
            max_new_tokens: 생성할 최대 토큰 수.
            temperature: 0이면 greedy, >0이면 sampling.
            return_logits: True면 첫 토큰 logit 반환 (s3용).
            return_hidden_states: True면 hidden states 반환 (s5, s7용).
            hidden_layer: 반환할 layer 인덱스 (None이면 모든 layer).

        Returns:
            GenerationOutput 객체.
        """
        prompt = self.build_chat_prompt(user_message, system_message)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        do_sample = temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else 1.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": return_logits,
            "output_hidden_states": return_hidden_states,
        }

        outputs = self.model.generate(**inputs, **gen_kwargs)

        # 새로 생성된 토큰만 디코딩
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs.sequences[0, input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 옵션: 첫 토큰 logit
        first_logits = None
        if return_logits and outputs.scores:
            first_logits = outputs.scores[0][0]  # (vocab_size,)

        # 옵션: hidden states
        hidden = None
        if return_hidden_states and outputs.hidden_states:
            # outputs.hidden_states: tuple of tuples
            # 첫 번째 step의 layer별 hidden states
            first_step = outputs.hidden_states[0]
            if hidden_layer is not None:
                hidden = first_step[hidden_layer]
            else:
                hidden = first_step

        return GenerationOutput(
            text=text,
            input_ids=inputs["input_ids"][0],
            output_ids=outputs.sequences[0],
            logits=first_logits,
            hidden_states=hidden,
        )

    @torch.inference_mode()
    def get_answer_logprobs(
        self,
        user_message: str,
        choice_tokens: list[str],
        system_message: Optional[str] = None,
    ) -> dict[str, float]:
        """
        선택지 토큰별 log probability를 계산합니다 (s3 신뢰도용).

        Args:
            user_message: 사용자 메시지.
            choice_tokens: 측정할 토큰 리스트 (예: ["A", "B", "C"]).
            system_message: 시스템 메시지.

        Returns:
            {token: logprob} 딕셔너리.
        """
        prompt = self.build_chat_prompt(user_message, system_message)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)
        last_logits = outputs.logits[0, -1, :]  # (vocab_size,)
        log_probs = torch.log_softmax(last_logits, dim=-1)

        result: dict[str, float] = {}
        for tok in choice_tokens:
            tok_ids = self.tokenizer.encode(tok, add_special_tokens=False)
            if not tok_ids:
                result[tok] = float("-inf")
                continue
            # 첫 토큰의 logprob 사용
            result[tok] = log_probs[tok_ids[0]].item()

        return result
