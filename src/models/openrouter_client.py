"""
OpenRouter 통합 API 클라이언트

OpenRouter는 OpenAI 호환 API를 제공하므로,
openai 라이브러리를 그대로 사용하되 base_url만 바꿔주면 됩니다.
이 하나의 클라이언트로 GPT-4o, GPT-3.5, LLaMA-3, Mistral 모두 호출 가능.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일에서 API 키를 불러옴
load_dotenv()


class OpenRouterClient:
    """
    OpenRouter를 통해 여러 LLM을 호출하는 클라이언트.

    사용 예시:
        client = OpenRouterClient()
        answer = client.ask("openai/gpt-4o", "한국의 수도는?")
    """

    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY가 설정되지 않았습니다.\n"
                ".env 파일에 OPENROUTER_API_KEY=your_key 를 추가해주세요."
            )

        # OpenAI 클라이언트를 OpenRouter 주소로 연결
        # base_url만 바꾸면 나머지는 OpenAI와 동일하게 사용 가능
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def ask(self, model_id, user_message, system_message=None, temperature=0, max_tokens=256):
        """
        모델에게 질문을 보내고 답변 텍스트를 반환합니다.

        Args:
            model_id: OpenRouter 모델 ID (예: "openai/gpt-4o")
            user_message: 사용자 메시지 (BBQ 질문 등)
            system_message: 시스템 메시지 (프롬프트 기법에 따라 다름, 없으면 생략)
            temperature: 0이면 항상 같은 답변 (재현성)
            max_tokens: 최대 응답 길이

        Returns:
            모델의 답변 텍스트 (str)
        """
        # 메시지 구성: system이 있으면 앞에 붙이고, user 메시지를 뒤에 추가
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # 응답에서 텍스트만 꺼내서 반환
        return response.choices[0].message.content.strip()
