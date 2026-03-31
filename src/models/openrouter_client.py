"""
OpenRouter 통합 API 클라이언트

OpenRouter는 OpenAI 호환 API를 제공하므로,
openai 라이브러리를 그대로 사용하되 base_url만 바꿔주면 됩니다.
이 하나의 클라이언트로 GPT-4o, GPT-3.5, LLaMA-3, Mistral 모두 호출 가능.

동기(ask) 및 비동기(ask_async, ask_batch) 호출을 모두 지원합니다.
"""

import asyncio
import os

from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

# .env 파일에서 API 키를 불러옴
load_dotenv()


class OpenRouterClient:
    """
    OpenRouter를 통해 여러 LLM을 호출하는 클라이언트.

    사용 예시:
        client = OpenRouterClient()

        # 동기 호출
        answer = client.ask("openai/gpt-4o", "한국의 수도는?")

        # 비동기 배치 호출
        answers = asyncio.run(client.ask_batch("openai/gpt-4o", messages_list, concurrency=10))
    """

    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY가 설정되지 않았습니다.\n"
                ".env 파일에 OPENROUTER_API_KEY=your_key 를 추가해주세요."
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        self.async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def ask(self, model_id, user_message, system_message=None, temperature=0, max_tokens=256):
        """
        동기 호출: 모델에게 질문을 보내고 답변 텍스트를 반환합니다.
        """
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

        return response.choices[0].message.content.strip()

    async def ask_async(self, model_id, user_message, system_message=None, temperature=0, max_tokens=256):
        """
        비동기 호출: 단일 질문을 비동기로 처리합니다.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        response = await self.async_client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()

    async def ask_batch(self, model_id, request_list, temperature=0, max_tokens=256, concurrency=10):
        """
        비동기 배치 호출: 여러 질문을 동시에 처리합니다.

        Args:
            model_id: 모델 ID
            request_list: (system_message, user_message) 튜플 리스트
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
            concurrency: 동시 호출 수 (기본 10)

        Returns:
            답변 텍스트 리스트 (입력 순서와 동일)
        """
        semaphore = asyncio.Semaphore(concurrency)
        results = [None] * len(request_list)

        async def _call(idx, system_msg, user_msg):
            async with semaphore:
                try:
                    answer = await self.ask_async(
                        model_id, user_msg, system_msg, temperature, max_tokens
                    )
                except Exception as e:
                    answer = ""
                results[idx] = answer

        tasks = []
        for i, (sys_msg, usr_msg) in enumerate(request_list):
            tasks.append(asyncio.create_task(_call(i, sys_msg, usr_msg)))

        await asyncio.gather(*tasks)
        return results
