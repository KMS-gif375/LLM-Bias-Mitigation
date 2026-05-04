"""
Question embedding 생성 유틸리티.

MoE gating network 입력용으로 sentence-transformer를 사용합니다.
한 번 계산한 embedding은 캐싱하여 재사용합니다.
"""

import json
from pathlib import Path

import torch


class EmbeddingExtractor:
    """
    sentence-transformers를 사용한 question embedding 생성기.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model_name: sentence-transformers 모델 ID.
            device: device 문자열.
        """
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers 미설치. `pip install sentence-transformers`"
            ) from e

        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._model.eval()

    def encode(self, text: str) -> torch.Tensor:
        """
        단일 텍스트를 embedding 벡터로 변환합니다.

        Args:
            text: 입력 텍스트.

        Returns:
            (embed_dim,) 텐서.
        """
        self._load()
        with torch.no_grad():
            vec = self._model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        return vec.cpu()

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """
        여러 텍스트를 batch로 인코딩합니다.

        Args:
            texts: 텍스트 리스트.

        Returns:
            (n, embed_dim) 텐서.
        """
        self._load()
        with torch.no_grad():
            vecs = self._model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        return vecs.cpu()


def build_question_text(item: dict) -> str:
    """
    BBQ instance에서 embedding용 텍스트를 만듭니다.

    Args:
        item: BBQ instance.

    Returns:
        context + question을 합친 문자열.
    """
    return f"{item.get('context', '')} {item.get('question', '')}".strip()


def cache_embeddings(
    items: list[dict],
    extractor: EmbeddingExtractor,
    cache_path: Path,
) -> dict[str, torch.Tensor]:
    """
    instance들의 embedding을 계산하고 캐싱합니다.

    Args:
        items: BBQ instance 리스트.
        extractor: EmbeddingExtractor.
        cache_path: 캐시 저장 경로 (.pt).

    Returns:
        {example_id: embedding_tensor} 딕셔너리.
    """
    if cache_path.exists():
        print(f"  [load cache] {cache_path}")
        return torch.load(cache_path, weights_only=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    texts = [build_question_text(item) for item in items]
    vecs = extractor.encode_batch(texts)

    embeddings = {
        item["example_id"]: vecs[i]
        for i, item in enumerate(items)
    }
    torch.save(embeddings, cache_path)
    return embeddings
