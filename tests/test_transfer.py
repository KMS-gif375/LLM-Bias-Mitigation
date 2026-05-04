"""
Cross-LLM + Transfer 모듈 단위 테스트.

LLM 호출이나 실제 데이터 없이 동작 가능한 부분만 검증합니다.

실행:
    pytest tests/test_transfer.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from src.cross_llm.gemma_pipeline import GemmaConfig
from src.cross_llm.qwen_pipeline import (
    QwenConfig,
    SIX_SIGNAL_KEYS,
    make_six_signal_moe,
    signals_dict_to_six_signal_tensor,
)
from src.models.moe_aggregator import MoEAggregator
from src.transfer.implicit_bbq import (
    ClusterRoutingStats,
    analyze_cluster_routing,
    load_implicit_bbq,
    save_transfer_result,
    transfer_evaluate,
)
from src.transfer.openbias import (
    DEFAULT_CATEGORY_TO_CLUSTER,
    OpenBiasTransferResult,
    RoutingAccuracyResult,
    compute_routing_accuracy,
    load_openbias,
    save_openbias_result,
    transfer_evaluate_openbias,
)


SIGNAL_DIM = 7
EMBED_DIM = 32


# =============================================================
# Helpers
# =============================================================
def _make_item(category: str = "Age", label: int = 0, condition: str = "ambig") -> dict:
    return {
        "example_id": f"{category}_{label}",
        "category": category,
        "label": label,
        "context_condition": condition,
        "question_polarity": "neg",
        "context": "ctx",
        "question": "q",
        "ans0": "a", "ans1": "b", "ans2": "Cannot be determined",
        "answer_info": {
            "ans0": ["a", "old"],
            "ans1": ["b", "nonOld"],
            "ans2": ["", "unknown"],
        },
        "additional_metadata": {"stereotyped_groups": ["old"]},
    }


def _dummy_signals(instance, primary_idx):
    return {f"s{i}_{name}": 0.5 for i, name in enumerate(
        ["evidence", "counterfactual", "confidence", "consistency",
         "bias_head", "prompt_sensitivity", "sae_feature"], 1
    )}


def _dummy_embedding(instance):
    return torch.zeros(EMBED_DIM)


# =============================================================
# Cross-LLM Configs
# =============================================================
class TestCrossLLMConfigs:
    def test_gemma_config_defaults(self):
        cfg = GemmaConfig()
        assert "gemma" in cfg.model_name.lower()
        assert "gemma-scope" in cfg.sae_repo.lower()
        assert isinstance(cfg.bias_head_indices, list)

    def test_qwen_config_defaults(self):
        cfg = QwenConfig()
        assert "qwen" in cfg.model_name.lower()
        assert cfg.s7_strategy == "zero_padding"


# =============================================================
# Qwen 6-signal helpers
# =============================================================
class TestQwenSixSignal:
    def test_six_signal_moe_shape(self):
        moe = make_six_signal_moe(embed_dim=EMBED_DIM, gating_hidden=8, expert_hidden=8)
        assert moe.signal_dim == 6
        signals = torch.rand(2, 6)
        embed = torch.randn(2, EMBED_DIM)
        out = moe(signals, embed)
        assert out.p.shape == (2,)

    def test_six_signal_keys_excludes_s7(self):
        assert "s7_sae_feature" not in SIX_SIGNAL_KEYS
        assert len(SIX_SIGNAL_KEYS) == 6

    def test_signals_dict_to_six_signal_tensor(self):
        signals = {
            "s1_evidence": 1.0,
            "s2_counterfactual": 0.5,
            "s3_confidence": 0.8,
            "s4_consistency": 0.6,
            "s5_bias_head": 0.2,
            "s6_prompt_sensitivity": 0.75,
            "s7_sae_feature": 0.999,         # 무시되어야 함
        }
        t = signals_dict_to_six_signal_tensor(signals)
        assert t.shape == (6,)
        assert t[0].item() == 1.0


# =============================================================
# ImplicitBBQ loader
# =============================================================
class TestImplicitBBQLoader:
    def test_load_jsonl(self, tmp_path):
        data_dir = tmp_path / "implicit_bbq"
        data_dir.mkdir()
        with open(data_dir / "Age.jsonl", "w") as f:
            for label in (0, 1):
                f.write(json.dumps(_make_item(label=label)) + "\n")

        items = load_implicit_bbq(data_dir)
        assert len(items) == 2
        assert all(it["category"] == "Age" for it in items)

    def test_filter_categories(self, tmp_path):
        data_dir = tmp_path / "implicit_bbq"
        data_dir.mkdir()
        for cat in ("Age", "Gender_identity"):
            with open(data_dir / f"{cat}.jsonl", "w") as f:
                f.write(json.dumps(_make_item(category=cat)) + "\n")

        items = load_implicit_bbq(data_dir, categories=["Age"])
        assert len(items) == 1
        assert items[0]["category"] == "Age"

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ImplicitBBQ"):
            load_implicit_bbq(tmp_path / "nope")


# =============================================================
# Cluster routing
# =============================================================
class TestClusterRouting:
    def test_routing_stats_shape(self):
        moe = MoEAggregator(
            signal_dim=SIGNAL_DIM, embed_dim=EMBED_DIM,
            num_experts=4, gating_hidden=8, expert_hidden=8,
        )
        items = [
            _make_item(category="Age"),
            _make_item(category="Age"),
            _make_item(category="Religion"),
        ]
        stats = analyze_cluster_routing(moe, items, _dummy_embedding)

        assert isinstance(stats, ClusterRoutingStats)
        assert "Age" in stats.avg_weights_per_category
        assert "Religion" in stats.avg_weights_per_category
        assert len(stats.overall_avg_weights) == 4
        assert stats.n_per_category["Age"] == 2

    def test_dominant_cluster_argmax(self):
        moe = MoEAggregator(
            signal_dim=SIGNAL_DIM, embed_dim=EMBED_DIM,
            num_experts=4, gating_hidden=8, expert_hidden=8,
        )
        items = [_make_item(category="Age")]
        stats = analyze_cluster_routing(moe, items, _dummy_embedding)
        dom = stats.dominant_cluster_per_category["Age"]
        assert 0 <= dom < 4


# =============================================================
# Transfer evaluation
# =============================================================
class TestTransferEvaluate:
    def test_basic_transfer(self):
        moe = MoEAggregator(
            signal_dim=SIGNAL_DIM, embed_dim=EMBED_DIM,
            num_experts=4, gating_hidden=8, expert_hidden=8, dropout=0.0,
        )
        items = [
            _make_item(category="Age", label=0, condition="ambig"),
            _make_item(category="Age", label=1, condition="disambig"),
        ]
        primary = ["A", "B"]
        result = transfer_evaluate(
            instances=items,
            primary_answers=primary,
            moe_model=moe,
            signal_extractor=_dummy_signals,
            embedding_extractor=_dummy_embedding,
            show_progress=False,
        )
        assert "accuracy_amb" in result.overall_metrics
        assert "Age" in result.metrics_per_category
        assert isinstance(result.routing_stats, ClusterRoutingStats)
        assert result.n_total == 2

    def test_length_mismatch_raises(self):
        moe = MoEAggregator(signal_dim=SIGNAL_DIM, embed_dim=EMBED_DIM,
                            num_experts=4, gating_hidden=8, expert_hidden=8)
        with pytest.raises(ValueError, match="길이 불일치"):
            transfer_evaluate(
                instances=[_make_item()],
                primary_answers=["A", "B"],
                moe_model=moe,
                signal_extractor=_dummy_signals,
                embedding_extractor=_dummy_embedding,
            )

    def test_save_result(self, tmp_path):
        moe = MoEAggregator(
            signal_dim=SIGNAL_DIM, embed_dim=EMBED_DIM,
            num_experts=4, gating_hidden=8, expert_hidden=8,
        )
        result = transfer_evaluate(
            instances=[_make_item()],
            primary_answers=["A"],
            moe_model=moe,
            signal_extractor=_dummy_signals,
            embedding_extractor=_dummy_embedding,
            show_progress=False,
        )
        out_path = tmp_path / "result.json"
        save_transfer_result(result, out_path)
        assert out_path.exists()

        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "overall_metrics" in data
        assert "routing_stats" in data


# =============================================================
# OpenBiasBench
# =============================================================
class TestOpenBias:
    def test_default_category_mapping(self):
        # 적어도 4개 cluster 모두 매핑된 카테고리가 있어야 함
        clusters = set(DEFAULT_CATEGORY_TO_CLUSTER.values())
        assert clusters == {0, 1, 2, 3}

    def test_compute_routing_accuracy(self):
        # Age는 cluster 1, Religion은 cluster 0 (DEFAULT 매핑 기준)
        stats = ClusterRoutingStats(
            avg_weights_per_category={
                "Age": [0.1, 0.7, 0.1, 0.1],          # dom=1, gt=1 ✓
                "Religion": [0.1, 0.1, 0.7, 0.1],     # dom=2, gt=0 ✗
            },
            dominant_cluster_per_category={"Age": 1, "Religion": 2},
            overall_avg_weights=[0.25, 0.25, 0.25, 0.25],
            n_per_category={"Age": 10, "Religion": 5},
        )
        acc = compute_routing_accuracy(stats)
        assert isinstance(acc, RoutingAccuracyResult)
        assert acc.accuracy == 10 / 15        # Age 10개 정답, Religion 5개 오답
        assert acc.accuracy_per_category["Age"] == 1.0
        assert acc.accuracy_per_category["Religion"] == 0.0
        assert acc.n_unmapped == 0

    def test_unmapped_category(self):
        stats = ClusterRoutingStats(
            avg_weights_per_category={"NewCategoryXYZ": [0.25] * 4},
            dominant_cluster_per_category={"NewCategoryXYZ": 0},
            overall_avg_weights=[0.25] * 4,
            n_per_category={"NewCategoryXYZ": 5},
        )
        acc = compute_routing_accuracy(stats)
        assert acc.n_unmapped == 5
        assert acc.accuracy == 0.0  # 평가 가능한 sample 없음

    def test_load_openbias_jsonl(self, tmp_path):
        data_dir = tmp_path / "openbias"
        data_dir.mkdir()
        with open(data_dir / "NewCategory.jsonl", "w") as f:
            f.write(json.dumps(_make_item(category="NewCategory")) + "\n")

        items = load_openbias(data_dir)
        assert len(items) == 1

    def test_load_openbias_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="OpenBiasBench"):
            load_openbias(tmp_path / "nope")

    def test_transfer_evaluate_openbias(self, tmp_path):
        moe = MoEAggregator(
            signal_dim=SIGNAL_DIM, embed_dim=EMBED_DIM,
            num_experts=4, gating_hidden=8, expert_hidden=8, dropout=0.0,
        )
        items = [
            _make_item(category="Age", label=0, condition="ambig"),
            _make_item(category="Religion", label=1, condition="disambig"),
        ]
        primary = ["A", "B"]
        result = transfer_evaluate_openbias(
            instances=items,
            primary_answers=primary,
            moe_model=moe,
            signal_extractor=_dummy_signals,
            embedding_extractor=_dummy_embedding,
            show_progress=False,
        )
        assert isinstance(result, OpenBiasTransferResult)

        out_path = tmp_path / "openbias_result.json"
        save_openbias_result(result, out_path)
        assert out_path.exists()
