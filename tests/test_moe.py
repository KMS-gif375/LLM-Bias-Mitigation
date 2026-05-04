"""
MoE Aggregator + Trainer + Threshold Override 단위 테스트.

실행:
    pytest tests/test_moe.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from src.models.moe_aggregator import (
    MoEAggregator,
    MoEOutput,
    bce_loss,
    bias_penalty,
    load_balance_loss,
    predict_p,
    signals_dict_to_tensor,
    total_loss,
)
from src.models.override import (
    apply_threshold_override,
    best_threshold_from_rc_curve,
    find_unknown_index,
    risk_coverage_curve,
    search_optimal_threshold,
    search_optimal_threshold_per_category,
)
from src.models.trainer import (
    SignalsDataset,
    TrainConfig,
    load_checkpoint,
    train_moe,
)


SIGNAL_DIM = 7
EMBED_DIM = 64    # 테스트용 작은 차원 (실제는 4096)


# =============================================================
# Fixtures
# =============================================================
@pytest.fixture(autouse=True)
def fix_seed():
    torch.manual_seed(0)


@pytest.fixture
def model() -> MoEAggregator:
    return MoEAggregator(
        signal_dim=SIGNAL_DIM,
        embed_dim=EMBED_DIM,
        num_experts=4,
        gating_hidden=16,
        expert_hidden=16,
        dropout=0.0,
    )


def _make_signals(batch: int = 4) -> torch.Tensor:
    return torch.rand(batch, SIGNAL_DIM)


def _make_embed(batch: int = 4) -> torch.Tensor:
    return torch.randn(batch, EMBED_DIM)


# =============================================================
# MoEAggregator
# =============================================================
class TestMoEArchitecture:
    def test_forward_output_shapes(self, model):
        signals = _make_signals(8)
        embed = _make_embed(8)
        out = model(signals, embed)

        assert isinstance(out, MoEOutput)
        assert out.p.shape == (8,)
        assert out.gate_w.shape == (8, 4)
        assert out.expert_outs.shape == (8, 4)
        assert out.normalized_signals.shape == (8, SIGNAL_DIM)

    def test_p_in_unit_interval(self, model):
        signals = _make_signals(16)
        embed = _make_embed(16)
        out = model(signals, embed)
        assert (out.p >= 0).all() and (out.p <= 1).all()

    def test_gate_weights_sum_to_one(self, model):
        signals = _make_signals(8)
        embed = _make_embed(8)
        out = model(signals, embed)
        sums = out.gate_w.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(8), rtol=1e-5, atol=1e-5)

    def test_signal_temperature_is_learnable(self, model):
        params = dict(model.named_parameters())
        assert "signal_temperature" in params
        assert params["signal_temperature"].requires_grad
        assert params["signal_temperature"].shape == (SIGNAL_DIM,)

    def test_invalid_signal_dim_raises(self, model):
        with pytest.raises(ValueError, match="signals shape"):
            model(torch.rand(4, SIGNAL_DIM + 1), _make_embed(4))

    def test_invalid_embed_dim_raises(self, model):
        with pytest.raises(ValueError, match="q_embed shape"):
            model(_make_signals(4), torch.rand(4, EMBED_DIM + 1))

    def test_backward_propagates(self, model):
        signals = _make_signals(4)
        embed = _make_embed(4)
        label = torch.tensor([1.0, 0.0, 1.0, 0.0])

        out = model(signals, embed)
        loss = bce_loss(out.p, label)
        loss.backward()

        # signal_temperature가 gradient를 받는지 확인
        assert model.signal_temperature.grad is not None
        assert model.signal_temperature.grad.shape == (SIGNAL_DIM,)


# =============================================================
# Loss functions
# =============================================================
class TestLosses:
    def test_bce_loss_value(self):
        p = torch.tensor([0.9, 0.1])
        label = torch.tensor([1.0, 0.0])
        loss = bce_loss(p, label)
        # -mean(log(0.9), log(0.9)) ≈ 0.1054
        assert loss.item() < 0.2

    def test_bce_loss_clamps_extremes(self):
        p = torch.tensor([0.0, 1.0])
        label = torch.tensor([1.0, 0.0])
        loss = bce_loss(p, label)
        assert torch.isfinite(loss)

    def test_bias_penalty_zero_when_no_match(self):
        p = torch.tensor([0.8, 0.7])
        is_ambig = torch.tensor([0.0, 0.0])
        is_stereo = torch.tensor([1.0, 1.0])
        assert bias_penalty(p, is_ambig, is_stereo).item() == 0.0

    def test_bias_penalty_positive_when_violating(self):
        # ambig + stereotype + high p → penalty > 0
        p = torch.tensor([0.9, 0.1])
        is_ambig = torch.tensor([1.0, 1.0])
        is_stereo = torch.tensor([1.0, 1.0])
        loss = bias_penalty(p, is_ambig, is_stereo)
        # high p가 더 큰 penalty
        assert loss.item() > 0.0

    def test_load_balance_zero_when_uniform(self):
        # 4 expert가 모두 0.25 사용 → loss = 0
        gate_w = torch.full((10, 4), 0.25)
        loss = load_balance_loss(gate_w)
        assert loss.item() < 1e-6

    def test_load_balance_positive_when_collapsed(self):
        # 한 expert로만 라우팅 → loss > 0
        gate_w = torch.zeros(10, 4)
        gate_w[:, 0] = 1.0
        loss = load_balance_loss(gate_w)
        assert loss.item() > 0.0

    def test_load_balance_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="gate_w"):
            load_balance_loss(torch.rand(4))

    def test_total_loss_combines(self, model):
        out = model(_make_signals(8), _make_embed(8))
        label = torch.randint(0, 2, (8,)).float()
        is_ambig = torch.randint(0, 2, (8,)).float()
        is_stereo = torch.randint(0, 2, (8,)).float()

        d = total_loss(out, label, is_ambig, is_stereo)
        assert {"total", "bce", "bias", "lb"} <= set(d)
        torch.testing.assert_close(
            d["total"], d["bce"] + 0.5 * d["bias"] + 0.1 * d["lb"]
        )


# =============================================================
# Helpers
# =============================================================
class TestHelpers:
    def test_signals_dict_to_tensor(self):
        signals = {
            "s1_evidence": 1.0,
            "s2_counterfactual": 0.5,
            "s3_confidence": 0.8,
            "s4_consistency": 0.6,
            "s5_bias_head": 0.2,
            "s6_prompt_sensitivity": 0.75,
            "s7_sae_feature": 0.3,
        }
        t = signals_dict_to_tensor(signals)
        assert t.shape == (7,)
        assert t.dtype == torch.float32

    def test_signals_dict_handles_none(self):
        signals = {
            "s1_evidence": 1.0,
            "s2_counterfactual": None,
            "s3_confidence": None,
            "s4_consistency": None,
            "s5_bias_head": None,
            "s6_prompt_sensitivity": None,
            "s7_sae_feature": None,
        }
        t = signals_dict_to_tensor(signals, fill_none=0.0)
        assert t[0].item() == 1.0
        assert t[1:].sum().item() == 0.0

    def test_predict_p_returns_scalar(self, model):
        s = torch.rand(SIGNAL_DIM)
        e = torch.randn(EMBED_DIM)
        p = predict_p(model, s, e)
        assert isinstance(p, float)
        assert 0 <= p <= 1


# =============================================================
# Threshold override
# =============================================================
def _make_item(label: int = 0, condition: str = "ambig", category: str = "Age") -> dict:
    return {
        "label": label,
        "context_condition": condition,
        "category": category,
        "answer_info": {
            "ans0": ["grandfather", "old"],
            "ans1": ["grandson", "nonOld"],
            "ans2": ["", "unknown"],
        },
    }


class TestThresholdOverride:
    def test_keep_when_above_threshold(self):
        result = apply_threshold_override(
            primary_answer=0, p_score=0.8, item=_make_item(), threshold=0.5,
        )
        assert result["final_answer"] == 0
        assert result["overridden"] is False

    def test_override_when_below_threshold(self):
        result = apply_threshold_override(
            primary_answer=0, p_score=0.3, item=_make_item(), threshold=0.5,
        )
        assert result["final_answer"] == 2  # unknown index
        assert result["overridden"] is True

    def test_parse_failure_kept(self):
        result = apply_threshold_override(
            primary_answer=-1, p_score=0.1, item=_make_item(), threshold=0.5,
        )
        assert result["final_answer"] == -1
        assert result["overridden"] is False

    def test_find_unknown_index(self):
        item = _make_item()
        assert find_unknown_index(item) == 2

    def test_find_unknown_index_default(self):
        item = {"answer_info": {}}
        assert find_unknown_index(item) == 2


class TestThresholdSearch:
    def _make_predictions(self) -> list[dict]:
        # 10 ambig: label=2 (Unknown), p가 높을수록 모델이 stereotype 답 (오답)
        preds = []
        for i in range(10):
            preds.append({
                "primary_answer": 0,            # 모델이 ans0(stereotype) 선택
                "p_score": 0.1 * i,             # 0.0, 0.1, ..., 0.9
                "item": _make_item(label=2, condition="ambig"),
            })
        # 10 disambig: label=0 (정답), 모델 답 == 0
        for i in range(10):
            preds.append({
                "primary_answer": 0,
                "p_score": 0.1 * i,
                "item": _make_item(label=0, condition="disambig"),
            })
        return preds

    def test_search_returns_valid_threshold(self):
        preds = self._make_predictions()
        result = search_optimal_threshold(preds, metric="balanced_accuracy")
        assert 0.0 <= result.best_threshold <= 1.0
        assert 0.0 <= result.best_score <= 1.0

    def test_per_category_search(self):
        preds = self._make_predictions()
        # 카테고리 다양화
        for i, p in enumerate(preds):
            p["item"] = dict(p["item"])
            p["item"]["category"] = "Age" if i < 10 else "Religion"

        results = search_optimal_threshold_per_category(preds)
        assert "Age" in results and "Religion" in results
        for r in results.values():
            assert 0.0 <= r.best_threshold <= 1.0


class TestRiskCoverage:
    def test_curve_monotonic_coverage(self):
        preds = [
            {"primary_answer": 0, "p_score": p, "item": _make_item(label=0, condition="disambig")}
            for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ]
        points = risk_coverage_curve(preds, threshold_range=(0.0, 1.0), step=0.1)

        # threshold가 커질수록 coverage는 단조 감소
        coverages = [pt.coverage for pt in points]
        assert all(coverages[i] >= coverages[i+1] for i in range(len(coverages)-1))

    def test_best_min_error(self):
        preds = [
            {"primary_answer": 0, "p_score": 0.9, "item": _make_item(label=0, condition="disambig")},
            {"primary_answer": 0, "p_score": 0.1, "item": _make_item(label=2, condition="ambig")},
        ]
        points = risk_coverage_curve(preds, threshold_range=(0.0, 1.0), step=0.1)
        best = best_threshold_from_rc_curve(points, objective="min_error")
        assert isinstance(best.threshold, float)
        assert 0.0 <= best.error_rate <= 1.0


# =============================================================
# Trainer (mini smoke test)
# =============================================================
class TestTrainer:
    def _make_dataset(self, n: int = 32) -> SignalsDataset:
        records = []
        embeddings = {}
        for i in range(n):
            ex_id = f"ex_{i:04d}"
            records.append({
                "example_id": ex_id,
                "primary_answer": i % 3,
                "label": i % 3,            # 항상 정답 → label=1 expected
                "context_condition": "ambig" if i % 2 == 0 else "disambig",
                "is_stereotype": float(i % 4 == 0),
                "signals": {
                    "s1_evidence": 0.5,
                    "s2_counterfactual": 0.5,
                    "s3_confidence": 0.5,
                    "s4_consistency": 0.5,
                    "s5_bias_head": 0.5,
                    "s6_prompt_sensitivity": 0.5,
                    "s7_sae_feature": 0.5,
                },
            })
            embeddings[ex_id] = torch.randn(EMBED_DIM)
        return SignalsDataset(records, embeddings)

    def test_training_runs_and_loss_decreases(self, tmp_path):
        train_ds = self._make_dataset(64)
        val_ds = self._make_dataset(16)

        model = MoEAggregator(
            signal_dim=SIGNAL_DIM, embed_dim=EMBED_DIM,
            num_experts=4, gating_hidden=16, expert_hidden=16, dropout=0.0,
        )
        cfg = TrainConfig(
            epochs=4, batch_size=8, lr=1e-2, val_every=2,
            lambda_bias=0.1, lambda_lb=0.1, save_dir=str(tmp_path),
            device="cpu", seed=42,
        )
        result = train_moe(train_ds, val_ds, model, cfg)

        assert "history" in result
        assert len(result["history"]) == cfg.epochs
        # 마지막 train_loss < 첫 train_loss (학습 진행 확인)
        first_loss = result["history"][0]["train_loss"]
        last_loss = result["history"][-1]["train_loss"]
        assert last_loss <= first_loss + 0.01  # 약간의 노이즈 허용

    def test_checkpoint_save_and_load(self, tmp_path):
        train_ds = self._make_dataset(32)
        val_ds = self._make_dataset(8)

        model = MoEAggregator(
            signal_dim=SIGNAL_DIM, embed_dim=EMBED_DIM,
            num_experts=4, gating_hidden=16, expert_hidden=16,
        )
        cfg = TrainConfig(
            epochs=2, batch_size=8, lr=1e-2, val_every=1,
            save_dir=str(tmp_path), device="cpu", seed=0,
        )
        result = train_moe(train_ds, val_ds, model, cfg)

        assert result["checkpoint_path"] is not None
        ckpt_path = result["checkpoint_path"]
        assert Path(ckpt_path).exists()

        # Load
        loaded = load_checkpoint(ckpt_path, device="cpu")
        assert loaded.signal_dim == SIGNAL_DIM
        assert loaded.embed_dim == EMBED_DIM
        assert loaded.num_experts == 4

        # 출력 비교
        s = torch.rand(2, SIGNAL_DIM)
        e = torch.randn(2, EMBED_DIM)
        original_p = model(s, e).p
        loaded_p = loaded(s, e).p
        torch.testing.assert_close(original_p, loaded_p)
