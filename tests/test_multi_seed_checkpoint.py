"""Regression tests for validation-best checkpoint restoration in multi-seed runs."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from src.analysis.multi_seed import _restore_checkpoint, run_single_seed


def test_restore_checkpoint_replaces_final_epoch_parameters(tmp_path):
    """Evaluation must use the validation-best weights, not final-epoch weights."""
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(9.0)
        model.bias.fill_(9.0)

    best_model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        best_model.weight.fill_(2.0)
        best_model.bias.fill_(-3.0)

    checkpoint = tmp_path / "moe_best.pt"
    torch.save({"model_state_dict": best_model.state_dict()}, checkpoint)

    _restore_checkpoint(model, checkpoint)

    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 2.0))
    torch.testing.assert_close(model.bias, torch.full_like(model.bias, -3.0))


def test_restore_checkpoint_accepts_raw_state_dict(tmp_path):
    """Archived checkpoints may contain a state dict without a payload wrapper."""
    model = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "raw_state.pt"
    expected = {
        "weight": torch.full_like(model.weight, 4.0),
        "bias": torch.full_like(model.bias, 5.0),
    }
    torch.save(expected, checkpoint)

    _restore_checkpoint(model, checkpoint)

    torch.testing.assert_close(model.weight, expected["weight"])
    torch.testing.assert_close(model.bias, expected["bias"])


def test_restore_checkpoint_none_is_noop():
    model = torch.nn.Linear(2, 1)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    _restore_checkpoint(model, None)

    for name, expected in before.items():
        torch.testing.assert_close(model.state_dict()[name], expected)


def test_restore_checkpoint_missing_file_raises(tmp_path):
    model = torch.nn.Linear(2, 1)

    with pytest.raises(FileNotFoundError):
        _restore_checkpoint(model, tmp_path / "missing.pt")


def test_single_seed_restores_best_before_any_inference():
    """Guard the ordering that prevents final-epoch cross-LLM evaluation."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(run_single_seed)))
    call_lines: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            call_lines.setdefault(node.func.id, []).append(node.lineno)

    train_line = min(call_lines["train_moe"])
    restore_line = min(call_lines["_restore_checkpoint"])
    first_predict_line = min(call_lines["_moe_predict_all"])

    assert train_line < restore_line < first_predict_line
