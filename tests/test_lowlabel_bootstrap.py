"""Schema-level checks for the approximate low-label bootstrap audit."""

import numpy as np
import pytest

from scripts.run_lowlabel_bootstrap import descriptive_bootstrap_tail_pvalue


def test_descriptive_bootstrap_tail_mass_has_add_one_floor():
    deltas = np.array([0.1, 0.2, 0.3, 0.4])
    assert descriptive_bootstrap_tail_pvalue(deltas) == pytest.approx(2 / 5)


def test_descriptive_bootstrap_tail_mass_rejects_invalid_samples():
    with pytest.raises(ValueError, match="must not be empty"):
        descriptive_bootstrap_tail_pvalue(np.array([]))
    with pytest.raises(ValueError, match="only finite"):
        descriptive_bootstrap_tail_pvalue(np.array([0.1, np.nan]))
