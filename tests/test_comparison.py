"""Contract of the TP02 comparison table.

``compare_models`` was the only package function with no test at all: the CLI uses
``train_and_select`` and nothing exercised the comparison helper itself. The test feeds
a synthetic frame (no CSV) through the real code path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from telco_churn import training


@pytest.fixture
def synthetic_frame(sample_features, sample_target) -> pd.DataFrame:
    """Five copies of the fixture (60 customers, 30 churners) with unique IDs."""
    blocks = []
    for repeat in range(5):
        block = sample_features.copy()
        block.insert(0, "customerID", [f"0000-SYN{repeat}{row:02d}" for row in range(len(block))])
        block["Churn"] = np.where(sample_target.to_numpy() == 1, "Yes", "No")
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def test_compare_models_returns_a_sorted_row_per_candidate(monkeypatch, synthetic_frame):
    """One row per candidate, full metric set, sorted by the primary metric (AP)."""
    monkeypatch.setattr(training, "read_telco_data", lambda source=None: synthetic_frame)

    results = training.compare_models()

    assert len(results) == len(training.candidate_models())
    assert set(results["model"]) == set(training.candidate_models())
    assert {
        "accuracy",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
    } <= set(results.columns)

    scores = results["average_precision"].to_list()
    assert scores == sorted(scores, reverse=True)
    assert results["average_precision"].between(0.0, 1.0).all()
