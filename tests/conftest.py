"""Shared fixtures for the TP04 behaviour tests.

The fixtures are small, invented data: no CSV read, no network, no MLflow. They
mirror the TP02 input contract (the six agreed columns, raw CSV-like values) so
that the real pipeline builder can be fitted on them in milliseconds.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from telco_churn.pipeline import build_pipeline
from telco_churn.training import SEED


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Twelve invented customers with the six documented feature columns.

    ``TotalCharges`` stays a string on purpose: the raw CSV delivers it that way
    and the pipeline owns the numeric parsing.
    """
    return pd.DataFrame(
        {
            "tenure": [1, 2, 5, 8, 12, 18, 24, 30, 36, 48, 60, 72],
            "MonthlyCharges": [
                29.85,
                56.95,
                53.85,
                42.30,
                70.70,
                99.65,
                89.10,
                29.75,
                104.80,
                45.25,
                80.85,
                20.15,
            ],
            "TotalCharges": [
                "29.85",
                "1889.5",
                "108.15",
                "1840.75",
                "151.65",
                "820.5",
                "1949.4",
                "301.9",
                "3046.05",
                "6844.5",
                "3487.95",
                "326.8",
            ],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "One year",
                "Two year",
            ],
            "InternetService": [
                "DSL",
                "Fiber optic",
                "DSL",
                "No",
                "Fiber optic",
                "DSL",
                "Fiber optic",
                "No",
                "DSL",
                "Fiber optic",
                "DSL",
                "No",
            ],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        }
    )


@pytest.fixture
def sample_target() -> pd.Series:
    """Binary target with both classes represented (6 churners out of 12)."""
    return pd.Series([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0], name="Churn")


@pytest.fixture
def fitted_pipeline(sample_features, sample_target):
    """The real TP02 pipeline (preprocessing + classifier) fitted on the fixture."""
    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=SEED))
    pipeline.fit(sample_features, sample_target)
    return pipeline
