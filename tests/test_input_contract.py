"""Input-contract tests: what the prediction boundary must accept and reject."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from telco_churn.pipeline import prepare_features
from telco_churn.prediction import predict_customers


def test_missing_required_feature_is_rejected(fitted_pipeline, sample_features):
    """Dropping a required column must fail loudly, not silently score the batch.

    Risk covered: a refactor that stops validating the feature list would let a
    column-shifted batch flow into the encoder and produce wrong predictions.
    """
    broken_batch = sample_features.drop(columns=["Contract"])

    with pytest.raises(KeyError, match="Contract"):
        predict_customers(fitted_pipeline, broken_batch)


@pytest.mark.parametrize(
    "column",
    ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "PaymentMethod"],
)
def test_every_documented_feature_is_required(fitted_pipeline, sample_features, column):
    """The documented six-column contract holds for each feature, not just one."""
    with pytest.raises(KeyError, match=column):
        predict_customers(fitted_pipeline, sample_features.drop(columns=[column]))


def test_target_and_identifier_are_not_model_inputs(sample_features):
    """``Churn`` and ``customerID`` may travel with the batch but never become features."""
    batch = sample_features.copy()
    batch["customerID"] = [f"0000-TEST{i:02d}" for i in range(len(batch))]
    batch["Churn"] = "No"

    prepared = prepare_features(batch)

    assert list(prepared.columns) == [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "PaymentMethod",
    ]


def test_blank_total_charges_is_treated_as_missing_and_imputed(fitted_pipeline, sample_features):
    """Documented parsing rule: a whitespace-only numeric cell is missing, not an error.

    Risk covered: dropping the blank-handling branch would make the raw CSV raise
    ``ValueError`` instead of being imputed.
    """
    batch = sample_features.copy()
    batch.loc[0, "TotalCharges"] = "   "

    predictions = predict_customers(fitted_pipeline, batch)

    assert len(predictions) == len(batch)
    assert set(np.unique(predictions)) <= {0, 1}


def test_malformed_numeric_value_still_raises(fitted_pipeline, sample_features):
    """Only blank cells are tolerated; a real parse error must surface."""
    batch = sample_features.copy()
    # Keep the column as ``object`` so a corrupt CSV-like cell can be represented.
    batch["MonthlyCharges"] = batch["MonthlyCharges"].astype(object)
    batch.loc[1, "MonthlyCharges"] = "not-a-number"

    with pytest.raises(ValueError):
        predict_customers(fitted_pipeline, batch)


def test_prediction_returns_one_row_per_customer(fitted_pipeline, sample_features):
    batch = pd.concat([sample_features, sample_features], ignore_index=True)

    predictions = predict_customers(fitted_pipeline, batch)

    assert predictions.shape == (len(batch),)
    assert set(np.unique(predictions)) <= {0, 1}
