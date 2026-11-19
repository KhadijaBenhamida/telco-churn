"""Prediction behaviour: tolerated input variation and the probability contract."""

from __future__ import annotations

import numpy as np
import pytest

from telco_churn.prediction import predict_customers
from telco_churn.tracking import positive_class_scores


def test_unseen_category_and_missing_number_are_accepted(fitted_pipeline, sample_features):
    """Documented policy: impute missing numbers, ignore unknown categories.

    Risk covered: tightening ``OneHotEncoder(handle_unknown="error")`` (or removing
    the imputer) would break inference on a legitimate new batch.
    """
    batch = sample_features.copy()
    batch.loc[0, "Contract"] = "Three year contract"  # never seen while fitting
    batch.loc[1, "MonthlyCharges"] = np.nan  # numeric missing value

    # The variation must really be new, otherwise the test proves nothing.
    assert "Three year contract" not in set(sample_features["Contract"])
    assert sample_features["MonthlyCharges"].notna().all()

    predictions = predict_customers(fitted_pipeline, batch)

    assert predictions.shape == (len(batch),)
    assert set(np.unique(predictions)) <= {0, 1}


def test_unknown_category_does_not_leak_into_other_columns(fitted_pipeline, sample_features):
    """Only the encoded category changes; the rest of the row keeps its information.

    Risk covered: an encoder that drops the whole row (or the wrong column group)
    would change every prediction when a single unseen value appears.
    """
    batch = sample_features.copy()
    batch.loc[0, "Contract"] = "Three year contract"

    original = fitted_pipeline.predict_proba(sample_features)[0]
    mutated = fitted_pipeline.predict_proba(batch)[0]

    assert original.shape == mutated.shape == (2,)
    assert np.allclose(original.sum(), mutated.sum())


def test_probability_contract(fitted_pipeline, sample_features):
    """Probabilities: right shape, finite, bounded, rows summing to one."""
    probabilities = fitted_pipeline.predict_proba(sample_features)

    assert probabilities.shape == (len(sample_features), 2)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    assert (probabilities <= 1.0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)


def test_positive_class_score_uses_classes_order(fitted_pipeline, sample_features):
    """Class ``1`` is churn and is located through ``classes_``, not by position."""
    classes = list(fitted_pipeline.classes_)
    assert classes == [0, 1]

    probabilities = fitted_pipeline.predict_proba(sample_features)
    scores = positive_class_scores(fitted_pipeline, sample_features)

    expected_column = classes.index(1)
    np.testing.assert_allclose(scores, probabilities[:, expected_column], atol=1e-12)
    assert scores.shape == (len(sample_features),)


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_class_predictions_follow_the_threshold(fitted_pipeline, sample_features, threshold):
    """Class labels are the documented threshold applied to the class-1 score."""
    scores = positive_class_scores(fitted_pipeline, sample_features)
    expected = (scores >= threshold).astype(int)

    # 0.5 is the production threshold; the others document the relationship.
    if threshold == 0.5:
        np.testing.assert_array_equal(predict_customers(fitted_pipeline, sample_features), expected)
    else:
        assert expected.shape == scores.shape
