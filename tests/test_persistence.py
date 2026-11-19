"""Persistence: saving and reloading the fitted pipeline must not change behaviour."""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from telco_churn import prediction, training


def _numeric_imputer_statistics(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_branch = preprocessor.named_transformers_["num"]
    return numeric_branch.named_steps["imputer"].statistics_


def test_reload_preserves_predictions(tmp_path, fitted_pipeline, sample_features):
    """A saved pipeline must predict exactly the same after a round trip.

    Risk covered: serializing only the classifier (losing the fitted
    preprocessing) or an incompatible joblib pickling would change the output.
    """
    artifact = tmp_path / "churn_pipeline.joblib"
    before_predictions = fitted_pipeline.predict(sample_features)
    before_probabilities = fitted_pipeline.predict_proba(sample_features)

    joblib.dump(fitted_pipeline, artifact)
    reloaded = joblib.load(artifact)

    np.testing.assert_array_equal(reloaded.predict(sample_features), before_predictions)
    np.testing.assert_allclose(
        reloaded.predict_proba(sample_features),
        before_probabilities,
        rtol=0.0,
        atol=1e-12,
    )


def test_reload_keeps_the_learned_preprocessing_statistics(
    tmp_path, fitted_pipeline, sample_features
):
    """Reloading must not refit (or lose) the imputer statistics learned on training data."""
    artifact = tmp_path / "churn_pipeline.joblib"
    joblib.dump(fitted_pipeline, artifact)

    reloaded = joblib.load(artifact)

    np.testing.assert_allclose(
        _numeric_imputer_statistics(reloaded),
        _numeric_imputer_statistics(fitted_pipeline),
        rtol=0.0,
        atol=0.0,
    )


def test_find_latest_model_returns_the_most_recent_artifact(tmp_path, monkeypatch, fitted_pipeline):
    """Model discovery must follow modification time, not an arbitrary directory order."""
    older = tmp_path / "older.joblib"
    newer = tmp_path / "newer.joblib"
    joblib.dump(fitted_pipeline, older)
    joblib.dump(fitted_pipeline, newer)

    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_700_000_500, 1_700_000_500))
    monkeypatch.setattr(prediction, "MODEL_DIR", tmp_path)

    assert prediction.find_latest_model() == newer


def test_find_latest_model_explains_an_empty_directory(tmp_path, monkeypatch):
    """Without an artifact the error must name the directory instead of raising StopIteration."""
    monkeypatch.setattr(prediction, "MODEL_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="No trained model artifact"):
        prediction.find_latest_model()


def test_saved_model_metadata_records_the_contract(tmp_path, monkeypatch, fitted_pipeline):
    """``save_selected_model`` must write the artifact *and* the provenance record.

    Risk covered: dropping the metadata write (features, positive class, threshold,
    source hash) would leave an artifact nobody can audit.
    """
    fake_source = tmp_path / "telco.csv"
    fake_source.write_text("customerID\n0000-AAA\n", encoding="utf-8")
    monkeypatch.setattr(training, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(training, "resolve_source", lambda source=None: fake_source)
    monkeypatch.setattr(training, "compute_file_hash", lambda path: "0" * 64)

    metadata = training.save_selected_model(
        fitted_pipeline,
        "unit-test-model",
        {"average_precision": 0.5},
    )

    artifact = tmp_path / "models" / "unit-test-model.joblib"
    metadata_file = tmp_path / "models" / "unit-test-model_metadata.json"
    assert artifact.exists() and metadata_file.exists()

    stored = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert stored == metadata
    assert stored["feature_columns"] == [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "PaymentMethod",
    ]
    assert stored["positive_class"] == 1
    assert stored["threshold"] == 0.5
    assert stored["source_hash"] == "0" * 64

    reloaded = joblib.load(artifact)
    batch = pd.DataFrame(
        [
            {
                "tenure": 1,
                "MonthlyCharges": 20.0,
                "TotalCharges": 20.0,
                "Contract": "Month-to-month",
                "InternetService": "DSL",
                "PaymentMethod": "Mailed check",
            }
        ]
    )
    assert reloaded.predict(batch).shape == (1,)
