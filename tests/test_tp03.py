from __future__ import annotations

import json
from pathlib import Path

import pytest

from telco_churn import tracking
from telco_churn.dataset import read_telco_data
from telco_churn.pipeline import FEATURES, build_pipeline, prepare_features
from telco_churn.training import SEED, make_split, prepare_target

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco.csv"

# Only the tests that read the real CSV need the file; the provenance/registry checks do not.
requires_csv = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="the real Telco CSV is git-ignored; add data/raw/telco.csv to run this test",
)


@pytest.fixture(scope="module")
def small_split():
    frame = read_telco_data(DATA_PATH)
    train_df, valid_df, _ = make_split(frame)
    return train_df.head(400), valid_df.head(200)


@requires_csv
def test_evaluation_uses_validation_protocol_and_threshold(small_split):
    from sklearn.linear_model import LogisticRegression

    train_df, valid_df = small_split
    model = build_pipeline(LogisticRegression(max_iter=500, random_state=SEED))
    model.fit(prepare_features(train_df), prepare_target(train_df))

    result = tracking.evaluate_candidate(
        model, prepare_features(valid_df), prepare_target(valid_df)
    )

    assert set(result["metrics"]) == {
        "val_average_precision",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_accuracy",
    }
    assert all(isinstance(value, float) for value in result["metrics"].values())
    assert result["threshold"] == 0.5
    assert sum(sum(row) for row in result["confusion_matrix"]) == len(valid_df)


@requires_csv
def test_positive_class_scores_match_classes_index(small_split):
    from sklearn.linear_model import LogisticRegression

    train_df, valid_df = small_split
    model = build_pipeline(LogisticRegression(max_iter=500, random_state=SEED))
    model.fit(prepare_features(train_df), prepare_target(train_df))

    scores = tracking.positive_class_scores(model, prepare_features(valid_df))
    expected = model.predict_proba(prepare_features(valid_df))[:, list(model.classes_).index(1)]
    assert scores.tolist() == pytest.approx(expected.tolist())


@requires_csv
def test_provenance_records_identity_and_split_fingerprints(small_split):
    train_df, valid_df = small_split
    provenance = tracking.build_provenance(
        run_name="unit-test",
        candidate="unit-test",
        model_type="LogisticRegression",
        hyperparameters={"C": 1.0},
        fit_seconds=0.1,
        train_df=train_df,
        valid_df=valid_df,
        metrics={"val_average_precision": 0.5},
    )

    assert len(provenance["data"]["source_sha256"]) == 64
    assert provenance["data"]["feature_columns"]["all"] == FEATURES
    assert provenance["data"]["excluded_columns"] == ["customerID", "Churn"]
    assert provenance["evaluation"]["primary_metric"] == "val_average_precision"
    assert provenance["evaluation"]["decision_threshold"] == 0.5
    assert provenance["evaluation"]["positive_class"] == 1

    fingerprints = {split["name"]: split for split in provenance["data"]["splits"]}
    assert fingerprints["train"]["rows"] == len(train_df)
    assert fingerprints["validation"]["rows"] == len(valid_df)
    assert (
        fingerprints["train"]["customer_id_sha256"]
        != fingerprints["validation"]["customer_id_sha256"]
    )


@requires_csv
def test_split_fingerprint_changes_with_allocation():
    frame = read_telco_data(DATA_PATH)
    first_train, first_valid, _ = make_split(frame, random_state=42)
    second_train, second_valid, _ = make_split(frame, random_state=7)

    first = tracking.split_fingerprint(first_train, "train")
    second = tracking.split_fingerprint(second_train, "train")
    assert first["customer_id_sha256"] != second["customer_id_sha256"]
    assert len(first_valid) == len(second_valid)


def test_confusion_matrix_figure_is_written(tmp_path):
    output = tracking.save_confusion_matrix_figure(
        [[90, 10], [30, 20]], tmp_path / "matrix.png", "unit test"
    )
    assert output.exists()
    assert output.stat().st_size > 0


def test_tracking_constants_describe_handover_contract(tmp_path):
    assert tracking.EXPERIMENT_NAME == "telco-churn-tp03"
    assert tracking.REGISTERED_MODEL_NAME == "telco-churn"
    assert tracking.MODEL_ALIAS == "candidate"
    assert tracking.ALIAS_MODEL_URI == "models:/telco-churn@candidate"
    assert tracking.DECISION_THRESHOLD == 0.5

    snapshot = tracking.tracking_snapshot()
    assert snapshot["tracking_uri"] == tracking.TRACKING_URI
    assert snapshot["backend_store"].endswith("mlflow.db")

    payload = tracking.write_json(tmp_path / "selection.json", {"run_id": "abc"})
    assert json.loads(payload.read_text(encoding="utf-8"))["run_id"] == "abc"


def test_candidate_configurations_cover_three_learned_models():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "train_mlflow.py"
    spec = importlib.util.spec_from_file_location("train_mlflow", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    candidates = module.candidate_estimators()
    learned = {name for name in candidates if name != "baseline-majority"}
    assert len(learned) >= 3
    assert set(module.describe_hyperparameters(candidates["logistic-C1"])) >= {"C", "max_iter"}


@requires_csv
def test_dataset_exposes_documented_feature_columns():
    frame = read_telco_data(DATA_PATH)
    prepared = prepare_features(frame)
    assert list(prepared.columns) == FEATURES
    assert "customerID" not in prepared.columns
    assert "Churn" not in prepared.columns
