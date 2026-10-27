from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .dataset import PROJECT_ROOT, compute_file_hash, read_telco_data, resolve_source
from .pipeline import FEATURES, TARGET_COLUMN, build_pipeline, prepare_features

MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SEED = 42


def make_split(frame: pd.DataFrame, random_state: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, remaining_df = train_test_split(
        frame,
        test_size=0.4,
        stratify=frame[TARGET_COLUMN],
        random_state=random_state,
    )
    valid_df, test_df = train_test_split(
        remaining_df,
        test_size=0.5,
        stratify=remaining_df[TARGET_COLUMN],
        random_state=random_state,
    )
    return train_df, valid_df, test_df


def prepare_target(frame: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.Series:
    return frame[target_column].map({"No": 0, "Yes": 1})


def safe_probability_score(model, X_valid: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_valid)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_valid)
    # Majority baselines do not provide a positive-class score; this keeps metrics explicit.
    return np.zeros(len(X_valid), dtype=float)


def evaluate_model(model, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict:
    predictions = model.predict(X_valid)
    probabilities = safe_probability_score(model, X_valid)
    accuracy = accuracy_score(y_valid, predictions)
    ap = average_precision_score(y_valid, probabilities)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_valid,
        predictions,
        average="binary",
        zero_division=0,
    )
    matrix = confusion_matrix(y_valid, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy),
        "average_precision": float(ap),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": matrix.tolist(),
    }


def candidate_models() -> dict[str, object]:
    return {
        "baseline_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=SEED,
        ),
    }


def compare_models(source: str | Path | None = None) -> pd.DataFrame:
    frame = read_telco_data(source)
    train_df, valid_df, test_df = make_split(frame)
    X_train = prepare_features(train_df)
    X_valid = prepare_features(valid_df)
    y_train = prepare_target(train_df)
    y_valid = prepare_target(valid_df)

    rows: list[dict] = []
    for name, estimator in candidate_models().items():
        model = build_pipeline(estimator)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_valid, y_valid)
        rows.append({"model": name, **metrics})

    results = pd.DataFrame(rows)
    results = results.sort_values("average_precision", ascending=False, ignore_index=True)
    return results


def save_selected_model(model, model_name: str, metrics: dict, source: str | Path | None = None) -> dict:
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    model_path = MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    source_path = resolve_source(source)
    metadata = {
        "model_name": model_name,
        "feature_columns": FEATURES,
        "target_column": TARGET_COLUMN,
        "random_state": SEED,
        "positive_class": 1,
        "threshold": 0.5,
        "metrics": metrics,
        "source_hash": compute_file_hash(source_path),
        "artifact": str(model_path),
    }
    metadata_path = MODEL_DIR / f"{model_name}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def select_best_model(results: pd.DataFrame) -> tuple[str, dict]:
    best_row = results.iloc[0].to_dict()
    return str(best_row["model"]), {k: v for k, v in best_row.items() if k != "model"}


def train_and_select(source: str | Path | None = None) -> tuple[pd.DataFrame, dict, object]:
    frame = read_telco_data(source)
    train_df, valid_df, _ = make_split(frame)
    X_train = prepare_features(train_df)
    X_valid = prepare_features(valid_df)
    y_train = prepare_target(train_df)
    y_valid = prepare_target(valid_df)

    results: list[dict] = []
    fitted_models: list[tuple[str, object]] = []
    for name, estimator in candidate_models().items():
        model = build_pipeline(estimator)
        model.fit(X_train, y_train)
        fitted_models.append((name, model))
        metrics = evaluate_model(model, X_valid, y_valid)
        results.append({"model": name, **metrics})

    comparison = pd.DataFrame(results).sort_values("average_precision", ascending=False, ignore_index=True)
    selected_name, selected_metrics = select_best_model(comparison)
    selected_model = dict(fitted_models)[selected_name]
    metadata = save_selected_model(selected_model, selected_name, selected_metrics, source=source)
    return comparison, metadata, selected_model


__all__ = [
    "SEED",
    "candidate_models",
    "compare_models",
    "evaluate_model",
    "make_split",
    "prepare_target",
    "save_selected_model",
    "select_best_model",
    "train_and_select",
]
