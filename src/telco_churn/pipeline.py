from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .dataset import PROJECT_ROOT, read_telco_data

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "Churn"


def clean_telco_numeric_columns(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Convert numeric columns to numeric dtypes while preserving malformed nonblank values as visible errors."""
    result = frame.copy()
    target_columns = columns or NUMERIC_FEATURES
    for column in target_columns:
        if column not in result.columns:
            continue
        series = result[column].astype("string").str.strip()
        blanks = series.eq("")
        if blanks.any():
            result.loc[blanks, column] = pd.NA
        result[column] = pd.to_numeric(series, errors="raise")
    return result


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(classifier):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with the documented feature schema and validated numeric types."""
    validated = frame.copy()
    for column in ["customerID", "Churn"]:
        if column in validated.columns:
            validated = validated.drop(columns=[column])
    validated = clean_telco_numeric_columns(validated, columns=["tenure", "MonthlyCharges", "TotalCharges"])
    missing = [feature for feature in FEATURES if feature not in validated.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    return validated[FEATURES]


def load_training_data(source: str | Path | None = None) -> pd.DataFrame:
    frame = read_telco_data(source)
    return clean_telco_numeric_columns(frame, columns=["tenure", "MonthlyCharges", "TotalCharges"])


__all__ = [
    "FEATURES",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "TARGET_COLUMN",
    "build_pipeline",
    "build_preprocessor",
    "clean_telco_numeric_columns",
    "prepare_features",
    "load_training_data",
]
