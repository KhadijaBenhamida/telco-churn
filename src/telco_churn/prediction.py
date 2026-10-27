from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .dataset import PROJECT_ROOT
from .pipeline import FEATURES, prepare_features

MODEL_DIR = PROJECT_ROOT / "models"


def find_latest_model() -> Path:
    candidates = sorted(MODEL_DIR.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No trained model artifact found in {MODEL_DIR}.")
    return candidates[0]


def load_model(model_path: str | Path | None = None):
    path = Path(model_path) if model_path is not None else find_latest_model()
    return joblib.load(path)


def predict_customers(model, customers: pd.DataFrame) -> np.ndarray:
    prepared = prepare_features(customers)
    missing = [column for column in FEATURES if column not in customers.columns]
    if missing:
        raise KeyError(f"Missing required input columns: {missing}")
    return model.predict(prepared)


__all__ = ["find_latest_model", "load_model", "predict_customers"]
