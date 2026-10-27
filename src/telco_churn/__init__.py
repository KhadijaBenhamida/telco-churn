"""Telco churn project package."""

from .dataset import build_report, read_telco_data
from .pipeline import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, build_pipeline, prepare_features
from .prediction import load_model, predict_customers
from .training import compare_models, train_and_select

__all__ = [
    "FEATURES",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "build_pipeline",
    "prepare_features",
    "build_report",
    "read_telco_data",
    "load_model",
    "predict_customers",
    "compare_models",
    "train_and_select",
]
__version__ = "0.1.0"

