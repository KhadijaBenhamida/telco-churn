from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from telco_churn.dataset import read_telco_data
from telco_churn.pipeline import FEATURES, prepare_features
from telco_churn.prediction import load_model, predict_customers
from telco_churn.training import make_split, prepare_target, train_and_select

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco.csv"


def test_data_split_has_no_customer_overlap():
    frame = read_telco_data(DATA_PATH)
    train_df, valid_df, test_df = make_split(frame)
    train_ids = set(train_df["customerID"])
    valid_ids = set(valid_df["customerID"])
    test_ids = set(test_df["customerID"])
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)


def test_pipeline_handles_numeric_parsing_and_missing_required_columns():
    frame = read_telco_data(DATA_PATH)
    prepared = prepare_features(frame)
    assert set(FEATURES) == set(prepared.columns)

    bad_frame = frame.drop(columns=["Contract"])
    with pytest.raises(KeyError):
        prepare_features(bad_frame)


def test_selected_model_reload_is_consistent():
    comparison, metadata, selected_model = train_and_select(DATA_PATH)
    assert not comparison.empty
    assert metadata["model_name"] in comparison["model"].values

    loaded_model = load_model(metadata["artifact"])
    frame = read_telco_data(DATA_PATH)
    _, valid_df, _ = make_split(frame)
    original = selected_model.predict(prepare_features(valid_df))
    reloaded = loaded_model.predict(prepare_features(valid_df))
    assert (original == reloaded).all()
