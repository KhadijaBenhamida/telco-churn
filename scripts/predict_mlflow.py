"""Fresh-process inference entry point for TP03.

Loads the registered pipeline through the ``candidate`` alias, predicts the same
small batch that the selected run recorded, and compares the results. Nothing is
fitted here: the process only parses inputs, loads the model and predicts.

Usage (new terminal, tracking server still running):

    uv run python scripts/predict_mlflow.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from telco_churn import tracking
from telco_churn.dataset import PROJECT_ROOT
from telco_churn.pipeline import FEATURES, prepare_features
from telco_churn.prediction import predict_customers

REPORTS_DIR = PROJECT_ROOT / "reports"


def describe_alias(model_uri: str) -> dict:
    """Resolve ``models:/<name>@<alias>`` through the registry, when applicable."""
    if not model_uri.startswith("models:/") or "@" not in model_uri:
        return {}
    name, alias = model_uri[len("models:/") :].split("@", maxsplit=1)
    try:
        version = MlflowClient().get_model_version_by_alias(name, alias)
    except Exception as error:  # noqa: BLE001 - evidence script reports instead of crashing
        return {"registered_model_name": name, "alias": alias, "resolution_error": str(error)}
    return {
        "registered_model_name": name,
        "alias": alias,
        "version": version.version,
        "version_status": version.status,
        "source_run_id": version.source,
        "resolved_alias_uri": f"models:/{name}@{alias}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load the registered churn pipeline through its alias and compare predictions."
    )
    parser.add_argument("--model-uri", type=str, default=tracking.ALIAS_MODEL_URI)
    parser.add_argument(
        "--reference",
        type=str,
        default=str(REPORTS_DIR / "tp03_reference_predictions.json"),
        help="Predictions recorded by the selected training run.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPORTS_DIR / "tp03_inference_check.json"),
        help="Where to write the agreement evidence.",
    )
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    tracking_uri = args.tracking_uri or tracking.TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Tracking URI : {tracking_uri}")
    print(f"Model URI    : {args.model_uri}")

    alias_info = describe_alias(args.model_uri)
    if alias_info:
        print(f"Alias        : {alias_info.get('alias')} -> version {alias_info.get('version')}")

    try:
        model_info = tracking.model_info_dict(args.model_uri)
    except Exception as error:  # noqa: BLE001
        model_info = {"model_uri": args.model_uri, "info_error": str(error)}

    pipeline = mlflow.sklearn.load_model(args.model_uri)  # no fit() in this process
    print(f"Loaded       : {type(pipeline).__name__}")

    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    batch = pd.DataFrame(reference["records"])[["customerID", *FEATURES]]
    predictions = predict_customers(pipeline, batch)
    probabilities = tracking.positive_class_scores(pipeline, prepare_features(batch))

    expected_predictions = [int(record["prediction"]) for record in reference["records"]]
    expected_probabilities = [float(record["probability"]) for record in reference["records"]]
    agreement = [
        int(actual) == expected
        for actual, expected in zip(predictions, expected_predictions, strict=True)
    ]
    probability_deltas = [
        abs(float(actual) - expected)
        for actual, expected in zip(probabilities, expected_probabilities, strict=True)
    ]

    rows = []
    for position, record in enumerate(reference["records"]):
        rows.append(
            {
                "customerID": record["customerID"],
                "recorded_prediction": expected_predictions[position],
                "reloaded_prediction": int(predictions[position]),
                "agree": agreement[position],
                "recorded_probability": round(expected_probabilities[position], 6),
                "reloaded_probability": round(float(probabilities[position]), 6),
                "absolute_delta": round(probability_deltas[position], 12),
            }
        )

    evidence = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tracking_uri": tracking_uri,
        "model_uri": args.model_uri,
        "alias": alias_info,
        "logged_model": model_info,
        "reference_file": str(Path(args.reference)),
        "reference_run_id": reference.get("run_id"),
        "reference_candidate": reference.get("candidate"),
        "threshold": reference.get("threshold", tracking.DECISION_THRESHOLD),
        "compared_rows": len(rows),
        "class_predictions_agree": all(agreement),
        "max_absolute_probability_delta": round(max(probability_deltas), 12),
        "probability_tolerance": args.tolerance,
        "probabilities_within_tolerance": max(probability_deltas) <= args.tolerance,
        "fresh_process": True,
        "fit_called_in_this_process": False,
        "rows": rows,
    }
    output_path = tracking.write_json(args.output, evidence)

    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print(f"Class predictions agree : {all(agreement)}")
    print(f"Max |probability delta| : {max(probability_deltas):.3e} (tolerance {args.tolerance:g})")
    print(f"Evidence written to     : {output_path}")

    if not all(agreement) or max(probability_deltas) > args.tolerance:
        raise SystemExit("Reloaded predictions disagree with the recorded run.")
    print("RESULT: reloaded pipeline reproduces the selected run on the same batch.")


if __name__ == "__main__":
    main()
