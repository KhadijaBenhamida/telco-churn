"""Tracked training entry point for TP03.

Runs the TP02 candidates (same six features, same split, same evaluation
protocol) as separate MLflow runs, exports the comparison table, selects the
candidate on validation average precision, then registers the selected fitted
pipeline under ``telco-churn`` with the ``candidate`` alias.

Usage (from the project root, tracking server running):

    uv run python scripts/train_mlflow.py --source "C:\\...\\datasets.csv"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from telco_churn import tracking
from telco_churn.dataset import PROJECT_ROOT, read_telco_data
from telco_churn.pipeline import FEATURES, build_pipeline, prepare_features
from telco_churn.training import SEED, make_split, prepare_target

REPORTS_DIR = PROJECT_ROOT / "reports"
HYPERPARAMETER_KEYS: dict[str, list[str]] = {
    "DummyClassifier": ["strategy"],
    "LogisticRegression": ["C", "penalty", "solver", "max_iter", "class_weight"],
    "DecisionTreeClassifier": ["criterion", "max_depth", "min_samples_leaf", "random_state"],
    "RandomForestClassifier": [
        "n_estimators",
        "criterion",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "random_state",
    ],
}


def candidate_estimators() -> dict[str, object]:
    """The three TP02 classifiers plus the majority baseline, unchanged settings."""
    return {
        "baseline-majority": DummyClassifier(strategy="most_frequent"),
        "logistic-C1": LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
        "tree-depth6": DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "forest-leaf5": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=SEED,
        ),
    }


def describe_hyperparameters(estimator) -> dict[str, object]:
    """Only the hyperparameters actually used by this candidate."""
    allowed = HYPERPARAMETER_KEYS.get(type(estimator).__name__, [])
    params = estimator.get_params()
    return {key: params[key] for key in allowed if key in params}


def reference_customers(valid_df: pd.DataFrame, count: int = 5) -> pd.DataFrame:
    """Deterministic small batch with complete numeric inputs, sorted by customerID."""
    usable = valid_df[valid_df["TotalCharges"].astype("string").str.strip().ne("")].copy()
    return usable.sort_values("customerID").head(count).reset_index(drop=True)


def to_native(value):
    """Convert numpy scalars so the reference batch is plain JSON."""
    return value.item() if hasattr(value, "item") else value


def run_candidates(
    source: str | Path | None,
    *,
    log_models: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict], dict[str, object], pd.DataFrame]:
    """Fit, evaluate and track every candidate; return comparison and artifacts."""
    frame = read_telco_data(source)
    train_df, valid_df, _test_df = make_split(frame)  # test set stays reserved
    X_train = prepare_features(train_df)
    X_valid = prepare_features(valid_df)
    y_train = prepare_target(train_df)
    y_valid = prepare_target(valid_df)
    input_example = X_train.head(3)  # unencoded feature columns, no Churn/customerID

    rows: list[dict] = []
    records: dict[str, dict] = {}
    fitted: dict[str, object] = {}

    for run_name, estimator in candidate_estimators().items():
        pipeline = build_pipeline(estimator)  # fresh preprocessing state per candidate
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        evaluation = tracking.evaluate_candidate(pipeline, X_valid, y_valid)
        model_type = type(estimator).__name__
        record = tracking.log_candidate_run(
            run_name=run_name,
            candidate=run_name,
            model_type=model_type,
            hyperparameters=describe_hyperparameters(estimator),
            fitted_pipeline=pipeline,
            train_df=train_df,
            valid_df=valid_df,
            evaluation=evaluation,
            fit_seconds=fit_seconds,
            input_example=input_example,
            source=source,
            log_model_artifact=log_models,
        )
        records[run_name] = record
        fitted[run_name] = pipeline
        rows.append(
            {
                "run_id": record["run_id"],
                "candidate": run_name,
                "model_type": model_type,
                **evaluation["metrics"],
                "fit_seconds": record["fit_seconds"],
                "model_uri": record["model_uri"],
            }
        )
        metrics = evaluation["metrics"]
        print(
            f"[tracked] {run_name:<18} run_id={record['run_id']} "
            f"AP={metrics['val_average_precision']:.4f} recall={metrics['val_recall']:.4f} "
            f"F1={metrics['val_f1']:.4f}",
            flush=True,
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["val_average_precision", "val_f1"],
        ascending=False,
        ignore_index=True,
    )
    comparison["decision"] = [
        "selected - highest validation AP" if position == 0 else "rejected"
        for position in range(len(comparison))
    ]
    return comparison, records, fitted, valid_df


def build_reference_predictions(model, reference_df: pd.DataFrame, record: dict) -> dict:
    """Predictions of the selected run on a fixed batch, for the fresh-process check."""
    features = prepare_features(reference_df)
    scores = tracking.positive_class_scores(model, features)
    predictions = (scores >= tracking.DECISION_THRESHOLD).astype(int)

    rows: list[dict] = []
    for position, (_, row) in enumerate(reference_df.iterrows()):
        entry = {"customerID": str(row["customerID"])}
        entry.update({column: to_native(row[column]) for column in FEATURES})
        entry["prediction"] = int(predictions[position])
        entry["probability"] = float(scores[position])
        rows.append(entry)

    return {
        "run_id": record["run_id"],
        "logged_model_uri": record["model_uri"],
        "candidate": record["run_name"],
        "threshold": tracking.DECISION_THRESHOLD,
        "features": FEATURES,
        "input_schema": "customerID + six unencoded feature columns, exactly as read from the CSV",
        "note": "class prediction = 1 when the class-1 probability is >= 0.5",
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track the TP02 churn candidates with MLflow and register the selected model."
    )
    parser.add_argument("--source", type=str, default=None, help="Path to the raw CSV dataset.")
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI.")
    parser.add_argument("--experiment", type=str, default=tracking.EXPERIMENT_NAME)
    parser.add_argument("--registered-name", type=str, default=tracking.REGISTERED_MODEL_NAME)
    parser.add_argument("--alias", type=str, default=tracking.MODEL_ALIAS)
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Log metrics/artifacts only (no logged model, no registration).",
    )
    args = parser.parse_args()

    experiment_id = tracking.configure_tracking(args.tracking_uri, args.experiment)
    print(f"Tracking URI : {args.tracking_uri or tracking.TRACKING_URI}")
    print(f"Experiment   : {args.experiment} (id={experiment_id})")

    comparison, records, fitted, valid_df = run_candidates(
        args.source, log_models=not args.skip_models
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = REPORTS_DIR / "tp03_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    selected = comparison.iloc[0].to_dict()
    selected_name = str(selected["candidate"])
    selected_record = records[selected_name]

    registry: dict = {}
    if not args.skip_models:
        registry = tracking.register_with_alias(
            selected_record["model_uri"],
            name=args.registered_name,
            alias=args.alias,
            run_id=selected_record["run_id"],
            description=(
                f"TP03 selection: {selected_name} "
                f"(validation AP {selected['val_average_precision']:.4f})"
            ),
        )

    reference_df = reference_customers(valid_df)
    reference_payload = build_reference_predictions(
        fitted[selected_name], reference_df, selected_record
    )
    reference_path = tracking.write_json(
        REPORTS_DIR / "tp03_reference_predictions.json", reference_payload
    )

    summary = {
        "tracking": tracking.tracking_snapshot(args.tracking_uri),
        "experiment": {"name": args.experiment, "experiment_id": experiment_id},
        "selection_rule": {
            "primary_metric": tracking.EVALUATION_RULES["primary_metric"],
            "tie_break": "validation F1",
            "evaluated_on": "validation split only; test split reserved",
            "decided_before_results": True,
        },
        "selected": {
            "candidate": selected_name,
            "run_id": selected_record["run_id"],
            "logged_model_uri": selected_record["model_uri"],
            "metrics": selected_record["metrics"],
            "confusion_matrix_rows_true_columns_predicted": selected_record["confusion_matrix"],
            "git_commit": selected_record["git_commit"],
            "data_sha256": selected_record["data_sha256"],
        },
        "registry": registry,
        "candidates": [
            {
                "run_id": row["run_id"],
                "candidate": row["candidate"],
                "model_type": row["model_type"],
                "val_average_precision": row["val_average_precision"],
                "val_precision": row["val_precision"],
                "val_recall": row["val_recall"],
                "val_f1": row["val_f1"],
                "val_accuracy": row["val_accuracy"],
                "fit_seconds": row["fit_seconds"],
                "decision": row["decision"],
            }
            for row in comparison.to_dict(orient="records")
        ],
        "candidate_model_uris": {name: item["model_uri"] for name, item in records.items()},
        "reference_predictions_file": str(reference_path.relative_to(PROJECT_ROOT)),
        "limitations": [
            "Local tracking storage only: run IDs and models:/ URIs do not resolve on another machine.",
            "One fixed split: the comparison is not proof of a statistically significant improvement.",
            "A hash identifies the CSV bytes; it cannot rebuild the file.",
            "An alias is a mutable reference, not a copy and not a deployment.",
        ],
    }
    selection_path = tracking.write_json(REPORTS_DIR / "tp03_selection.json", summary)

    print()
    print(
        comparison[
            [
                "candidate",
                "val_average_precision",
                "val_precision",
                "val_recall",
                "val_f1",
                "decision",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Comparison CSV   : {comparison_path}")
    print(f"Selection JSON   : {selection_path}")
    print(f"Reference batch  : {reference_path}")
    for name, item in records.items():
        print(f"  run {name:<18} {item['run_id']}  model_uri={item['model_uri']}")
    if registry:
        print()
        print(
            f"Registered model : {registry['registered_model_name']} version {registry['version']}"
        )
        print(
            f"Alias            : {registry['alias']} -> version {registry['alias_now_points_to_version']}"
        )
        print(f"Alias model URI  : {registry['alias_model_uri']}")
    print()
    print(f"Git commit       : {selected_record['git_commit']}")
    print(f"Data SHA-256     : {selected_record['data_sha256']}")


if __name__ == "__main__":
    main()
