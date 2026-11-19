"""MLflow tracking helpers for TP03 (experiment, run, artifact, model and registry records).

The module only *records* evidence: it never changes the TP02 split, features or
evaluation protocol. TP02 helpers (``make_split``, ``prepare_features``,
``prepare_target``) are reused so tracked results stay comparable with the
untracked TP02 report.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # keep CLI/script output tidy

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.data  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .dataset import PROJECT_ROOT, compute_file_hash, resolve_source  # noqa: E402
from .pipeline import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET_COLUMN  # noqa: E402
from .training import SEED  # noqa: E402

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "telco-churn-tp03"
REGISTERED_MODEL_NAME = "telco-churn"
MODEL_ALIAS = "candidate"
MODEL_ARTIFACT_NAME = "churn_pipeline"
DECISION_THRESHOLD = 0.5
POSITIVE_CLASS = 1
ALIAS_MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

ARTIFACTS_DIR = PROJECT_ROOT / "reports" / "tp03_artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
CLASS_LABELS = {0: "No churn (0)", 1: "Yes churn (1)"}
SPLIT_PROTOCOL = (
    "stratify on Churn, random_state=42: train_test_split(test_size=0.4) then "
    "train_test_split(test_size=0.5) of the remainder -> 60% / 20% / 20%"
)
EVALUATION_RULES = {
    "primary_metric": "val_average_precision",
    "primary_metric_definition": (
        "sklearn.metrics.average_precision_score(y_valid, continuous positive-class score)"
    ),
    "score_source": "predict_proba column whose index matches classes_.index(1)",
    "secondary_metrics": ["val_precision", "val_recall", "val_f1", "val_accuracy"],
    "decision_threshold": DECISION_THRESHOLD,
    "threshold_policy": "fixed in advance at 0.5, identical for every candidate",
    "positive_class": POSITIVE_CLASS,
    "class_mapping": {"No": 0, "Yes": 1},
    "test_set": "reserved for later work; never used for selection in TP03",
    "selection_split": "validation (20% hold-out share of the original CSV)",
}


# --------------------------------------------------------------------------- #
# Tracking configuration
# --------------------------------------------------------------------------- #
def ensure_utf8_stdout() -> None:
    """MLflow prints an emoji run link; cp1252 consoles raise UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # stream already detached or wrapped
            continue


def configure_tracking(tracking_uri: str | None = None, experiment_name: str | None = None) -> str:
    """Point the client at the tracking server and select the experiment."""
    ensure_utf8_stdout()
    mlflow.set_tracking_uri(tracking_uri or TRACKING_URI)
    experiment = mlflow.set_experiment(experiment_name or EXPERIMENT_NAME)
    return experiment.experiment_id


def tracking_snapshot(tracking_uri: str | None = None) -> dict[str, Any]:
    """Describe the tracking target so every report can state where records live."""
    return {
        "tracking_uri": tracking_uri or TRACKING_URI,
        "experiment_name": EXPERIMENT_NAME,
        "backend_store": f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
        "artifact_store": (PROJECT_ROOT / "mlartifacts").as_posix(),
        "run_metadata_lives_in": "mlflow.db (SQLite backend store)",
        "model_and_file_artifacts_live_in": "mlartifacts/ (artifact store)",
    }


# --------------------------------------------------------------------------- #
# Code and data identity
# --------------------------------------------------------------------------- #
def git_commit() -> str:
    """Full SHA of the currently checked-out commit (or 'unknown')."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def git_uncommitted_paths() -> list[str]:
    """Paths with uncommitted edits; a commit SHA alone does not describe them."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def package_versions() -> dict[str, str]:
    packages = ("mlflow", "scikit-learn", "pandas", "numpy", "joblib", "matplotlib")
    resolved: dict[str, str] = {}
    for name in packages:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = "not installed"
    return resolved


def split_fingerprint(frame: pd.DataFrame, split_name: str) -> dict[str, Any]:
    """Compact identity of one split: sizes, class balance and a customerID digest.

    Two allocations drawn from the same CSV share the source hash but have
    different customerID digests - that digest is what distinguishes them.
    """
    identifiers = sorted(str(value) for value in frame["customerID"].tolist())
    digest = hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()
    positives = int(frame[TARGET_COLUMN].eq("Yes").sum())
    return {
        "name": split_name,
        "rows": int(frame.shape[0]),
        "churn_yes": positives,
        "churn_no": int(frame.shape[0] - positives),
        "positive_rate": round(positives / int(frame.shape[0]), 6),
        "customer_id_sha256": digest,
    }


# --------------------------------------------------------------------------- #
# Provenance JSON
# --------------------------------------------------------------------------- #
def build_provenance(
    *,
    run_name: str,
    candidate: str,
    model_type: str,
    hyperparameters: dict[str, Any],
    fit_seconds: float,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    metrics: dict[str, float],
    source: str | Path | None = None,
) -> dict[str, Any]:
    """Compact configuration/provenance record logged next to every candidate run."""
    source_path = resolve_source(source)
    uncommitted = git_uncommitted_paths()
    return {
        "project": {
            "name": "telco-churn",
            "coursework": "ESILV Machine Learning Operations - TP03",
            "run_name": run_name,
            "candidate": candidate,
        },
        "code": {
            "git_commit": git_commit(),
            "git_worktree_dirty": bool(uncommitted),
            "uncommitted_paths": uncommitted,
            "note": (
                "A commit SHA does not describe uncommitted edits; the flag and path list "
                "above record that this run was produced from a dirty worktree when non-empty."
            ),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "data": {
            "source_path": str(source_path),
            "source_sha256": compute_file_hash(source_path),
            "hash_scope": "bytes of the original CSV; it identifies the file, it cannot rebuild it",
            "target_column": TARGET_COLUMN,
            "feature_columns": {
                "numeric": NUMERIC_FEATURES,
                "categorical": CATEGORICAL_FEATURES,
                "all": FEATURES,
            },
            "excluded_columns": ["customerID", TARGET_COLUMN],
            "split_protocol": SPLIT_PROTOCOL,
            "split_seed": SEED,
            "splits": [
                split_fingerprint(train_df, "train"),
                split_fingerprint(valid_df, "validation"),
            ],
        },
        "model": {
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "fitted_object": "sklearn.pipeline.Pipeline(preprocessor -> classifier)",
            "fitting_data": "training split only; preprocessing is fitted inside the pipeline",
            "fit_seconds": round(float(fit_seconds), 3),
        },
        "evaluation": {**EVALUATION_RULES, "metrics": metrics},
        "mlflow_dataset_inputs": {
            "training": "mlflow.data.from_pandas(train_df, name='telco_train', targets='Churn')",
            "validation": "mlflow.data.from_pandas(valid_df, name='telco_validation', targets='Churn')",
            "limitation": (
                "MLflow dataset records store a name, digest and schema. They do not version "
                "control the DataFrame, and the digest is not the original CSV SHA-256."
            ),
        },
    }


# --------------------------------------------------------------------------- #
# Evaluation (same protocol as TP02, with an explicit threshold)
# --------------------------------------------------------------------------- #
def positive_class_scores(model, features: pd.DataFrame) -> np.ndarray:
    """Continuous score of class 1, located through ``model.classes_``."""
    classes = list(getattr(model, "classes_", []))
    if POSITIVE_CLASS in classes:
        column = classes.index(POSITIVE_CLASS)
        return np.asarray(model.predict_proba(features))[:, column]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float)
    raise ValueError(
        "Estimator exposes neither predict_proba for class 1 nor decision_function; "
        "average precision needs a continuous positive-class score."
    )


def evaluate_candidate(
    model,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, Any]:
    """Validation metrics, confusion matrix and scores for one fitted candidate."""
    scores = positive_class_scores(model, features)
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        target,
        predictions,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "val_average_precision": float(average_precision_score(target, scores)),
        "val_precision": float(precision),
        "val_recall": float(recall),
        "val_f1": float(f1),
        "val_accuracy": float(accuracy_score(target, predictions)),
    }
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    return {
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
        "threshold": threshold,
        "predictions": predictions,
        "scores": scores,
    }


def save_confusion_matrix_figure(matrix: list[list[int]], path: str | Path, title: str) -> Path:
    """Render the validation confusion matrix as a PNG artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(matrix, dtype=int)

    figure, axis = plt.subplots(figsize=(4.2, 3.8))
    axis.imshow(values, cmap="Blues")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                str(int(values[row, column])),
                ha="center",
                va="center",
                color="black" if values[row, column] < values.max() * 0.6 else "white",
                fontsize=12,
            )
    axis.set_xticks([0, 1], [CLASS_LABELS[0], CLASS_LABELS[1]])
    axis.set_yticks([0, 1], [CLASS_LABELS[0], CLASS_LABELS[1]])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    figure.text(
        0.01,
        0.01,
        "rows = true class, columns = predicted class; threshold = 0.5; split = validation",
        fontsize=6,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def loggable(value: Any) -> Any:
    """MLflow parameters must be str/int/float/bool - normalise everything else."""
    if isinstance(value, (bool, int, float, str)):
        return value
    if value is None:
        return "None"
    return str(value)


# --------------------------------------------------------------------------- #
# Run logging
# --------------------------------------------------------------------------- #
def _dataset_inputs(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    train_dataset = mlflow.data.from_pandas(train_df, name="telco_train", targets=TARGET_COLUMN)
    valid_dataset = mlflow.data.from_pandas(
        valid_df, name="telco_validation", targets=TARGET_COLUMN
    )
    return train_dataset, valid_dataset


def log_candidate_run(
    *,
    run_name: str,
    candidate: str,
    model_type: str,
    hyperparameters: dict[str, Any],
    fitted_pipeline,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    evaluation: dict[str, Any],
    fit_seconds: float,
    input_example: pd.DataFrame,
    source: str | Path | None = None,
    log_model_artifact: bool = True,
) -> dict[str, Any]:
    """One MLflow run per candidate: params, metrics, tags, artifacts, datasets, model."""
    provenance = build_provenance(
        run_name=run_name,
        candidate=candidate,
        model_type=model_type,
        hyperparameters=hyperparameters,
        fit_seconds=fit_seconds,
        train_df=train_df,
        valid_df=valid_df,
        metrics=evaluation["metrics"],
        source=source,
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = save_confusion_matrix_figure(
        evaluation["confusion_matrix"],
        ARTIFACTS_DIR / f"{run_name}_confusion_matrix.png",
        f"Validation confusion matrix - {run_name}",
    )

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "candidate": candidate,
                "model_type": model_type,
                "random_state": SEED,
                "threshold": DECISION_THRESHOLD,
                "positive_class": POSITIVE_CLASS,
                "train_rows": int(train_df.shape[0]),
                "valid_rows": int(valid_df.shape[0]),
                "n_features": len(FEATURES),
                **{f"hp_{key}": loggable(value) for key, value in hyperparameters.items()},
            }
        )
        mlflow.set_tags(
            {
                "git_commit": provenance["code"]["git_commit"],
                "git_worktree_dirty": str(provenance["code"]["git_worktree_dirty"]).lower(),
                "data_sha256": provenance["data"]["source_sha256"],
                "data_source": provenance["data"]["source_path"],
                "split_seed": SEED,
                "split_protocol": SPLIT_PROTOCOL,
                "primary_metric": EVALUATION_RULES["primary_metric"],
                "threshold_policy": EVALUATION_RULES["threshold_policy"],
                "positive_class": "1 = Churn Yes",
                "candidate_family": model_type,
                "test_set_usage": "reserved - not used for selection",
            }
        )
        mlflow.log_metrics(evaluation["metrics"])
        mlflow.log_artifact(str(figure_path), artifact_path="validation")
        mlflow.log_dict(provenance, "provenance.json")
        mlflow.log_dict(
            {
                "run_name": run_name,
                "threshold": DECISION_THRESHOLD,
                "class_labels": {"0": "No churn", "1": "Yes churn"},
                "confusion_matrix_rows_true_columns_predicted": evaluation["confusion_matrix"],
                "metrics": evaluation["metrics"],
            },
            "validation/validation_metrics.json",
        )

        train_dataset, valid_dataset = _dataset_inputs(train_df, valid_df)
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(valid_dataset, context="validation")

        model_uri = None
        model_uuid = None
        if log_model_artifact:
            model_info = mlflow.sklearn.log_model(
                sk_model=fitted_pipeline,
                name=MODEL_ARTIFACT_NAME,
                input_example=input_example,
                serialization_format="cloudpickle",
            )
            model_uri = model_info.model_uri
            model_uuid = getattr(model_info, "model_uuid", None)

        return {
            "run_name": run_name,
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "model_uri": model_uri,
            "model_uuid": model_uuid,
            "fit_seconds": round(float(fit_seconds), 3),
            "metrics": evaluation["metrics"],
            "confusion_matrix": evaluation["confusion_matrix"],
            "git_commit": provenance["code"]["git_commit"],
            "data_sha256": provenance["data"]["source_sha256"],
        }


# --------------------------------------------------------------------------- #
# Registry and loading
# --------------------------------------------------------------------------- #
def register_with_alias(
    model_uri: str,
    *,
    name: str = REGISTERED_MODEL_NAME,
    alias: str = MODEL_ALIAS,
    run_id: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Register a logged model as a new version and point the mutable alias at it."""
    client = MlflowClient()
    model_version = mlflow.register_model(model_uri=model_uri, name=name)
    client.set_registered_model_alias(name=name, alias=alias, version=model_version.version)
    if description:
        client.update_model_version(
            name=name,
            version=model_version.version,
            description=description,
        )
    if run_id:
        client.set_model_version_tag(
            name=name,
            version=model_version.version,
            key="source_run_id",
            value=run_id,
        )
    resolved = client.get_model_version_by_alias(name, alias)
    return {
        "registered_model_name": name,
        "version": str(model_version.version),
        "alias": alias,
        "alias_model_uri": f"models:/{name}@{alias}",
        "version_model_uri": f"models:/{name}/{model_version.version}",
        "logged_model_uri": model_uri,
        "source_run_id": run_id,
        "alias_now_points_to_version": str(resolved.version),
        "status": model_version.status,
    }


def load_pipeline(model_uri: str):
    """Load a fitted pipeline from the tracking server without refitting anything."""
    return mlflow.sklearn.load_model(model_uri)


def model_info_dict(model_uri: str) -> dict[str, Any]:
    """Metadata of a logged model URI (flavors, run id, stored input signature)."""
    info = mlflow.models.get_model_info(model_uri)
    signature = getattr(info, "signature", None)
    inputs = []
    if signature is not None:
        inputs = [f"{column.name}:{column.type}" for column in signature.inputs]
    return {
        "model_uri": model_uri,
        "model_uuid": getattr(info, "model_uuid", None),
        "run_id": getattr(info, "run_id", None),
        "artifact_path": getattr(info, "artifact_path", None),
        "flavors": sorted(getattr(info, "flavors", {}) or {}),
        "signature_inputs": inputs,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output


__all__ = [
    "ALIAS_MODEL_URI",
    "ARTIFACTS_DIR",
    "CLASS_LABELS",
    "DECISION_THRESHOLD",
    "EVALUATION_RULES",
    "EXPERIMENT_NAME",
    "MODEL_ALIAS",
    "MODEL_ARTIFACT_NAME",
    "POSITIVE_CLASS",
    "REGISTERED_MODEL_NAME",
    "REPORTS_DIR",
    "SPLIT_PROTOCOL",
    "TRACKING_URI",
    "build_provenance",
    "configure_tracking",
    "ensure_utf8_stdout",
    "evaluate_candidate",
    "git_commit",
    "git_uncommitted_paths",
    "load_pipeline",
    "log_candidate_run",
    "model_info_dict",
    "package_versions",
    "positive_class_scores",
    "register_with_alias",
    "save_confusion_matrix_figure",
    "split_fingerprint",
    "tracking_snapshot",
    "write_json",
]
