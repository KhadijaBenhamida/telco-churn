"""TP03 section 1 smoke check: one experiment, one run, one parameter.

Confirms that the training client reaches the tracking server before the full
workflow is instrumented.

    uv run python scripts/tp03_smoke_run.py
"""

from __future__ import annotations

import argparse

import mlflow
from mlflow.tracking import MlflowClient

from telco_churn import tracking
from telco_churn.dataset import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a minimal MLflow run and read it back.")
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--experiment", type=str, default=tracking.EXPERIMENT_NAME)
    args = parser.parse_args()

    experiment_id = tracking.configure_tracking(args.tracking_uri, args.experiment)
    with mlflow.start_run(run_name="smoke-check") as run:
        mlflow.log_param("purpose", "tracking connectivity check")
        mlflow.log_param("threshold", tracking.DECISION_THRESHOLD)
        mlflow.log_metric("smoke_value", 1.0)
        mlflow.set_tag("stage", "tp03-section-1")
        run_id = run.info.run_id

    fetched = MlflowClient().get_run(run_id)
    snapshot = tracking.tracking_snapshot(args.tracking_uri)

    print(f"Tracking URI   : {snapshot['tracking_uri']}")
    print(f"Experiment     : {args.experiment}")
    print(f"Experiment ID  : {experiment_id}")
    print(f"Run ID         : {run_id}")
    print(f"Recorded params: {dict(fetched.data.params)}")
    print(f"Backend store  : {snapshot['backend_store']}")
    print(f"Artifact store : {snapshot['artifact_store']}")
    print(f"Database file exists : {(PROJECT_ROOT / 'mlflow.db').exists()}")
    print(f"UI             : {snapshot['tracking_uri']}/#/experiments/{experiment_id}/runs/{run_id}")


if __name__ == "__main__":
    main()
