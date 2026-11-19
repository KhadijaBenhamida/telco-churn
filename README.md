# Telco Churn project

This repository contains the Telco Customer Churn project for the ESILV MLOps TP01 and TP02 workflow. The objective is to keep the raw data immutable, document the source clearly, reproduce the inspection step, and train a reusable churn model with preprocessing and a saved artifact.

## Business objective

The task is to predict whether a customer will churn (`Churn = Yes` / `No`) from a customer snapshot. The project keeps the original CSV unchanged and treats the raw file as the authoritative source.

## Repository layout

- `data/raw/` — local copy of the source dataset used for project work
- `data/interim/`, `data/processed/` — reserved for future preparation steps
- `src/telco_churn/` — reusable Python package for dataset inspection, pipeline construction and training logic
- `scripts/inspect_data.py` — inspection entry point for TP01
- `scripts/train_model.py` — model training and comparison entry point for TP02
- `references/data_source.md` — dataset source, hash and provenance details
- `reports/` — generated inspection and comparison reports
- `models/` — saved fitted pipeline artifacts and metadata
- `evidence/tp01/` — environment comparison records before and after `uv sync --locked`
- `tests/` — focused checks for split integrity and reload behavior

## Source data provenance

The source file used in this project is:

`C:\Users\khadi\OneDrive\Desktop\MLops\datasets.csv`

Its SHA-256 hash is:

`62ca518b1ebbcd4c832c18c1a21629b03abe9033fef66fe2e38d8d030658a20c`

This file is not modified in a spreadsheet or editor. A local working copy lives in `data/raw/telco.csv`, but the external file remains the reference snapshot.

## Environment setup

The project uses Python 3.12 and `uv`.

```powershell
$env:Path = "$HOME\.local\bin;$env:Path"
cd "C:\Users\khadi\OneDrive\Desktop\MLops\TP1-2 MLOPS\telco-churn"
uv sync
```

Check that the package imports correctly:

```powershell
uv run python -c "import telco_churn; print(telco_churn.__file__)"
```

## TP01 — inspection and data provenance

Generate the inspection report:

```powershell
cd "C:\Users\khadi\OneDrive\Desktop\MLops\TP1-2 MLOPS\telco-churn"
uv run python scripts/inspect_data.py --source "C:\Users\khadi\OneDrive\Desktop\MLops\datasets.csv"
```

The command writes the JSON report to `reports/data_overview.json` and includes:

- dataset dimensions and field types
- target counts and proportions
- duplicate customer ID count
- missing values and whitespace-only checks
- source hash and runtime environment metadata

A locked recreation check is also included:

```powershell
uv sync --locked
uv run python scripts/inspect_data.py --source "C:\Users\khadi\OneDrive\Desktop\MLops\datasets.csv"
```

## TP02 — training workflow

Train and compare candidates with the shared preprocessing pipeline:

```powershell
cd "C:\Users\khadi\OneDrive\Desktop\MLops\TP1-2 MLOPS\telco-churn"
uv run python scripts/train_model.py --source "C:\Users\khadi\OneDrive\Desktop\MLops\datasets.csv"
```

This workflow:

- creates a stratified train/validation/test split
- uses a preprocessing pipeline with numeric and categorical branches
- compares baseline, logistic regression, decision tree, and random forest models
- stores the best fitted model and metadata in `models/`

The model metadata contains the chosen feature list, positive class definition, random seed, threshold, and evaluation metrics.

## Model artifact and inference

The selected artifact is saved in `models/` and can be reloaded in a fresh Python process:

```powershell
uv run python -c "import joblib; from telco_churn.prediction import load_model; from telco_churn.dataset import read_telco_data; from telco_churn.pipeline import prepare_features; from telco_churn.training import make_split; df = read_telco_data(r'C:\Users\khadi\OneDrive\Desktop\MLops\datasets.csv'); _, valid_df, _ = make_split(df); model = load_model(); preds = model.predict(prepare_features(valid_df)); print(preds[:10])"
```

This confirms that the artifact can be loaded and used without retraining.

## TP03 — track, compare and recover experiments with MLflow

TP03 adds experiment tracking to the TP02 workflow. The six features, the
stratified 60/20/20 customer allocation, the preprocessing pipeline and the
validation protocol are unchanged; only the recording layer is new.

### Install the tracking dependencies

```powershell
cd "C:\Users\khadi\OneDrive\Desktop\MLops\TP1-2 MLOPS\telco-churn"
uv add "mlflow>=3,<4" matplotlib
uv run mlflow --version
```

Two environment notes for this machine:

- `sqlalchemy` is pinned to `>=2.0,<2.1`. SQLAlchemy 2.1 ships compiled
  extensions (`*_cy.pyd`) that the local Windows application-control policy
  blocks; 2.0 falls back to its pure-Python implementation and starts normally.
  This is a server-side dependency only; MLflow accepts `sqlalchemy<3`.
- The `mlflow.exe` console script of a freshly installed package can be blocked
  by the same policy. The helper scripts therefore launch the server with
  `uv run python -m mlflow server ...`, the exact equivalent of
  `uv run mlflow server ...`.

### 1. Start the tracking server (terminal 1, keep it running)

```powershell
cd "C:\Users\khadi\OneDrive\Desktop\MLops\TP1-2 MLOPS\telco-churn"
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
```

Equivalent helper (background, writes `mlflow-server.log`):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_tracking_server.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/stop_tracking_server.ps1
```

Open <http://127.0.0.1:5000>. The experiment is `telco-churn-tp03`.

### 2. Recorded workflow (terminal 2)

```powershell
# section 1 check: one experiment, one run, one parameter, then read it back
uv run python scripts/tp03_smoke_run.py

# four tracked candidate runs + comparison + registration + reference batch
uv run python scripts/train_mlflow.py --source "C:\Users\khadi\OneDrive\Desktop\MLops\datasets.csv"

# fresh-process reload through the alias and prediction agreement check
uv run python scripts/predict_mlflow.py
```

`scripts/train_mlflow.py` creates one run per candidate
(`baseline-majority`, `logistic-C1`, `tree-depth6`, `forest-leaf5`), each with:

| Logged item | Content |
|---|---|
| Parameters | `model_type`, real hyperparameters (`hp_*`), `random_state`, `threshold`, `train_rows`, `valid_rows`, `n_features` |
| Metrics | `val_average_precision` (primary), `val_precision`, `val_recall`, `val_f1`, `val_accuracy` |
| Tags | `git_commit`, `git_worktree_dirty`, `data_sha256`, `split_protocol`, `primary_metric`, `threshold_policy`, `test_set_usage` |
| Artifacts | `validation/<run>_confusion_matrix.png`, `provenance.json`, `validation/validation_metrics.json` |
| Dataset inputs | `telco_train` (context `training`), `telco_validation` (context `validation`) |
| Model | the complete fitted `Pipeline(preprocessor -> classifier)` as `churn_pipeline` |

Generated evidence stays in the repository (small text files):

- `reports/tp03_comparison.csv` — the candidate table
- `reports/tp03_selection.json` — run IDs, model URIs, registry identifiers, limitations
- `reports/tp03_reference_predictions.json` — predictions of the selected run on five customers
- `reports/tp03_inference_check.json` — reload agreement evidence
- `reports/tp03_artifacts/` — local copies of the confusion-matrix PNGs
- `reports/tp03_summary.md` — the deliverable summary

### 3. Where the tracking data lives

| Component | Location | Holds |
|---|---|---|
| Backend store | `mlflow.db` (SQLite) | experiments, runs, params, metrics, tags, dataset records, model registry versions and aliases |
| Artifact store | `mlartifacts/` | model files, confusion matrices, `provenance.json` |
| Tracking server | `http://127.0.0.1:5000` | receives logging/query requests, serves the UI |
| Training client | `uv run python ...` | runs the training code and sends records to the server |

`mlflow.db` and `mlartifacts/` are ignored by Git but kept on disk for the demo.
Stopping the server never deletes them: restart it from the same project root
with the same `--backend-store-uri sqlite:///mlflow.db` and
`--artifacts-destination ./mlartifacts`, and every run, registered version and
alias is still there.

### 4. Handover identifiers and recovery

| Identifier | Question it answers |
|---|---|
| Run ID | which execution produced the result |
| Logged model URI (`runs:/<run_id>/churn_pipeline`) | which recorded model artifact to load |
| Registered name + version (`telco-churn`, e.g. `1`) | which version belongs to the managed model |
| Alias `candidate` | which version currently fills that role |

To recover the selected pipeline without running training again:

```powershell
uv run python scripts/predict_mlflow.py --model-uri "models:/telco-churn@candidate"
```

Values recorded for this submission (always re-derivable from
`reports/tp03_selection.json`):

- selected run `eb5c7ba6dcd64484996c312cd9dfeba3` (`forest-leaf5`, validation AP 0.6526)
- logged model `models:/m-477af62b21674c358b732911532f7a50`
- registered model `telco-churn` version `1`, alias `candidate`
- Git commit `91da09b8d5f9612f61b47768eabe80252133df54`, data SHA-256 `62ca518b…a20c`

The script loads the logged pipeline (no `fit()` call), predicts five documented
customers and compares them with `reports/tp03_reference_predictions.json`.
Class predictions must match exactly; probabilities are compared with tolerance
`1e-9`. A specific version can be pinned with `--model-uri "models:/telco-churn/<version>"`.

### 5. Known limitations

- Tracking is local: `mlflow.db`, `mlartifacts/` and `models:/telco-churn@candidate`
  only resolve on this machine. A classmate needs the database **and** the
  artifact directory (or a shared tracking server) plus a compatible
  environment (`uv sync --locked`).
- The CSV SHA-256 identifies the file bytes; it cannot rebuild the CSV. Keep the
  original file and the recorded per-split `customer_id_sha256` fingerprints
  (`provenance.json`) to distinguish two allocations of the same CSV.
- MLflow dataset records store a name, digest and schema; they are not dataset
  version control, and their digest differs from the original-file SHA-256.
- An alias is a mutable pointer, not a copy and not a deployment. A process that
  already loaded the model keeps its in-memory object; it must load again to see
  a new version.
- One fixed split: the comparison is a classroom comparison, not a statistical
  significance test.

## Verification


```powershell
uv run pytest -q
```

The checks confirm:

- customer IDs do not overlap across splits
- required features are present and missing features fail clearly
- the selected pipeline reloads without changing predictions
