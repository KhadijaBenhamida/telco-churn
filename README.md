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

## Verification

The repository includes a small validation suite:

```powershell
uv run pytest -q
```

The checks confirm:

- customer IDs do not overlap across splits
- required features are present and missing features fail clearly
- the selected pipeline reloads without changing predictions

## Handover note

This project is intended to be reusable by another engineer. The raw data is fixed, the environment is defined with `uv`, the inspection report is reproducible, and the model training workflow runs from a terminal entry point. The fitted artifact remains separate from the source code, which is the correct place for reusable training logic.
