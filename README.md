# Telco Churn project

This project implements the TP01 reproducible ML setup for the IBM Telco Customer Churn dataset. The goal is to make the source dataset, inspection flow, and report generation easy to understand, reproduce, and hand over to another engineer.

## Business question

We are trying to predict whether a customer will churn (`Churn = Yes` or `No`) based on a snapshot of account and service information. The project intentionally stops at data inspection and provenance in TP01; later work will add preprocessing and a modeling pipeline.

## Project structure

- `data/raw/` : local raw dataset copy used in the project (git-ignored to keep the authoritative copy outside Git)
- `data/interim/` and `data/processed/` : reserved for later pipeline stages
- `src/telco_churn/` : importable Python package containing reusable inspection logic
- `scripts/inspect_data.py` : CLI entry point for generating the JSON report
- `references/data_source.md` : provenance details and SHA-256 hash for the source snapshot
- `reports/data_overview.json` : generated inspection report
- `evidence/tp01/` : environment and reproduction records

## Source dataset and provenance

The authoritative dataset used for this project is:

`C:\Users\khadi\OneDrive\Desktop\MLops\datasets.csv`

The byte identity is recorded in `references/data_source.md` and the SHA-256 hash is:

`62ca518b1ebbcd4c832c18c1a21629b03abe9033fef66fe2e38d8d030658a20c`

The file is intentionally not edited in a spreadsheet tool. A local copy is stored in `data/raw/telco.csv` for convenience, but the original file outside the Git working tree remains the source of truth.

## Environment setup

This project uses `uv` and Python 3.12.

```powershell
$env:Path = "$HOME\.local\bin;$env:Path"
cd "C:\Users\khadi\OneDrive\Desktop\MLops\TP1-2 MLOPS\telco-churn"
uv sync
```

Then confirm the package imports:

```powershell
uv run python -c "import telco_churn; print(telco_churn.__file__)"
```

## Inspection command

Generate the project inspection report:

```powershell
cd "C:\Users\khadi\OneDrive\Desktop\MLops\TP1-2 MLOPS\telco-churn"
uv run python scripts/inspect_data.py --source "C:\Users\khadi\OneDrive\Desktop\MLops\datasets.csv"
```

This saves the machine-readable result to `reports/data_overview.json` and records the dataset hash, field types, target distribution, duplicate customer IDs, and missing-value checks.

## What the report includes

- Dataset dimensions and dtypes
- Churn target counts and proportions
- Number of duplicate `customerID` values
- Missing-value counts for each field
- Explicit whitespace-only checks after `str.strip()`
- Source hash and interpreter/package metadata

## Reproduction check

Once the lock file is present, the environment can be recreated without changing the dependency selection:

```powershell
uv sync --locked
uv run python scripts/inspect_data.py --source "C:\Users\khadi\OneDrive\Desktop\MLops\datasets.csv"
```

The project keeps environment records in `evidence/tp01/` so a teammate can compare the interpreter and package versions before and after recreation.

## Notes for TP02

This TP stops at inspection. Later work should add preprocessing, train/validation logic, a fitted model artifact under `models/`, and tests for save/reload behavior. The code that builds the model belongs in the package source tree; the fitted pipeline belongs in `models/`.
