from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COPY = PROJECT_ROOT / "data" / "raw" / "telco.csv"
FALLBACK_SOURCE = Path(r"C:\Users\khadi\OneDrive\Desktop\MLops\datasets.csv")


def resolve_source(source: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if source is not None:
        candidates.append(Path(source))
    candidates.extend([DEFAULT_COPY, FALLBACK_SOURCE])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "No source dataset was found. Provide --source or place the dataset at data/raw/telco.csv."
    )


def normalize_quoted_csv(raw_text: str) -> str:
    lines = [line.strip().strip('"') for line in raw_text.splitlines() if line.strip()]
    return "\n".join(lines)


def read_telco_data(source: str | Path | None = None) -> pd.DataFrame:
    path = resolve_source(source)
    raw_text = path.read_text(encoding="utf-8-sig")
    cleaned_text = normalize_quoted_csv(raw_text)
    df = pd.read_csv(StringIO(cleaned_text))
    df.columns = [str(col).strip() for col in df.columns]
    return df


def compute_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(source: str | Path | None = None, output_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_source(source)
    df = read_telco_data(path)
    target_counts = df["Churn"].value_counts(dropna=False)
    target_proportions = df["Churn"].value_counts(dropna=False, normalize=True)

    string_df = df.astype("string")
    whitespace_only_by_column = {
        column: int(string_df[column].str.strip().eq("").sum())
        for column in string_df.columns
    }
    whitespace_only_total = sum(whitespace_only_by_column.values())

    report = {
        "source": {
            "path": str(path),
            "sha256": compute_file_hash(path),
        },
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "dtypes": {key: str(value) for key, value in df.dtypes.items()},
        },
        "target": {
            "counts": {str(key): int(value) for key, value in target_counts.items()},
            "proportions": {str(key): float(value) for key, value in target_proportions.items()},
        },
        "duplicate_customer_ids": int(df["customerID"].duplicated().sum()),
        "missing_values": {
            "na_counts_by_column": {str(key): int(value) for key, value in df.isna().sum().items()},
            "total_missing_cells": int(df.isna().sum().sum()),
            "whitespace_only_values_by_column": whitespace_only_by_column,
            "total_whitespace_only_values": whitespace_only_total,
        },
        "environment": {
            "python_version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "pandas_version": version("pandas"),
            "python_executable": sys.executable,
        },
    }

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the Telco churn dataset and save a machine-readable report.")
    parser.add_argument("--source", type=str, default=None, help="Path to the dataset to inspect.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "data_overview.json"),
        help="Where to save the inspection JSON report.",
    )
    args = parser.parse_args()

    report = build_report(args.source, args.output)
    print(json.dumps({
        "rows": report["dataset"]["rows"],
        "columns": report["dataset"]["columns"],
        "target_counts": report["target"]["counts"],
        "duplicate_customer_ids": report["duplicate_customer_ids"],
        "source_sha256": report["source"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
