from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from telco_churn.training import compare_models, train_and_select


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare churn models for the Telco dataset.")
    parser.add_argument("--source", type=str, default=None, help="Path to the raw CSV source dataset.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "reports" / "model_comparison.json"),
        help="Path for the comparison JSON report.",
    )
    args = parser.parse_args()

    comparison, metadata, _ = train_and_select(args.source)
    comparison.to_json(args.output, orient="records", indent=2)
    print(json.dumps({
        "selected_model": metadata["model_name"],
        "selected_metrics": metadata["metrics"],
        "artifact": metadata["artifact"],
    }, indent=2))


if __name__ == "__main__":
    main()
