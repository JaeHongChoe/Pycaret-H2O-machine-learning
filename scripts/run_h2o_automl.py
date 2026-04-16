#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from doac_pipeline.h2o_automl import run_h2o_automl  # noqa: E402
from doac_pipeline.preprocess import (  # noqa: E402
    DEFAULT_DROP_COLUMNS,
    DEFAULT_ID_COLUMNS,
    TARGET_COLUMN,
    build_preprocess_summary,
    check_missing_columns,
    drop_categorical_missing_rows,
    load_dataset,
    prepare_dataset,
    split_train_test,
    split_train_valid,
)
from doac_pipeline.utils import write_dataframe, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the H2O AutoML pipeline extracted from doac.ipynb.")
    parser.add_argument("--data-path", required=True, help="Path to the Excel dataset used in the notebook.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "h2o_automl"),
        help="Directory where CSV/JSON outputs will be stored.",
    )
    parser.add_argument("--target-column", default=TARGET_COLUMN, help="Target column name.")
    parser.add_argument(
        "--id-column",
        dest="id_columns",
        action="append",
        default=None,
        help="Column to drop as identifier. Can be passed multiple times.",
    )
    parser.add_argument(
        "--drop-column",
        dest="drop_columns",
        action="append",
        default=None,
        help="Additional feature column to drop before training. Can be passed multiple times.",
    )
    parser.add_argument("--test-size", type=float, default=0.3, help="Train/test split ratio for the test set.")
    parser.add_argument("--valid-size", type=float, default=0.2, help="Train/valid split ratio inside train.")
    parser.add_argument("--random-state", type=int, default=15, help="Random seed for train/test split.")
    parser.add_argument("--max-runtime-secs", type=int, default=60 * 60 * 16, help="H2O AutoML runtime budget.")
    parser.add_argument(
        "--exclude-algo",
        dest="exclude_algos",
        action="append",
        default=None,
        help="H2O algorithms to exclude. Can be passed multiple times.",
    )
    parser.add_argument("--min-mem-size-gb", type=int, default=16, help="Minimum H2O memory allocation in GB.")
    parser.add_argument("--max-mem-size-gb", type=int, default=16, help="Maximum H2O memory allocation in GB.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    id_columns = tuple(args.id_columns) if args.id_columns else DEFAULT_ID_COLUMNS
    drop_columns = tuple(args.drop_columns) if args.drop_columns else DEFAULT_DROP_COLUMNS
    exclude_algos = tuple(args.exclude_algos) if args.exclude_algos else ("DRF", "GLM")

    raw_dataframe = load_dataset(args.data_path)
    missing_columns = check_missing_columns(raw_dataframe)
    cleaned_dataframe = drop_categorical_missing_rows(raw_dataframe, missing_columns)
    prepared_dataframe = prepare_dataset(
        cleaned_dataframe,
        target_column=args.target_column,
        id_columns=id_columns,
        drop_columns=drop_columns,
    )

    train_test_split_result = split_train_test(
        prepared_dataframe,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_valid_split_result = split_train_valid(
        train_test_split_result.train,
        valid_size=args.valid_size,
        random_state=args.random_state,
    )

    preprocess_summary = build_preprocess_summary(
        raw_dataframe,
        cleaned_dataframe,
        prepared_dataframe,
        train_test_split_result.train,
        train_test_split_result.test,
        target_column=args.target_column,
    )
    preprocess_summary["missing_columns"] = missing_columns
    preprocess_summary["valid_shape"] = list(train_valid_split_result.valid.shape)
    preprocess_summary["id_columns"] = list(id_columns)
    preprocess_summary["drop_columns"] = list(drop_columns)
    write_json(output_dir / "preprocess_summary.json", preprocess_summary)

    result = run_h2o_automl(
        train_valid_split_result.train,
        train_valid_split_result.valid,
        train_test_split_result.test,
        target_column=args.target_column,
        max_runtime_secs=args.max_runtime_secs,
        exclude_algos=exclude_algos,
        min_mem_size_gb=args.min_mem_size_gb,
        max_mem_size_gb=args.max_mem_size_gb,
        seed=args.random_state,
    )

    write_dataframe(output_dir / "leaderboard.csv", result.leaderboard)
    write_dataframe(output_dir / "variable_importance.csv", result.variable_importance)
    write_dataframe(output_dir / "validation_predictions.csv", result.validation_predictions)
    write_dataframe(output_dir / "test_predictions.csv", result.test_predictions)
    write_json(output_dir / "validation_metrics.json", result.validation_metrics)
    write_json(output_dir / "test_metrics.json", result.test_metrics)
    write_json(
        output_dir / "run_summary.json",
        {
            **result.summary,
            "data_path": args.data_path,
            "output_dir": str(output_dir),
            "id_columns": list(id_columns),
            "drop_columns": list(drop_columns),
            "test_size": args.test_size,
            "valid_size": args.valid_size,
            "random_state": args.random_state,
            "exclude_algos": list(exclude_algos),
        },
    )

    print(f"Saved H2O AutoML artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
