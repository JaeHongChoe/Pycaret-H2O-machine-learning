from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from pandas.api.types import is_categorical_dtype, is_object_dtype, is_string_dtype
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "GIB"
DEFAULT_ID_COLUMNS = ("number",)
DEFAULT_DROP_COLUMNS = ("Mortality", "Intracranial hemorrhage", "D6")
TARGET_MAP = {1: 1, 2: 0}


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class TrainValidSplit:
    train: pd.DataFrame
    valid: pd.DataFrame


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the source Excel file used by the original notebook."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return pd.read_excel(dataset_path, engine="openpyxl")


def check_missing_columns(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Return column-level missing-value metadata for reporting."""
    missing_columns: list[dict[str, object]] = []
    for column in dataframe.columns:
        missing_count = int(dataframe[column].isna().sum())
        if missing_count:
            missing_columns.append(
                {
                    "column": column,
                    "dtype": str(dataframe[column].dtype),
                    "missing_count": missing_count,
                }
            )
    return missing_columns


def drop_categorical_missing_rows(
    dataframe: pd.DataFrame,
    missing_columns: Iterable[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Drop rows only for categorical-like columns with missing values.

    The notebook removed rows for categorical missing values rather than
    performing a broader imputation strategy, so this function preserves
    that original decision while broadening dtype support to string/category.
    """
    result = dataframe.copy()
    missing_columns = list(missing_columns or check_missing_columns(result))
    for item in missing_columns:
        column = str(item["column"])
        series = result[column]
        if is_object_dtype(series) or is_string_dtype(series) or is_categorical_dtype(series):
            result = result.dropna(subset=[column])
    return result.reset_index(drop=True)


def prepare_dataset(
    dataframe: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    id_columns: Sequence[str] = DEFAULT_ID_COLUMNS,
    drop_columns: Sequence[str] = DEFAULT_DROP_COLUMNS,
) -> pd.DataFrame:
    """Apply the fixed column-removal and label-mapping logic from the notebook."""
    result = dataframe.copy()
    if target_column not in result.columns:
        raise KeyError(f"Target column '{target_column}' was not found in the dataset.")

    removable_columns = [
        column
        for column in [*id_columns, *drop_columns]
        if column in result.columns and column != target_column
    ]
    if removable_columns:
        result = result.drop(columns=removable_columns)

    result[target_column] = result[target_column].replace(TARGET_MAP)

    return result


def split_train_test(
    dataframe: pd.DataFrame,
    *,
    test_size: float = 0.3,
    random_state: int = 15,
    shuffle: bool = True,
) -> DatasetSplit:
    """Reproduce the notebook's train/test split in a reusable form."""
    train_frame, test_frame = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    return DatasetSplit(
        train=train_frame.reset_index(drop=True),
        test=test_frame.reset_index(drop=True),
    )


def split_train_valid(
    dataframe: pd.DataFrame,
    *,
    valid_size: float = 0.2,
    random_state: int = 15,
    shuffle: bool = True,
) -> TrainValidSplit:
    """Split the training frame into train/valid sets for H2O AutoML."""
    train_frame, valid_frame = train_test_split(
        dataframe,
        test_size=valid_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    return TrainValidSplit(
        train=train_frame.reset_index(drop=True),
        valid=valid_frame.reset_index(drop=True),
    )


def _target_distribution(dataframe: pd.DataFrame, target_column: str) -> dict[str, int]:
    if target_column not in dataframe.columns:
        return {}
    counts = dataframe[target_column].value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def build_preprocess_summary(
    raw_dataframe: pd.DataFrame,
    cleaned_dataframe: pd.DataFrame,
    prepared_dataframe: pd.DataFrame,
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
) -> dict[str, object]:
    """Build a compact preprocessing report for README-friendly artifacts."""
    removed_rows_after_na_cleanup = int(len(raw_dataframe) - len(cleaned_dataframe))
    removed_columns = [column for column in raw_dataframe.columns if column not in prepared_dataframe.columns]
    feature_columns = [column for column in prepared_dataframe.columns if column != target_column]

    return {
        "raw_shape": list(raw_dataframe.shape),
        "cleaned_shape": list(cleaned_dataframe.shape),
        "prepared_shape": list(prepared_dataframe.shape),
        "train_shape": list(train_dataframe.shape),
        "test_shape": list(test_dataframe.shape),
        "target_column": target_column,
        "removed_rows_after_na_cleanup": removed_rows_after_na_cleanup,
        "removed_columns": removed_columns,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "train_target_distribution": _target_distribution(train_dataframe, target_column),
        "test_target_distribution": _target_distribution(test_dataframe, target_column),
    }
