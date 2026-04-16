from __future__ import annotations

import inspect
from dataclasses import dataclass

import pandas as pd

from .metrics import classification_metrics


@dataclass(frozen=True)
class PyCaretRunResult:
    compare_table: pd.DataFrame
    test_metrics_table: pd.DataFrame
    predictions: pd.DataFrame
    summary: dict[str, object]


def _normalize_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    normalized = predictions.copy()
    rename_map = {}
    if "Label" in normalized.columns and "prediction_label" not in normalized.columns:
        rename_map["Label"] = "prediction_label"
    if "Score" in normalized.columns and "prediction_score" not in normalized.columns:
        rename_map["Score"] = "prediction_score"
    return normalized.rename(columns=rename_map)


def run_pycaret_baseline(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    *,
    target_column: str = "GIB",
    session_id: int = 123,
    tune_iterations: int = 200,
    candidate_models: tuple[str, ...] = ("et", "gbc", "lr"),
    normalize: bool = True,
    use_gpu: bool = False,
) -> PyCaretRunResult:
    """Run the baseline-model comparison flow extracted from ``doac.ipynb``."""
    from pycaret.classification import (
        blend_models,
        compare_models,
        create_model,
        finalize_model,
        predict_model,
        pull,
        setup,
        tune_model,
    )

    setup_kwargs = {
        "data": train_dataframe,
        "target": target_column,
        "normalize": normalize,
        "session_id": session_id,
        "use_gpu": use_gpu,
        "html": False,
    }
    signature = inspect.signature(setup)
    if "silent" in signature.parameters:
        setup_kwargs["silent"] = True
    if "verbose" in signature.parameters:
        setup_kwargs["verbose"] = False

    setup(**setup_kwargs)

    compare_models(sort="AUC")
    compare_table = pull().copy()

    trained_models = {
        model_name: create_model(model_name, cross_validation=False)
        for model_name in candidate_models
    }
    tuned_models = {
        model_name: tune_model(model, optimize="AUC", n_iter=tune_iterations)
        for model_name, model in trained_models.items()
    }

    blended_model = blend_models(estimator_list=list(tuned_models.values()), optimize="AUC")
    final_model = finalize_model(blended_model)

    predictions = _normalize_prediction_columns(predict_model(final_model, data=test_dataframe).copy())
    test_metrics_table = pull().copy()

    summary = {
        "target_column": target_column,
        "candidate_models": list(candidate_models),
        "tune_iterations": tune_iterations,
        "train_rows": int(len(train_dataframe)),
        "test_rows": int(len(test_dataframe)),
        "prediction_rows": int(len(predictions)),
        "prediction_metrics": classification_metrics(predictions, truth_column=target_column),
        "compare_table_rows": int(len(compare_table)),
        "compare_table_columns": list(compare_table.columns),
        "final_model_class": type(final_model).__name__,
    }

    return PyCaretRunResult(
        compare_table=compare_table,
        test_metrics_table=test_metrics_table,
        predictions=predictions,
        summary=summary,
    )
