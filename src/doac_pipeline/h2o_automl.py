from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class H2ORunResult:
    leaderboard: pd.DataFrame
    variable_importance: pd.DataFrame
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    validation_metrics: dict[str, float | str | None]
    test_metrics: dict[str, float | str | None]
    summary: dict[str, object]


def _rename_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    renamed = predictions.copy()
    rename_map = {}
    if "predict" in renamed.columns:
        rename_map["predict"] = "prediction_label"
    if "p1" in renamed.columns:
        rename_map["p1"] = "prediction_score"
    if "p0" in renamed.columns:
        rename_map["p0"] = "prediction_score_negative"
    return renamed.rename(columns=rename_map)


def _metric_value(metric_output: object) -> float | None:
    if not metric_output:
        return None
    if isinstance(metric_output, list) and metric_output:
        first_item = metric_output[0]
        if isinstance(first_item, (list, tuple)) and len(first_item) >= 2:
            try:
                return float(first_item[1])
            except (TypeError, ValueError):
                return None
    return None


def _extract_metrics(performance: object, *, model_id: str) -> dict[str, float | str | None]:
    return {
        "model_id": model_id,
        "accuracy": _metric_value(performance.accuracy()),
        "precision": _metric_value(performance.precision()),
        "recall": _metric_value(performance.recall()),
        "f1": _metric_value(performance.F1()),
        "auc": float(performance.auc()) if performance.auc() is not None else None,
    }


def _to_h2o_frame(h2o_module: object, dataframe: pd.DataFrame, target_column: str) -> object:
    h2o_frame = h2o_module.H2OFrame(dataframe)
    if target_column in dataframe.columns:
        h2o_frame[target_column] = h2o_frame[target_column].asfactor()
    return h2o_frame


def _variable_importance_dataframe(leader_model: object) -> pd.DataFrame:
    try:
        variable_importance = leader_model.varimp(use_pandas=True)
        if isinstance(variable_importance, pd.DataFrame):
            return variable_importance
    except TypeError:
        pass

    rows = leader_model.varimp() or []
    if not rows:
        return pd.DataFrame(columns=["variable", "relative_importance", "scaled_importance", "percentage"])
    return pd.DataFrame(
        rows,
        columns=["variable", "relative_importance", "scaled_importance", "percentage"],
    )


def run_h2o_automl(
    train_dataframe: pd.DataFrame,
    valid_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    *,
    target_column: str = "GIB",
    max_runtime_secs: int = 60 * 60 * 16,
    exclude_algos: tuple[str, ...] = ("DRF", "GLM"),
    min_mem_size_gb: int = 16,
    max_mem_size_gb: int = 16,
    seed: int = 15,
) -> H2ORunResult:
    """Run the final H2O AutoML training/evaluation loop and return extracted artifacts."""
    import h2o
    from h2o.automl import H2OAutoML

    try:
        h2o.init(min_mem_size=min_mem_size_gb, max_mem_size=max_mem_size_gb)
        h2o.no_progress()

        predictors = [column for column in train_dataframe.columns if column != target_column]

        h2o_train = _to_h2o_frame(h2o, train_dataframe, target_column)
        h2o_valid = _to_h2o_frame(h2o, valid_dataframe, target_column)
        h2o_test = _to_h2o_frame(h2o, test_dataframe, target_column)

        automl = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            exclude_algos=list(exclude_algos),
            seed=seed,
            sort_metric="AUC",
        )
        automl.train(
            x=predictors,
            y=target_column,
            training_frame=h2o_train,
            leaderboard_frame=h2o_valid,
        )

        leaderboard = automl.leaderboard.as_data_frame()
        validation_performance = automl.leader.model_performance(h2o_valid)
        test_performance = automl.leader.model_performance(h2o_test)

        validation_predictions = _rename_prediction_columns(automl.leader.predict(h2o_valid).as_data_frame())
        validation_predictions[target_column] = valid_dataframe[target_column].reset_index(drop=True)

        test_predictions = _rename_prediction_columns(automl.leader.predict(h2o_test).as_data_frame())
        test_predictions[target_column] = test_dataframe[target_column].reset_index(drop=True)

        validation_metrics = _extract_metrics(validation_performance, model_id=automl.leader.model_id)
        test_metrics = _extract_metrics(test_performance, model_id=automl.leader.model_id)
        variable_importance = _variable_importance_dataframe(automl.leader)

        summary = {
            "target_column": target_column,
            "predictor_count": len(predictors),
            "predictors": predictors,
            "leader_model_id": automl.leader.model_id,
            "leaderboard_rows": int(len(leaderboard)),
            "max_runtime_secs": max_runtime_secs,
            "exclude_algos": list(exclude_algos),
            "seed": seed,
            "train_rows": int(len(train_dataframe)),
            "valid_rows": int(len(valid_dataframe)),
            "test_rows": int(len(test_dataframe)),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        }

        return H2ORunResult(
            leaderboard=leaderboard,
            variable_importance=variable_importance,
            validation_predictions=validation_predictions,
            test_predictions=test_predictions,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            summary=summary,
        )
    finally:
        try:
            h2o.cluster().shutdown(prompt=False)
        except Exception:
            pass
