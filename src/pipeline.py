"""End-to-end orchestration.

Usage:
    python -m src.pipeline                # train + evaluate
    python -m src.pipeline --train        # train only -- persists artifacts
    python -m src.pipeline --evaluate     # evaluate only -- loads cached test arrays + models
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel as MLlibLRModel
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

from src import config
from src.data import load_raw, make_spark_session, undersample_balanced
from src.evaluate import compute_metrics, plot_roc_curve, text_report
from src.features import build_feature_pipeline, drop_redundant_columns, extract_date_parts
from src.logistic_regression import LogisticRegression
from src.train import fit_custom_lr, fit_mllib_lr, fit_xgboost, two_phase_grid_search

logger = logging.getLogger(__name__)

CUSTOM_LR_PATH = config.MODELS_DIR / "custom_lr.joblib"
MLLIB_LR_DIR = config.MODELS_DIR / "mllib_lr"
XGB_PATH = config.MODELS_DIR / "xgboost.joblib"
FEATURE_PIPELINE_DIR = config.MODELS_DIR / "feature_pipeline"
TEST_ARRAYS_PATH = config.MODELS_DIR / "test_arrays.npz"
BEST_PARAMS_PATH = config.MODELS_DIR / "best_params.json"


def _collect_features(df: DataFrame, label_col: str = config.TARGET_COL) -> tuple[np.ndarray, np.ndarray]:
    """Pull the `features` column (a DenseVector) and the label out of Spark into numpy."""
    rows = df.select("features", label_col).collect()
    X = np.array([row["features"].toArray() for row in rows], dtype=np.float64)
    y = np.array([row[label_col] for row in rows], dtype=np.float64)
    return X, y


def _mllib_test_scores(model: MLlibLRModel, test_df: DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Run MLlib inference and return (y_true, y_score_positive_class)."""
    predictions = model.transform(test_df)
    take_positive = udf(lambda v: float(v[1]), DoubleType())
    predictions = predictions.withColumn("p_pos", take_positive("probability"))
    rows = predictions.select(config.TARGET_COL, "p_pos", "prediction").collect()
    y_true = np.array([row[config.TARGET_COL] for row in rows], dtype=np.float64)
    y_score = np.array([row["p_pos"] for row in rows], dtype=np.float64)
    return y_true, y_score


def run_train() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    spark = make_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    logger.info("Loading raw CSVs from %s", config.DATA_DIR)
    df = load_raw(spark)
    logger.info("Undersampling + sampling %s%% slice...", int(config.SAMPLE_FRACTION * 100))
    df = undersample_balanced(df)
    df = extract_date_parts(df)
    df = drop_redundant_columns(df)
    df = df.dropna(how="any")

    logger.info("Fitting feature pipeline...")
    feature_pipeline = build_feature_pipeline().fit(df)
    df = feature_pipeline.transform(df)

    train_df, test_df = df.randomSplit(list(config.TRAIN_TEST_SPLIT), seed=config.RANDOM_STATE)

    logger.info("Collecting train/test feature arrays to numpy...")
    X_train, y_train = _collect_features(train_df)
    X_test, y_test = _collect_features(test_df)
    logger.info("Train: %s, Test: %s", X_train.shape, X_test.shape)

    logger.info("Tuning custom LR via two-phase grid search...")
    best_params, best_cv = two_phase_grid_search(X_train, y_train, X_test, y_test)
    logger.info("Best params: %s (F1=%.4f)", best_params, best_cv)

    custom_model = fit_custom_lr(X_train, y_train, best_params)

    logger.info("Fitting MLlib LR for comparison...")
    mllib_model = fit_mllib_lr(train_df)
    y_test_mllib, y_score_mllib = _mllib_test_scores(mllib_model, test_df)

    logger.info("Fitting XGBoost...")
    xgb_model = fit_xgboost(X_train, y_train)
    y_score_xgb = xgb_model.predict_proba(X_test)[:, 1]

    joblib.dump(custom_model, CUSTOM_LR_PATH)
    joblib.dump(xgb_model, XGB_PATH)
    mllib_model.write().overwrite().save(str(MLLIB_LR_DIR))
    feature_pipeline.write().overwrite().save(str(FEATURE_PIPELINE_DIR))
    np.savez(
        TEST_ARRAYS_PATH,
        X_test=X_test,
        y_test=y_test,
        y_test_mllib=y_test_mllib,
        y_score_mllib=y_score_mllib,
        y_score_xgb=y_score_xgb,
    )
    BEST_PARAMS_PATH.write_text(json.dumps({"best_params": best_params, "grid_search_f1": best_cv}, indent=2))

    _evaluate_and_report(custom_model, X_test, y_test, y_test_mllib, y_score_mllib, y_score_xgb)
    spark.stop()


def run_evaluate() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not TEST_ARRAYS_PATH.exists():
        raise FileNotFoundError(
            f"Cached test arrays missing at {TEST_ARRAYS_PATH}. Run `python -m src.pipeline --train` first."
        )
    logger.info("Loading cached test arrays + custom LR...")
    arrays = np.load(TEST_ARRAYS_PATH)
    X_test, y_test = arrays["X_test"], arrays["y_test"]
    y_test_mllib, y_score_mllib = arrays["y_test_mllib"], arrays["y_score_mllib"]
    y_score_xgb = arrays["y_score_xgb"]
    custom_model = joblib.load(CUSTOM_LR_PATH)

    _evaluate_and_report(custom_model, X_test, y_test, y_test_mllib, y_score_mllib, y_score_xgb)


def _evaluate_and_report(
    custom_model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_test_mllib: np.ndarray,
    y_score_mllib: np.ndarray,
    y_score_xgb: np.ndarray,
) -> None:
    y_pred_custom = custom_model.predict(X_test)
    y_score_custom = custom_model.predict_proba(X_test)

    print("\n=== Custom LR ===")
    print(text_report(y_test, y_pred_custom))
    print(json.dumps(compute_metrics(y_test, y_pred_custom), indent=2))
    auc_custom = plot_roc_curve(
        y_test, y_score_custom, "Custom LR (from scratch)", config.FIGURES_DIR / "roc_custom.png"
    )

    print("\n=== MLlib LR ===")
    y_pred_mllib = (y_score_mllib > 0.5).astype(int)
    print(text_report(y_test_mllib, y_pred_mllib))
    print(json.dumps(compute_metrics(y_test_mllib, y_pred_mllib), indent=2))
    auc_mllib = plot_roc_curve(
        y_test_mllib, y_score_mllib, "PySpark MLlib LR", config.FIGURES_DIR / "roc_mllib.png"
    )

    print("\n=== XGBoost ===")
    y_pred_xgb = (y_score_xgb > 0.5).astype(int)
    print(text_report(y_test, y_pred_xgb))
    print(json.dumps(compute_metrics(y_test, y_pred_xgb), indent=2))
    auc_xgb = plot_roc_curve(
        y_test, y_score_xgb, "XGBoost", config.FIGURES_DIR / "roc_xgboost.png"
    )

    print(f"\nCustom AUC: {auc_custom:.4f}  |  MLlib AUC: {auc_mllib:.4f}  |  XGBoost AUC: {auc_xgb:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PySpark feature prep + from-scratch logistic regression for flight cancellations."
    )
    parser.add_argument("--train", action="store_true", help="Train only -- persists artifacts.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only -- loads cached arrays + models.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    run_train_flag = args.train or not (args.train or args.evaluate)
    run_evaluate_flag = args.evaluate or not (args.train or args.evaluate)

    if run_train_flag:
        run_train()
    elif run_evaluate_flag:
        run_evaluate()


if __name__ == "__main__":
    main()
