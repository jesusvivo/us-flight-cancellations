"""Spark session factory and dataset loaders."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col

from src import config


def make_spark_session(app_name: str = "us-flight-cancellations") -> SparkSession:
    """Single source of truth for SparkSession configuration.

    Pins the worker + driver Python to the current interpreter so PySpark doesn't
    spawn workers under a system Python that lacks the project's dependencies.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def load_raw(spark: SparkSession, data_dir: Path = config.DATA_DIR) -> DataFrame:
    """Read all yearly CSVs in `data_dir` into a single DataFrame, keeping only
    pre-departure columns (everything we'd know before the flight is scheduled).
    """
    df = spark.read.csv(str(data_dir / "*.csv"), header=True, inferSchema=True)
    return df.select(list(config.COLUMNS_TO_KEEP))


def undersample_balanced(
    df: DataFrame,
    label_col: str = config.TARGET_COL,
    positive_value: str = "1.0",
    sample_fraction: float = config.SAMPLE_FRACTION,
) -> DataFrame:
    """Balance the class distribution by undersampling the majority class, then
    take a fraction of the result so the from-scratch numpy LR can fit in memory.

    Ports the original `resample()` from the notebook. The two-step combine
    (undersample → fraction) preserves the original 10% slice taken AFTER the
    balance, which keeps the test set's class distribution representative.
    """
    positives = df.filter(col(label_col) == positive_value)
    negatives = df.filter(col(label_col) != positive_value)
    total_positives = positives.count()
    total_negatives = negatives.count()
    ratio = float(total_positives) / float(total_negatives)
    sampled_negatives = negatives.sample(withReplacement=False, fraction=ratio, seed=config.RANDOM_STATE)
    balanced = sampled_negatives.union(positives)
    return balanced.sample(withReplacement=False, fraction=sample_fraction, seed=config.RANDOM_STATE)
