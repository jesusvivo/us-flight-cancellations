"""Smoke test for the Spark feature pipeline on a tiny in-memory DataFrame.

Slow because SparkSession spin-up is slow (~5 s on a cold JVM), but real — runs
the actual `extract_date_parts` + `build_feature_pipeline` code paths against a
shape that matches the production schema, just much smaller.
"""
from __future__ import annotations

import os
import sys

import pytest

pyspark = pytest.importorskip("pyspark")

from pyspark.sql import SparkSession  # noqa: E402

from src import config  # noqa: E402
from src.features import build_feature_pipeline, drop_redundant_columns, extract_date_parts  # noqa: E402


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    s = (
        SparkSession.builder
        .appName("us-flight-cancellations-tests")
        .master("local[1]")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield s
    s.stop()


def _sample_rows() -> list[dict]:
    return [
        {
            "FL_DATE": "2018-01-15",
            "OP_CARRIER": "AA",
            "ORIGIN": "JFK",
            "DEST": "LAX",
            "CRS_DEP_TIME": 800,
            "CRS_ARR_TIME": 1100,
            "CANCELLED": 0.0,
            "CRS_ELAPSED_TIME": 300,
            "DISTANCE": 2475,
        },
        {
            "FL_DATE": "2018-03-22",
            "OP_CARRIER": "DL",
            "ORIGIN": "ATL",
            "DEST": "ORD",
            "CRS_DEP_TIME": 1430,
            "CRS_ARR_TIME": 1600,
            "CANCELLED": 1.0,
            "CRS_ELAPSED_TIME": 130,
            "DISTANCE": 606,
        },
    ]


def test_extract_date_parts_creates_weekday_and_month(spark: SparkSession):
    df = spark.createDataFrame(_sample_rows())
    out = extract_date_parts(df)
    assert "FL_DATE" not in out.columns
    assert "FL_DATE_WEEKDAY" in out.columns
    assert "FL_DATE_MONTH" in out.columns


def test_feature_pipeline_produces_features_vector(spark: SparkSession):
    df = spark.createDataFrame(_sample_rows())
    df = extract_date_parts(df)
    df = drop_redundant_columns(df)

    fitted = build_feature_pipeline().fit(df)
    transformed = fitted.transform(df)
    assert "features" in transformed.columns

    # The assembled feature vector must contain exactly the configured features.
    first_row = transformed.select("features").first()
    assert first_row["features"].size == len(config.FEATURE_COLS)
