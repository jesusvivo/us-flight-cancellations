"""Spark feature pipeline: date parts → categorical indexing → assemble → scale."""
from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql.functions import dayofweek, month

from src import config


def extract_date_parts(df: DataFrame) -> DataFrame:
    """Derive weekday + month from `FL_DATE`, then drop the original column.

    The model can't usefully consume a date string; weekday + month carry the
    seasonal and weekly cancellation signal.
    """
    df = df.withColumn("FL_DATE_WEEKDAY", dayofweek("FL_DATE"))
    df = df.withColumn("FL_DATE_MONTH", month("FL_DATE"))
    return df.drop("FL_DATE")


def drop_redundant_columns(df: DataFrame) -> DataFrame:
    """Drop columns that strongly correlate with kept ones (CRS_ARR_TIME ↔ CRS_DEP_TIME,
    DISTANCE ↔ CRS_ELAPSED_TIME). Keeping the lower-variance representative reduces
    multicollinearity without losing signal.
    """
    return df.drop(*config.REDUNDANT_COLS_TO_DROP)


def build_feature_pipeline() -> Pipeline:
    """Spark ML Pipeline: StringIndexer per categorical column → VectorAssembler → StandardScaler.

    Outputs a `features` column (StandardScaler.outputCol) consumable by both the
    from-scratch numpy LR (after collecting) and MLlib's `LogisticRegression`.
    """
    indexers = [
        StringIndexer(inputCol="OP_CARRIER", outputCol="AIRLINE_ID", handleInvalid="keep"),
        StringIndexer(
            inputCol="ORIGIN", outputCol="ORIGIN_ID", stringOrderType="alphabetDesc", handleInvalid="keep"
        ),
        StringIndexer(
            inputCol="DEST", outputCol="DEST_ID", stringOrderType="alphabetDesc", handleInvalid="keep"
        ),
    ]
    assembler = VectorAssembler(inputCols=list(config.FEATURE_COLS), outputCol="vectorized_features")
    scaler = StandardScaler(inputCol="vectorized_features", outputCol="features")
    return Pipeline(stages=[*indexers, assembler, scaler])
