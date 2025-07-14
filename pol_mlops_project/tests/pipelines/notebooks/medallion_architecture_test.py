# Databricks notebook source
# COMMAND ----------
from pyspark.sql import SparkSession

# MAGIC %pip install pytest

dbutils.library.restartPython()

import pytest

import sys

sys.path.insert(0, '/Workspace/Users/pablo.garcia@marionete.co.uk/.bundle/pol_mlops_project/dev/files/pipelines/')

from MedallionArchitecture import *

spark = SparkSession.builder.getOrCreate()

sample = spark.createDataFrame(
    [
        (10.0, 1.2, "10001", "10002"),  # válida
        (-5.0, 0.8, "10001", "10002"),  # fare negativa
        (12.0, -3.0, "10001", "10002")  # distancia negativa
    ],
    ["fare_amount", "trip_distance", "pickup_zip", "dropoff_zip"]
)

expected = spark.createDataFrame(
    [(10.0, "10001", "10002")],
    ["fare_amount", "pickup_zip", "dropoff_zip"]
)

result = compute_trip_dropoff_features(sample)
assert (
        result.subtract(expected).count() == 0 and
        expected.subtract(result).count() == 0
), "DataFrames are not equal"

# ------------------------------------------------------------------ #
# 1 ─ Sample input: two pickups in zip 10001, one in 10002           #
# ------------------------------------------------------------------ #
sample = spark.createDataFrame(
    [
        (10.0, 1.2, "10001", "10002"),  # 10001 ↦ fare 10
        (14.0, 0.8, "10001", "10003"),  # 10001 ↦ fare 14
        (8.0, 1.0, "10002", "10004"),  # 10002 ↦ fare 8
    ],
    ["fare_amount", "trip_distance", "pickup_zip", "dropoff_zip"],
)

# ------------------------------------------------------------------ #
# 2 ─ Expected output: average fare per pickup ZIP                   #
#     10001 → (10 + 14) / 2 = 12                                      #
#     10002 → 8                                                      #
# ------------------------------------------------------------------ #
expected = spark.createDataFrame(
    [
        ("10001", 12.0),
        ("10002", 8.0),
    ],
    ["zip", "avg_fare_per_zip"],
)

# ------------------------------------------------------------------ #
# 3 ─ Run transformation                                             #
# ------------------------------------------------------------------ #
result = compute_trip_pickup_features(sample)

# ------------------------------------------------------------------ #
# 4 ─ Assert equality (order-independent)                            #
# ------------------------------------------------------------------ #
assert (
        result.subtract(expected).count() == 0
        and expected.subtract(result).count() == 0
), "DataFrames are not equal"

