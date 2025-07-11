# Databricks notebook source
# COMMAND ----------


import json, sys, pathlib, importlib

# ----- CONFIG ----------------------------------------------------------
LEVELS_UP_TO_FILES = 3   #  tests/pipelines/notebook  -> parents[3] = .../files
# ----------------------------------------------------------------------

wk_path = json.loads(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()
)["extraContext"]["notebook_path"]

files_dir_ws = pathlib.PurePosixPath(wk_path).parents[LEVELS_UP_TO_FILES]

files_dir_driver = f"/Workspace{files_dir_ws}"

if files_dir_driver not in sys.path:
    sys.path.insert(0, files_dir_driver)
    print(f"🔗  Added a sys.path → {files_dir_driver}")

from pyspark.sql import SparkSession

from pipelines.MedallionArchitecture import compute_trip_dropoff_features

spark = SparkSession.builder.getOrCreate()

sample = spark.createDataFrame(
    [
        (10.0, 1.2, "10001", "10002"),   # válida
        (-5.0, 0.8, "10001", "10002"),   # fare negativa
        (12.0, -3.0, "10001", "10002")   # distancia negativa
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

