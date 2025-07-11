# Databricks notebook source
# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

#MAGIC %run ../../../pipelines/MedallionArchitecture
from pyspark.sql import SparkSession
from pyspark.sql.functions import assert_true

from pipelines.MedallionArchitecture import hello_world

hello_world()
# spark = SparkSession.builder.getOrCreate()
#
# sample = spark.createDataFrame(
#     [
#         (10.0, 1.2, "10001", "10002"),   # válida
#         (-5.0, 0.8, "10001", "10002"),   # fare negativa
#         (12.0, -3.0, "10001", "10002")   # distancia negativa
#     ],
#     ["fare_amount", "trip_distance", "pickup_zip", "dropoff_zip"]
# )
#
# expected = spark.createDataFrame(
#     [(10.0, "10001", "10002")],
#     ["fare_amount", "pickup_zip", "dropoff_zip"]
# )
#
# assert_true (trip_dropoff_features(sample), expected)


