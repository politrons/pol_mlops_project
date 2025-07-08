# Databricks notebook source
import databricks.automl
import mlflow

dbutils.widgets.text(
    "training_data_path",
    "/databricks-datasets/nyctaxi-with-zipcodes/subsampled",
    label="Path to the training data",
)
input_table_path = dbutils.widgets.get("training_data_path")

raw_data = spark.read.format("delta").load(input_table_path)
raw_data.display()

# Throw AutoML with DataFrame
databricks.automl.regress(
    dataset=raw_data,
    target_col="fare_amount",
    timeout_minutes=5,
)

