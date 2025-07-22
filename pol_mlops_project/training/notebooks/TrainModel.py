# Databricks notebook source

# COMMAND ----------
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------
import os

from mlflow.utils.databricks_utils import dbutils

notebook_path = '/Workspace/' + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------
# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import os
from mlflow.tracking import MlflowClient
import mlflow

# Notebook Environment
dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# Path to the Hive-registered Delta table containing the training data.
dbutils.widgets.text(
    "training_data_path",
    "/databricks-datasets/nyctaxi-with-zipcodes/subsampled",
    label="Path to the training data",
)

# MLflow experiment name.
dbutils.widgets.text(
    "experiment_name",
    f"/pol_dev-pol_mlops_project-experiment",
    label="MLflow experiment name",
)

# Unity Catalog registered model name to use for the trained model.
dbutils.widgets.text(
    "model_name", "pol_dev.pol_mlops_project.pol_mlops_project-plain-model", label="Full (Three-Level) Model Name"
)

dbutils.widgets.text(
    "pickup_features_table",
    "pol_dev.pol_mlops_project.fe_trip_pickup_features",
    label="Pickup Features Table (unused)",
)

dbutils.widgets.text(
    "dropoff_features_table",
    "pol_dev.pol_mlops_project.fe_trip_dropoff_features",
    label="Dropoff Features Table (unused)",
)

dbutils.widgets.text(
    "data_path",
    "taxi_training_data",
    label="data_path",
)


# COMMAND ----------
# DBTITLE 1,Capture widget values
input_table_path = dbutils.widgets.get("training_data_path")
experiment_name  = dbutils.widgets.get("experiment_name")
model_name       = dbutils.widgets.get("model_name")

# COMMAND ----------
# DBTITLE 1,MLflow setup
mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------
# DBTITLE 1,Load raw Spark data
raw_data = spark.read.format("delta").load(input_table_path)
print(f"Raw rows: {raw_data.count():,}")
raw_data.display()

# COMMAND ----------
# DBTITLE 1,Train LightGBM
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

from importlib import import_module

training_data_mod = import_module("data." + dbutils.widgets.get("data_path"))
X_pdf, y_pdf = training_data_mod.get_training_data(raw_data)
X_train, X_test, y_train, y_test = train_test_split(X_pdf, y_pdf, random_state=123)

model = lgb.LGBMRegressor(
    num_leaves=32,
    objective="regression",
    n_estimators=100,
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
print(f"Test RMSE: {rmse:.4f}")

# COMMAND ----------
# DBTITLE 1,Infer signature & input_example
from mlflow.models import infer_signature

sample_input = X_train.iloc[:100].copy()
sample_output = model.predict(sample_input)

import pandas as pd
sample_output_df = pd.DataFrame({"prediction": sample_output})

signature = infer_signature(sample_input, sample_output_df)
input_example = sample_input.head(5)

print("Signature inferred:")
print(signature)

# COMMAND ----------
# DBTITLE 1,Log model (MLflow LightGBM flavor + Registry)
import json

def get_latest_model_version(model_name: str) -> int:
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# Fin de run previo si existiera
mlflow.end_run()

with mlflow.start_run(run_name=f"train_plain_7fe_{env}") as run:
    run_id = run.info.run_id
    # Params / metrics
    mlflow.log_param("env", env)
    mlflow.log_param("training_data_path", input_table_path)
    mlflow.log_param("feature_engineering_inline", True)
    mlflow.log_metric("rmse", rmse)

    # mlflow.log_dict({"features": feature_cols, "label": LABEL_COL}, "training_columns.json")

    # Log model
    mlflow.lightgbm.log_model(
        model,
        artifact_path="model",
        registered_model_name=model_name,
        signature=signature,
        input_example=input_example,
    )

# COMMAND ----------
# DBTITLE 1,Retrieve model URI & return to job flow
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
print(f"Registered model version: {model_version}")
print(f"Model URI: {model_uri}")

# set for next jobs
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)

# COMMAND ----------
dbutils.notebook.exit(model_uri)
