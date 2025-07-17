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

# DBTITLE 1, Notebook arguments

# Unity Catalog registered model name to use for the trained mode.
dbutils.widgets.text(
    "model_name", "pol_dev.pol_mlops_project.pol_mlops_project-model", label="Full (Three-Level) Model Name"
)

model_name = dbutils.widgets.get("model_name")

dbutils.widgets.text(
    "experiment_name",
    f"/pol_dev-pol_mlops_project-experiment",
    label="MLflow experiment name",
)

experiment_name = dbutils.widgets.get("experiment_name")

import mlflow

mlflow.set_registry_uri('databricks-uc')
mlflow.set_experiment(experiment_name)

def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


# COMMAND ----------
import mlflow
from mlflow import MlflowClient

import sys

dbutils.widgets.text("model_path",
                     "/Workspace/Users/pablo.garcia@marionete.co.uk/.bundle/pol_mlops_project/dev/files/custom_model/classes/",
                     label="module_name")
model_path = dbutils.widgets.get("model_path")

dbutils.widgets.text("module_name", "cascade_ab_model", label="module_name")
module_name = dbutils.widgets.get("module_name")

dbutils.widgets.text("module_func_name", "CascadeABModel", label="module_func_name")
module_func_name = dbutils.widgets.get("module_func_name")

sys.path.append(model_path)

from importlib import import_module

mod = import_module(module_name)
custom_model = getattr(mod, module_func_name)

model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"

wrapper_model = custom_model(model_uri, model_uri)

# ----------
# Log model
# ----------
with mlflow.start_run(run_name="Custom model"):
    # infer signature on a tiny sample (fast)
    import pandas as pd
    import mlflow
    from mlflow.models import infer_signature

    # ------------------------------------------------------------------
    # Define the input columns your wrapped model will accept
    # ------------------------------------------------------------------
    INPUT_COLUMNS = [
        "trip_distance",
        "pickup_zip",
        "dropoff_zip",
        "mean_fare_window_1h_pickup_zip",
        "count_trips_window_1h_pickup_zip",
        "count_trips_window_30m_dropoff_zip",
        "dropoff_is_weekend",
    ]

    # ------------------------------------------------------------------
    # Build a small pandas DataFrame that matches your serving payload
    # (use realistic types; multiple rows help catch dtype issues)
    # ------------------------------------------------------------------
    sample_df = pd.DataFrame(
        [
            [2.5, 7002, 7002, 8.5, 1, 1, 0],
            [1.2, 10018, 10167, 6.75, 2, 2, 0],
        ],
        columns=INPUT_COLUMNS,
        dtype="float64"  # cast everything float; OR specify per-column below
    )

    # ------------------------------------------------------------------
    # Build a *dummy* output DataFrame just to define the schema.
    # We don't need to run the real model for signature inference.
    # ------------------------------------------------------------------
    sample_out_df = pd.DataFrame({
        "pred_a": [0.0, 0.0],
        "pred_b": [0.0, 0.0],
    })

    # ------------------------------------------------------------------
    # Infer a signature from the sample inputs & outputs
    # ------------------------------------------------------------------
    signature = infer_signature(model_input=sample_df, model_output=sample_out_df)

    wrapper_model_name = "pol_dev.pol_mlops_project.pol_mlops_project-custom-ab-model"

    mlflow.lightgbm.log_model(
        wrapper_model,
        artifact_path="model",
        registered_model_name=wrapper_model_name,
        signature=signature
    )
