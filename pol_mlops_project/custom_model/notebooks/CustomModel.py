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
    "model_name", "pol_dev.pol_mlops_project.pol_mlops_project-plain-model", label="Full (Three-Level) Model Name"
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
from mlflow import MlflowClient
import mlflow

# ----------
# Log model
# ----------
with mlflow.start_run(run_name="Custom model"):
    import os
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
    # Log custom model using model_path
    # ------------------------------------------------------------------
    signature = infer_signature(model_input=sample_df, model_output=sample_out_df)

    wrapper_model_name="pol_dev.pol_mlops_project.pol_mlops_project-custom-ab-plain-model"

    model_path = '/Workspace/' + os.path.dirname(
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())  + "/cascade_ab_plain_model.py"

    model_version = get_latest_model_version(model_name)
    model_uri = f"models:/{model_name}/{model_version}"

    import mlflow
    from mlflow import MlflowClient

    base_model_uri = f"models:/{model_name}@champion"
    local_base = mlflow.artifacts.download_artifacts(base_model_uri)

    mlflow.set_registry_uri("databricks-uc")

    model_config = {}

    mlflow.end_run()

    with mlflow.start_run():
        logged_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model_path,
            artifacts={"model_a": local_base, "model_b": local_base},
            model_config=model_config,
            pip_requirements=["mlflow", "pandas","lightgbm"],
            registered_model_name=wrapper_model_name,
            signature=signature,
        )

    # ------------------------------------------------------------------
    # Add alias to model
    # ------------------------------------------------------------------
    client = MlflowClient(registry_uri="databricks-uc")
    target_alias="champion"
    wrapper_model_version = get_latest_model_version(wrapper_model_name)
    client.set_registered_model_alias(
        name=wrapper_model_name,
        alias=target_alias,
        version=str(wrapper_model_version))
