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
    "model_name", "", label="Full (Three-Level) Model Name"
)

model_name = dbutils.widgets.get("model_name")

dbutils.widgets.text(
    "custom_model_file_name", "", label="custom_model_file_name"
)

custom_model_file_name = dbutils.widgets.get("custom_model_file_name")

dbutils.widgets.text(
    "custom_model_name", "", label="custom_model_name"
)

custom_model_name = dbutils.widgets.get("custom_model_name")

dbutils.widgets.text( "experiment_name","", label="MLflow experiment name")

dbutils.widgets.text("DATABRICKS_HOST", "", label="Databricks Host")

dbutils.widgets.text("signature_path", "", label="signature_path")

DATABRICKS_HOST = dbutils.widgets.get("DATABRICKS_HOST")
TOKEN = dbutils.secrets.get("my-scope", "databricks-token")

experiment_name = dbutils.widgets.get("experiment_name")

import mlflow

mlflow.set_registry_uri('databricks-uc')
mlflow.set_experiment(experiment_name)


# COMMAND ----------
from mlflow import MlflowClient
import mlflow

def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# ----------
# Log model
# ----------
with mlflow.start_run(run_name="Custom model"):
    import os
    import pandas as pd
    import mlflow
    from mlflow.models import infer_signature

    # ------------------------------------------------------------------
    # Log custom model using model_path
    # ------------------------------------------------------------------

    from importlib import import_module

    mod = import_module("signatures." + dbutils.widgets.get("signature_path"))
    signature = mod.get_signature()

    model_path = '/Workspace/' + os.path.dirname(
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())  + "/models/" + custom_model_file_name

    model_version = get_latest_model_version(model_name)
    model_uri = f"models:/{model_name}/{model_version}"

    import mlflow
    from mlflow import MlflowClient

    base_model_uri = f"models:/{model_name}@champion"
    local_base = mlflow.artifacts.download_artifacts(base_model_uri)

    mlflow.set_registry_uri("databricks-uc")

    model_config = {"host": DATABRICKS_HOST, "token": TOKEN}

    mlflow.end_run()

    with mlflow.start_run():
        logged_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model_path,
            artifacts={"model_a": local_base, "model_b": local_base},
            model_config=model_config,
            pip_requirements=["mlflow", "pandas","lightgbm"],
            registered_model_name=custom_model_name,
            signature=signature,
        )

    # ------------------------------------------------------------------
    # Add alias to model
    # ------------------------------------------------------------------
    client = MlflowClient(registry_uri="databricks-uc")
    target_alias="champion"
    wrapper_model_version = get_latest_model_version(custom_model_name)
    client.set_registered_model_alias(
        name=custom_model_name,
        alias=target_alias,
        version=str(wrapper_model_version))
