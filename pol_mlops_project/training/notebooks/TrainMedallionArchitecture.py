# Databricks notebook source
# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os

from mlflow.utils.databricks_utils import dbutils

notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path


# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()
# ------------------------------------------------------------------------------
# Widgets – reuse catalog / schema / experiment / model name
# ------------------------------------------------------------------------------
dbutils.widgets.text("catalog",      "pol_dev",            "Catalog")
dbutils.widgets.text("schema",       "pol_mlops_project",  "Schema")
dbutils.widgets.text("experiment",   "/pol_dev-pol_mlops_project-experiment",
                     "MLflow Experiment")
# Unity Catalog registered model name to use for the trained mode.
dbutils.widgets.text(
    "model_name", "pol_dev.pol_mlops_project.pol_mlops_project-model", label="Full (Three-Level) Model Name"
)


catalog      = dbutils.widgets.get("catalog")
schema       = dbutils.widgets.get("schema")
experiment   = dbutils.widgets.get("experiment")
model_name   = dbutils.widgets.get("model_name")

# ------------------------------------------------------------------------------
# MLflow setup
# ------------------------------------------------------------------------------
import mlflow.lightgbm

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment)

# ── Load dependencies from features --------------------------------------------------------
%cd ../features
from importlib import import_module

mod = import_module("medallion_model_trainning")
train_lightgbm = getattr(mod, "train_lightgbm")

mod = import_module("medallion_model_registry")
log_lightgbm_model = getattr(mod, "log_lightgbm_model")

# ── Spark → Pandas --------------------------------------------------------
from pyspark.sql import SparkSession

spark = SparkSession.getActiveSession()
gold_df = spark.table("pol_dev.pol_mlops_project.gold_nyc_taxi")
pdf     = gold_df.toPandas()

# ── Pure training ---------------------------------------------------------
params = {"objective": "regression", "metric": "rmse", "num_leaves": 32}
model, rmse = train_lightgbm(pdf, params)
print(f"RMSE on hold-out: {rmse:.4f}")

# ── Register model --------------------------------------
print("Registering Model " + model_name)
model_uri, model_version = log_lightgbm_model(
    model,
    sample_X = pdf.drop("fare_amount", axis=1).head(10),
    model_name = model_name,
    experiment = "/pol_dev-pol_mlops_project-experiment",
)
print(f"Registered at {model_uri}")


# Make values available to downstream tasks (if run in a Job)
dbutils.jobs.taskValues.set("model_uri",   model_uri)
dbutils.jobs.taskValues.set("model_name",  model_name)
dbutils.jobs.taskValues.set("model_version", model_version)

dbutils.notebook.exit(model_uri)

