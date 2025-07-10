# Databricks notebook source
##################################################################################
# Training Notebook (consumes pre-built Medallion tables)
#
# Reads:
#   pol_dev.pol_mlops_project.gold_nyc_taxi
# Trains:
#   LightGBM regression
# Logs / registers model in Unity Catalog
##################################################################################
# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os

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
from mlflow.models import infer_signature
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
import lightgbm as lgb
from sklearn.model_selection import train_test_split

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment)

# ------------------------------------------------------------------------------
# Load Gold data
# ------------------------------------------------------------------------------
gold_table = f"{catalog}.{schema}.gold_nyc_taxi"
gold_df    = spark.table(gold_table)

# Convert to pandas
pdf = gold_df.toPandas()

# ------------------------------------------------------------------------------
# Train / test split
# ------------------------------------------------------------------------------
train, test = train_test_split(pdf, random_state=42, test_size=0.2)
X_train, y_train = train.drop("fare_amount", axis=1), train["fare_amount"]
X_test,  y_test  = test.drop("fare_amount",  axis=1), test["fare_amount"]

# ------------------------------------------------------------------------------
# Train LightGBM
# ------------------------------------------------------------------------------
params = {"objective": "regression", "metric": "rmse", "num_leaves": 32}
model  = lgb.train(params,
                   lgb.Dataset(X_train, label=y_train),
                   num_boost_round=50)

# ------------------------------------------------------------------------------
# Log model and metrics
# ------------------------------------------------------------------------------
with mlflow.start_run() as run:
    # 1. optional: autolog metrics / params
    mlflow.lightgbm.autolog(log_models=False)

    # 2. infer signature on a tiny sample (fast)
    sample     = X_train.head(10)
    signature  = infer_signature(sample, model.predict(sample))

    # 3. log the model + signature
    mlflow.lightgbm.log_model(
        model,
        artifact_path="model",
        registered_model_name=model_name,
        signature=signature
    )


def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"

# Make values available to downstream tasks (if run in a Job)
dbutils.jobs.taskValues.set("model_uri",   model_uri)
dbutils.jobs.taskValues.set("model_name",  model_name)
dbutils.jobs.taskValues.set("model_version", model_version)

print(f"✅ Model registered: {model_uri}")
dbutils.notebook.exit(model_uri)

