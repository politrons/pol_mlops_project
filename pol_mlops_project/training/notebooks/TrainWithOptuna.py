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
dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment")
env = dbutils.widgets.get("env")
dbutils.widgets.text("catalog",      "pol_dev",            "Catalog")
dbutils.widgets.text("schema",       "pol_mlops_project",  "Schema")
dbutils.widgets.text("experiment",   "/pol_dev-pol_mlops_project-experiment", "MLflow Experiment")
# Unity Catalog registered model name to use for the trained mode.
dbutils.widgets.text(
    "model_name", "pol_dev.pol_mlops_project.pol_mlops_project-model", label="Full (Three-Level) Model Name"
)


catalog      = dbutils.widgets.get("catalog")
schema       = dbutils.widgets.get("schema")
experiment   = dbutils.widgets.get("experiment")
model_name   = dbutils.widgets.get("model_name")

# ------------------------------------------------------------------------------
# MLflow setup  Optuna hyper-parameter search
# ------------------------------------------------------------------------------
import mlflow
import optuna
from optuna_integration.lightgbm import LightGBMPruningCallback
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment)

# ------------------------------------------------------------------------------
# Load Gold data
# ------------------------------------------------------------------------------
gold_table = f"{catalog}.{schema}.gold_nyc_taxi"
gold_df    = spark.table(gold_table)

# Convert to pandas
pdf = gold_df.toPandas()

train_df, valid_df = train_test_split(pdf, random_state=42)
# Return data without label=fare_amount and also the label to be used later for validation
X_train, y_train = train_df.drop("fare_amount", axis=1), train_df["fare_amount"]
X_valid, y_valid = valid_df.drop("fare_amount", axis=1), valid_df["fare_amount"]

lgb_train = lgb.Dataset(X_train, label=y_train.values)
lgb_valid = lgb.Dataset(X_valid, label=y_valid.values)

# ─── MINIMAL base params ──────────────────────────────────────────────────────
BASE_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "feature_pre_filter": False,
}

def lgbm_objective(trial):
    params = BASE_PARAMS | {
        # To add hyperparameters to be tuned to get the best performance
        "num_leaves"       : trial.suggest_int("num_leaves", 16, 128, step=8),
        "learning_rate"    : trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    with mlflow.start_run(nested=True):
        mlflow.lightgbm.autolog(log_models=False)
        gbm = lgb.train(
            params,
            train_set=lgb_train,
            num_boost_round=2000,
            valid_sets=[lgb_valid],
            callbacks=[LightGBMPruningCallback(trial, "rmse")],
        )
        return gbm.best_score["valid_0"]["rmse"]

study = optuna.create_study(
    study_name=f"lgbm_optuna_{env}",
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
)

study.optimize(lgbm_objective, n_trials=4)
print("Best RMSE:", study.best_value)
print("Best params:", study.best_params)

with mlflow.start_run() as run:
    mlflow.start_run(run_name="best_lgbm_model", nested=True)
    mlflow.lightgbm.autolog()

    # ------------------------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------------------------
    best_params = BASE_PARAMS | study.best_params
    model = lgb.train(
        best_params,
        train_set=lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_valid],
    )

    # ------------------------------------------------------------------------------
    # Log model and metrics
    # ------------------------------------------------------------------------------
    # infer signature on a tiny sample (fast)
    sample     = X_train.head(10)
    signature  = infer_signature(sample, model.predict(sample))

    # log the model + signature
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
