# Databricks notebook source
################################################################################
# Model-Training Notebook (Feature Store + Optuna hyper-parameter tuning)
# “Ultra-minimal” version: no MLflow callback, nested run per trial, and the
# LightGBM base-params list contains only what’s strictly required.
################################################################################

# COMMAND ----------
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------
import os
notebook_path = "/Workspace/" + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .notebookPath()
    .get()
)
%cd $notebook_path

# COMMAND ----------
# Install dependencies
# MAGIC %pip install -r ../../requirements.txt
# MAGIC %pip install --quiet "optuna==3.6.0" "optuna-integration==3.6.0" "lightgbm==4.3.0"

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# Notebook parameters
dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment")
env = dbutils.widgets.get("env")

dbutils.widgets.text("training_data_path",
                     "/databricks-datasets/nyctaxi-with-zipcodes/subsampled",
                     label="Training data path")
dbutils.widgets.text("experiment_name",
                     "/pol_dev-pol_mlops_project-experiment",
                     label="MLflow experiment")
dbutils.widgets.text("model_name",
                     "pol_dev.pol_mlops_project.pol_mlops_project-model",
                     label="Model name (catalog.schema.name)")
dbutils.widgets.text("pickup_features_table",
                     "pol_dev.pol_mlops_project.trip_pickup_features",
                     label="Pickup features table")
dbutils.widgets.text("dropoff_features_table",
                     "pol_dev.pol_mlops_project.trip_dropoff_features",
                     label="Drop-off features table")

# COMMAND ----------
# Retrieve widget values
input_table_path        = dbutils.widgets.get("training_data_path")
experiment_name         = dbutils.widgets.get("experiment_name")
model_name              = dbutils.widgets.get("model_name")
pickup_features_table   = dbutils.widgets.get("pickup_features_table")
dropoff_features_table  = dbutils.widgets.get("dropoff_features_table")

# COMMAND ----------
# Initialise MLflow
import mlflow
mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
# Load raw data
raw_data = spark.read.format("delta").load(input_table_path)

# COMMAND ----------
# Helper functions
from datetime import timedelta, timezone
import math
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from mlflow.tracking import MlflowClient

def rounded_unix_timestamp(dt, minutes=15):
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(nsecs / (60 * minutes)) * (60 * minutes) - nsecs
    return int((dt + timedelta(seconds=delta)).replace(tzinfo=timezone.utc).timestamp())

rounded_unix_timestamp_udf = F.udf(rounded_unix_timestamp, IntegerType())

def rounded_taxi_data(df):
    return (
        df.withColumn(
            "rounded_pickup_datetime",
            F.to_timestamp(rounded_unix_timestamp_udf(df["tpep_pickup_datetime"], F.lit(15))),
        )
        .withColumn(
            "rounded_dropoff_datetime",
            F.to_timestamp(rounded_unix_timestamp_udf(df["tpep_dropoff_datetime"], F.lit(30))),
        )
        .drop("tpep_pickup_datetime")
        .drop("tpep_dropoff_datetime")
    )

def get_latest_model_version(name: str) -> int:
    latest = 1
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{name}'"):
        latest = max(latest, int(mv.version))
    return latest

# COMMAND ----------
taxi_data = rounded_taxi_data(raw_data)

# COMMAND ----------
# Feature look-ups
from databricks.feature_engineering import FeatureLookup

pickup_feature_lookups = [
    FeatureLookup(
        table_name=pickup_features_table,
        feature_names=[
            "mean_fare_window_1h_pickup_zip",
            "count_trips_window_1h_pickup_zip",
        ],
        lookup_key=["pickup_zip"],
        timestamp_lookup_key=["rounded_pickup_datetime"],
    ),
]
dropoff_feature_lookups = [
    FeatureLookup(
        table_name=dropoff_features_table,
        feature_names=[
            "count_trips_window_30m_dropoff_zip",
            "dropoff_is_weekend",
        ],
        lookup_key=["dropoff_zip"],
        timestamp_lookup_key=["rounded_dropoff_datetime"],
    ),
]

# COMMAND ----------
# Build TrainingSet
from databricks.feature_engineering import FeatureEngineeringClient

mlflow.end_run()
mlflow.start_run(run_name="optuna_lightgbm_training")

fe = FeatureEngineeringClient()
training_set = fe.create_training_set(
    df=taxi_data,
    feature_lookups=pickup_feature_lookups + dropoff_feature_lookups,
    label="fare_amount",
    exclude_columns=["rounded_pickup_datetime", "rounded_dropoff_datetime"],
)
training_df = training_set.load_df()

# COMMAND ----------
# --------------------  Optuna hyper-parameter search  -------------------------
import optuna
from optuna_integration.lightgbm import LightGBMPruningCallback
import lightgbm as lgb
from sklearn.model_selection import train_test_split

pdf = training_df.toPandas()[training_df.columns]
train_df, valid_df = train_test_split(pdf, random_state=123, test_size=0.2)
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

# COMMAND ----------
# Retrain final model with best params
mlflow.end_run()
mlflow.start_run(run_name="best_lgbm_model")
mlflow.lightgbm.autolog()

best_params = BASE_PARAMS | study.best_params
best_model = lgb.train(
    best_params,
    train_set=lgb_train,
    num_boost_round=5000,
    valid_sets=[lgb_valid],
)

# COMMAND ----------
# Register model
fe.log_model(
    model=best_model,
    artifact_path="model_packaged",
    flavor=mlflow.lightgbm,
    training_set=training_set,
    registered_model_name=model_name,
)

# COMMAND ----------
model_version = get_latest_model_version(model_name)
model_uri     = f"models:/{model_name}/{model_version}"

dbutils.jobs.taskValues.set("model_uri",     model_uri)
dbutils.jobs.taskValues.set("model_name",    model_name)
dbutils.jobs.taskValues.set("model_version", model_version)

dbutils.notebook.exit(model_uri)
