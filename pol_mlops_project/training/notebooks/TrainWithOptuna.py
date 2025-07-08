# Databricks notebook source
##################################################################################
# Medallion Architecture + Model-Training Notebook
#
# 1. Bronze : raw → Delta
# 2. Silver : cleaning and transformations
# 3. Gold   : feature engineering → Feature Store + LightGBM training
#
# Widget parameters:
#   * catalog
#   * schema
#   * experiment_name
#   * model_name
##################################################################################

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
# Widgets
dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment")
env = dbutils.widgets.get("env")
dbutils.widgets.text("catalog", "pol_dev", "Catalog")
dbutils.widgets.text("schema", "pol_mlops_project", "Schema")
dbutils.widgets.text("experiment_name", "/pol_dev-pol_mlops_project-experiment",  "MLflow Experiment")
dbutils.widgets.text(
    "model_name", "pol_dev.pol_mlops_project.pol_mlops_project-model", label="Full (Three-Level) Model Name"
)

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.functions import col, avg
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from mlflow.tracking import MlflowClient
import mlflow.lightgbm
import mlflow
import lightgbm as lgb
from sklearn.model_selection import train_test_split

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")

mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri("databricks-uc")

# -----------------
# Bronze layer
# -----------------
bronze_table = f"{catalog}.{schema}.bronze_nyc_taxi"
raw_df = (
    spark.read.format("delta")
    .load("/databricks-datasets/nyctaxi-with-zipcodes/subsampled")
)
raw_df.write.mode("overwrite").format("delta").saveAsTable(bronze_table)

# -----------------
# Silver layer
# -----------------
silver_table = f"{catalog}.{schema}.silver_nyc_taxi"
silver_df = (
    raw_df
    .filter(col("trip_distance") > 0)
    .filter(col("fare_amount") > 0)
    .select("fare_amount", "pickup_zip", "dropoff_zip",
            "tpep_pickup_datetime", "tpep_dropoff_datetime")
)
silver_df.write.mode("overwrite").format("delta").saveAsTable(silver_table)

# ----------------------------
# Gold layer – feature tables
# ----------------------------
from databricks.feature_engineering import FeatureEngineeringClient

pickup_features_table = f"{catalog}.{schema}.trip_pickup_features"
dropoff_features_table = f"{catalog}.{schema}.trip_dropoff_features"

featureEngineering = FeatureEngineeringClient()

# ---- PICKUP FEATURES ----------------------------------------------------------
pickup_features = (
    silver_df
    .groupBy("pickup_zip")
    .agg(avg("fare_amount")
         .alias("avg_fare_per_zip"))
    .withColumnRenamed("pickup_zip", "zip")  # primary key column
)

# Drop any old table (removes stale schema & constraints)
spark.sql(f"DROP TABLE IF EXISTS {pickup_features_table}")

# Re-create as a feature table with PK = zip
featureEngineering.create_table(
    name=pickup_features_table,
    primary_keys=["zip"],
    df=pickup_features,
)
# Overwrite with latest data
featureEngineering.write_table(
    name=pickup_features_table,
    df=pickup_features,
    mode="merge",
)

# ---- DROPOFF FEATURES ---------------------------------------------------------
dropoff_features = (
    silver_df
    .groupBy("dropoff_zip")
    .count()
    .withColumnRenamed("count", "trip_count")  # keep original PK = dropoff_zip
)

spark.sql(f"DROP TABLE IF EXISTS {dropoff_features_table}")

featureEngineering.create_table(
    name=dropoff_features_table,
    primary_keys=["dropoff_zip"],
    df=dropoff_features,
)

featureEngineering.write_table(
    name=dropoff_features_table,
    df=dropoff_features,
    mode="merge",
)

# COMMAND ----------
# -----------------
# Build the training set
# -----------------
from datetime import timedelta, timezone
import math
from pyspark.sql.types import IntegerType


def rounded_unix_timestamp(dt, num_minutes=15):
    """Ceil dt to the next *num_minutes* bucket and return seconds since epoch."""
    secs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(secs / (60 * num_minutes)) * (60 * num_minutes) - secs
    return int((dt + timedelta(seconds=delta)).replace(tzinfo=timezone.utc).timestamp())


def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


rounded_unix_timestamp_udf = F.udf(rounded_unix_timestamp, IntegerType())

rounded_df = (
    silver_df
    .withColumn(
        "rounded_pickup_datetime",
        F.to_timestamp(rounded_unix_timestamp_udf(col("tpep_pickup_datetime"), F.lit(15)))
    )
    .withColumn(
        "rounded_dropoff_datetime",
        F.to_timestamp(rounded_unix_timestamp_udf(col("tpep_dropoff_datetime"), F.lit(30)))
    )
    .drop("tpep_pickup_datetime", "tpep_dropoff_datetime")
    .withColumn("zip", col("pickup_zip"))  # helper key for feature lookup
)

# COMMAND ----------
# -----------------
# Feature lookups
# -----------------
pickup_lookup = FeatureLookup(
    table_name=pickup_features_table,
    feature_names=["avg_fare_per_zip"],
    lookup_key=["zip"],
)

dropoff_lookup = FeatureLookup(
    table_name=dropoff_features_table,
    feature_names=["trip_count"],
    lookup_key=["dropoff_zip"],
)

# COMMAND ----------
# -----------------
# Train the model
# -----------------
featureEngineering = FeatureEngineeringClient()
mlflow.end_run()
mlflow.start_run()

training_set = featureEngineering.create_training_set(
    df=rounded_df,
    feature_lookups=[pickup_lookup, dropoff_lookup],
    label="fare_amount",
    exclude_columns=[
        "rounded_pickup_datetime", "rounded_dropoff_datetime", "zip"
    ],
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
featureEngineering.log_model(
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
