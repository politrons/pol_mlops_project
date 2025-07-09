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
# Widgets
dbutils.widgets.text("catalog", "pol_dev", "Catalog")
dbutils.widgets.text("schema", "pol_mlops_project", "Schema")
dbutils.widgets.text("experiment_name", "/pol_dev-pol_mlops_project-experiment",
                     "MLflow Experiment")
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

train_df = training_set.load_df()

# Save the gold table (optional)
train_df.write.mode("overwrite").option("overwriteSchema", "true") \
    .format("delta").saveAsTable(f"{catalog}.{schema}.gold_nyc_taxi")

data = train_df.toPandas()
train, test = train_test_split(data, random_state=42)
X_train, y_train = train.drop("fare_amount", axis=1), train["fare_amount"]
X_test, y_test = test.drop("fare_amount", axis=1), test["fare_amount"]

params = {"objective": "regression", "metric": "rmse", "num_leaves": 32}
model = lgb.train(params, lgb.Dataset(X_train, label=y_train), num_boost_round=50)

# COMMAND ----------
# -----------------
# Log and register the model
# -----------------
featureEngineering.log_model(
    model=model,  # specify model
    artifact_path="model_packaged",
    flavor=mlflow.lightgbm,
    training_set=training_set,
    registered_model_name=model_name,
)

# The returned model URI is needed by the model deployment notebook.
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.notebook.exit(model_uri)
