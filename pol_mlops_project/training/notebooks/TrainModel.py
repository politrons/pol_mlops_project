# Databricks notebook source

# COMMAND ----------
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------
import os

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

# (Widgets preservados por compatibilidad; no usados pero mantenidos por jobs)
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

# Extra: fracción de muestreo opcional para entrenamiento rápido (0<frac<=1)
dbutils.widgets.text("sample_fraction", "1.0", label="Sample fraction [0-1]")

# COMMAND ----------
# DBTITLE 1,Capture widget values
input_table_path = dbutils.widgets.get("training_data_path")
experiment_name  = dbutils.widgets.get("experiment_name")
model_name       = dbutils.widgets.get("model_name")
sample_fraction  = float(dbutils.widgets.get("sample_fraction"))

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
# DBTITLE 1,Helper functions
from datetime import timedelta, timezone
import math
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql import Window


def rounded_unix_timestamp(dt, num_minutes=15):
    """
    Ceil datetime dt to the upper multiple of num_minutes, return unix ts (int seconds, UTC).
    Esto imita la lógica del notebook FE original para alinear joins de ventana.
    """
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(nsecs / (60 * num_minutes)) * (60 * num_minutes) - nsecs
    return int((dt + timedelta(seconds=delta)).replace(tzinfo=timezone.utc).timestamp())


rounded_unix_timestamp_udf = F.udf(rounded_unix_timestamp, IntegerType())


def rounded_taxi_data(taxi_data_df):
    """Añade columnas redondeadas pickup/dropoff para alineación de ventanas."""
    taxi_data_df = (
        taxi_data_df.withColumn(
            "rounded_pickup_datetime",
            F.to_timestamp(
                rounded_unix_timestamp_udf(
                    taxi_data_df["tpep_pickup_datetime"], F.lit(15)
                )
            ),
        )
        .withColumn(
            "rounded_dropoff_datetime",
            F.to_timestamp(
                rounded_unix_timestamp_udf(
                    taxi_data_df["tpep_dropoff_datetime"], F.lit(30)
                )
            ),
        )
        .drop("tpep_pickup_datetime")
        .drop("tpep_dropoff_datetime")
    )
    taxi_data_df.createOrReplaceTempView("taxi_data")
    return taxi_data_df


def get_latest_model_version(model_name: str) -> int:
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------
# DBTITLE 1,Round timestamps (as FE notebook)
taxi_data = rounded_taxi_data(raw_data)
print(f"After rounding: {taxi_data.count():,} rows")
taxi_data.display()

# COMMAND ----------
# DBTITLE 1,Recompute engineered features (7 total)
# Emulamos las 4 columnas que antes venían de la Feature Store.
# Usamos ventanas analíticas _trailing_ basadas en segundos.

# Para usar rangeBetween con segundos, convertimos timestamps a long (segundos Unix).
taxi_data = taxi_data.withColumn("pickup_ts_unix",   F.col("rounded_pickup_datetime").cast("long")) \
                       .withColumn("dropoff_ts_unix", F.col("rounded_dropoff_datetime").cast("long"))

# Ventanas:
w_pickup_1h   = Window.partitionBy("pickup_zip").orderBy(F.col("pickup_ts_unix")).rangeBetween(-3600, 0)   # 1h = 3600s
w_dropoff_30m = Window.partitionBy("dropoff_zip").orderBy(F.col("dropoff_ts_unix")).rangeBetween(-1800, 0)  # 30m = 1800s

# Agregados:
taxi_data_fe = (
    taxi_data
    .withColumn("mean_fare_window_1h_pickup_zip",       F.avg("fare_amount").over(w_pickup_1h))
    .withColumn("count_trips_window_1h_pickup_zip",      F.count(F.lit(1)).over(w_pickup_1h))
    .withColumn("count_trips_window_30m_dropoff_zip",    F.count(F.lit(1)).over(w_dropoff_30m))
    .withColumn("dropoff_is_weekend",                    F.when(F.dayofweek("rounded_dropoff_datetime").isin(1,7), 1).otherwise(0))
)

# Limpia columnas auxiliares si no las quieres persistir en entrenamiento final
taxi_data_fe.cache()
print(f"Rows after FE: {taxi_data_fe.count():,}")
taxi_data_fe.display()

# COMMAND ----------
# DBTITLE 1,Seleccionar columnas finales de entrenamiento
from pyspark.sql import types as T

LABEL_COL = "fare_amount"
DROP_COLS = ["rounded_pickup_datetime", "rounded_dropoff_datetime", "pickup_ts_unix", "dropoff_ts_unix"]

feature_cols = [
    "trip_distance",
    "pickup_zip",
    "dropoff_zip",
    "mean_fare_window_1h_pickup_zip",
    "count_trips_window_1h_pickup_zip",
    "count_trips_window_30m_dropoff_zip",
    "dropoff_is_weekend",
]

select_cols = [LABEL_COL] + feature_cols

train_df_spark = taxi_data_fe.select(*select_cols)

# Opcional: sample for faster iteration
if sample_fraction < 1.0:
    train_df_spark = train_df_spark.sample(withReplacement=False, fraction=sample_fraction, seed=42)
    print(f"Sampled fraction {sample_fraction}; rows: {train_df_spark.count():,}")

train_df_spark.display()

# COMMAND ----------
# DBTITLE 1,Spark -> Pandas + type cleanup
# ADVERTENCIA: para datasets grandes, considera usar LightGBM Spark o incremental training.
train_pdf = train_df_spark.toPandas()
print("Pandas shape:", train_pdf.shape)

# Cast label
train_pdf[LABEL_COL] = train_pdf[LABEL_COL].astype(float)

# Cast all features to numeric float (LightGBM friendly)
for c in feature_cols:
    train_pdf[c] = train_pdf[c].astype(float)

# (Si pickup_zip/dropoff_zip deberían ser categóricos, podrías dejarlos tal cual; LightGBM puede trabajar con float-coded IDs)

X_pdf = train_pdf[feature_cols]
y_pdf = train_pdf[LABEL_COL]

print("Train X shape:", X_pdf.shape)
print("Feature cols:", feature_cols)

# COMMAND ----------
# DBTITLE 1,Train LightGBM
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

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

# Sample (limit para no loggear enorme)
sample_input = X_train.iloc[:100].copy()
sample_output = model.predict(sample_input)

# Para salida nombrada (recomendado):
import pandas as pd
sample_output_df = pd.DataFrame({"prediction": sample_output})

signature = infer_signature(sample_input, sample_output_df)
input_example = sample_input.head(5)

print("Signature inferred:")
print(signature)

# COMMAND ----------
# DBTITLE 1,Log model (MLflow LightGBM flavor + Registry)
import json

# Fin de run previo si existiera
mlflow.end_run()

with mlflow.start_run(run_name=f"train_plain_7fe_{env}") as run:
    run_id = run.info.run_id
    # Params / metrics
    mlflow.log_param("env", env)
    mlflow.log_param("training_data_path", input_table_path)
    mlflow.log_param("feature_engineering_inline", True)
    mlflow.log_metric("rmse", rmse)

    # Cols usadas
    mlflow.log_dict({"features": feature_cols, "label": LABEL_COL}, "training_columns.json")

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

# Propaga valores a jobs encadenados
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)

# COMMAND ----------
# Finaliza notebook con URI (permite chaining en workflows)
dbutils.notebook.exit(model_uri)
