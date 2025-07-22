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



def get_training_data(raw_data):
    taxi_data = rounded_taxi_data(raw_data)
    print(f"After rounding: {taxi_data.count():,} rows")
    taxi_data.display()

    # COMMAND ----------
    # DBTITLE 1,Recompute engineered features (7 total)
    taxi_data = taxi_data.withColumn("pickup_ts_unix",   F.col("rounded_pickup_datetime").cast("long")) \
                           .withColumn("dropoff_ts_unix", F.col("rounded_dropoff_datetime").cast("long"))

    w_pickup_1h   = Window.partitionBy("pickup_zip").orderBy(F.col("pickup_ts_unix")).rangeBetween(-3600, 0)   # 1h = 3600s
    w_dropoff_30m = Window.partitionBy("dropoff_zip").orderBy(F.col("dropoff_ts_unix")).rangeBetween(-1800, 0)  # 30m = 1800s

    taxi_data_fe = (
        taxi_data
        .withColumn("mean_fare_window_1h_pickup_zip",       F.avg("fare_amount").over(w_pickup_1h))
        .withColumn("count_trips_window_1h_pickup_zip",      F.count(F.lit(1)).over(w_pickup_1h))
        .withColumn("count_trips_window_30m_dropoff_zip",    F.count(F.lit(1)).over(w_dropoff_30m))
        .withColumn("dropoff_is_weekend",                    F.when(F.dayofweek("rounded_dropoff_datetime").isin(1,7), 1).otherwise(0))
    )

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
    # if sample_fraction < 1.0:
    #     train_df_spark = train_df_spark.sample(withReplacement=False, fraction=sample_fraction, seed=42)
    #     print(f"Sampled fraction {sample_fraction}; rows: {train_df_spark.count():,}")

    train_df_spark.display()
    train_pdf = train_df_spark.toPandas()
    print("Pandas shape:", train_pdf.shape)

    # Cast label
    train_pdf[LABEL_COL] = train_pdf[LABEL_COL].astype(float)

    # Cast all features to numeric float (LightGBM friendly)
    for c in feature_cols:
        train_pdf[c] = train_pdf[c].astype(float)

    X_pdf = train_pdf[feature_cols]
    y_pdf = train_pdf[LABEL_COL]

    print("Train X shape:", X_pdf.shape)
    print("Feature cols:", feature_cols)
    return X_pdf, y_pdf