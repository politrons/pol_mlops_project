# Delta Live Tables is Databricks’ fully managed, declarative pipeline engine that turns simple SQL or Python definitions
# into continuously updated, ACID-compliant Delta tables—automatically handling orchestration, scaling, data-quality enforcement,
# lineage tracking, and fault recovery, so you can focus purely on business logic instead of pipeline plumbing.
import dlt
from pyspark.sql import functions as F
from pyspark.sql.functions import col, avg


# ------------------------------------------------------------------
# Bronze
# ------------------------------------------------------------------
@dlt.table(name="bronze_nyc_taxi",
           comment="Raw NYC taxi demo subset",
           table_properties={"quality": "bronze"})
def bronze():
    return (
        spark.readStream.format("delta")
        .load("/databricks-datasets/nyctaxi-with-zipcodes/subsampled")
        .withColumn("ingest_ts", F.current_timestamp())
    )


# ------------------------------------------------------------------
# Silver
# ------------------------------------------------------------------
@dlt.table(name="silver_nyc_taxi",
           comment="Cleaned trips (positive fare)",
           table_properties={"quality": "silver"})
def silver():
    df = dlt.readStream("bronze_nyc_taxi")
    if "trip_distance" in df.columns:
        df = df.filter(col("trip_distance") > 0)
    return (
        df.filter(col("fare_amount") > 0)
        .select("fare_amount", "pickup_zip", "dropoff_zip")
    )


# ------------------------------------------------------------------
# Pickup feature
# ------------------------------------------------------------------
def compute_trip_pickup_features(df):
    return (
        df
        .groupBy("pickup_zip")
        .agg(avg("fare_amount").alias("avg_fare_per_zip"))
        .withColumnRenamed("pickup_zip", "zip")  # PK = zip
    )


@dlt.table(name="trip_pickup_features",
           comment="Average fare per pickup ZIP",
           table_properties={"quality": "gold"})
def trip_pickup_features():
    silver_df = dlt.readStream("silver_nyc_taxi")
    return compute_trip_pickup_features(silver_df)


# ------------------------------------------------------------------
# Drop-off feature
# ------------------------------------------------------------------
def compute_trip_dropoff_features(df):
    return (
        df.filter((F.col("fare_amount") > 0) & (F.col("trip_distance") > 0))
        .select("fare_amount", "pickup_zip", "dropoff_zip")
    )


# --------------------------------------------------------------------------- #
# 2. DLT wrapper (production entry-point)                                     #
# --------------------------------------------------------------------------- #
@dlt.table(
    name="trip_dropoff_features",
    comment="Gold-quality features for ML models (one row per trip drop-off)",
    table_properties={"quality": "gold"},
)
def trip_dropoff_features():
    """
    Delta Live Tables wrapper around `compute_trip_dropoff_features`.

    DLT injects a streaming DataFrame from the silver layer; the wrapper simply
    delegates to the pure function so that business logic remains testable.
    """
    silver_df = dlt.readStream("silver_nyc_taxi")
    return compute_trip_dropoff_features(silver_df)


# ------------------------------------------------------------------
# Gold table – model-ready
# ------------------------------------------------------------------
@dlt.table(name="gold_nyc_taxi",
           comment="Denormalised training set",
           table_properties={"quality": "gold"})
def gold():
    silver = dlt.readStream("silver_nyc_taxi")
    pickup = dlt.read("trip_pickup_features").withColumnRenamed("zip", "pickup_zip")
    dropoff = dlt.read("trip_dropoff_features")
    return (
        silver.join(pickup, "pickup_zip", "left")
        .join(dropoff, "dropoff_zip", "left")
        .select("fare_amount", "pickup_zip", "dropoff_zip",
                "avg_fare_per_zip", "trip_count")
    )
