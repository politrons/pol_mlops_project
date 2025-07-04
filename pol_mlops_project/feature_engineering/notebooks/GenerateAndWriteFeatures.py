# Databricks notebook source
"""
Generate and Write Features Notebook (two‑in‑one)
-------------------------------------------------
This notebook computes **two** Feature Store tables in a single run:
    • pol_dev.pol_mlops_project.trip_pickup_features
    • pol_dev.pol_mlops_project.trip_dropoff_features

It is intended to be executed as the task `write_feature_table_job` defined in
`pol_mlops_project/resources/feature-engineering-workflow-resource.yml`.

Global Parameters (widgets) ─ required for *both* targets
--------------------------------------------------------
* input_table_path   – Path to the source Delta table.
* input_start_date   – Optional lower bound for `timestamp_column`.
* input_end_date     – Optional upper bound for `timestamp_column`.
* primary_keys       – Comma‑separated primary key columns shared by all targets.

Target‑specific settings are kept in the in‑notebook list `targets` so that you
can easily add or remove feature tables by editing a single structure.
"""

# -------------------------------------------
# 0. Import and define widgets
# -------------------------------------------

dbutils.widgets.text(
    "input_table_path",
    "/databricks-datasets/nyctaxi-with-zipcodes/subsampled",
    label="Input Table Name",
)
dbutils.widgets.text("input_start_date", "", label="Input Start Date")
dbutils.widgets.text("input_end_date", "", label="Input End Date")
dbutils.widgets.text(
    "primary_keys", "zip", label="Primary key columns, comma‑separated."
)

# -------------------------------------------
# 1. Define the targets to generate
#    • Add a new dict here to create more tables.
# -------------------------------------------

targets = [
    {
        "output_table_name": "pol_dev.pol_mlops_project.trip_pickup_features",
        "timestamp_column": "tpep_pickup_datetime",
        "features_transform_module": "pickup_features",
    },
    {
        "output_table_name": "pol_dev.pol_mlops_project.trip_dropoff_features",
        "timestamp_column": "tpep_dropoff_datetime",
        "features_transform_module": "dropoff_features",
    },
]

# -------------------------------------------
# 2. Resolve global variables
# -------------------------------------------

input_table_path = dbutils.widgets.get("input_table_path")
input_start_date = dbutils.widgets.get("input_start_date")
input_end_date = dbutils.widgets.get("input_end_date")
primary_keys = dbutils.widgets.get("primary_keys")

assert input_table_path, "`input_table_path` widget must be provided."

# -------------------------------------------
# 3. Read the raw input data once
# -------------------------------------------

raw_data = spark.read.format("delta").load(input_table_path)

from databricks.feature_engineering import FeatureEngineeringClient
from importlib import import_module

fe = FeatureEngineeringClient()

# -------------------------------------------
# 4. Loop through each target and create/write its feature table
# -------------------------------------------

for t in targets:
    output_table_name = t["output_table_name"]
    ts_column = t["timestamp_column"]
    features_module = t["features_transform_module"]

    print(f"\n▶️  Processing {output_table_name} …")

    # 4.1  Dynamic import of the transformation logic
    mod = import_module(features_module)
    compute_features_fn = getattr(mod, "compute_features_fn")

    features_df = compute_features_fn(
        input_df=raw_data,
        timestamp_column=ts_column,
        start_date=input_start_date,
        end_date=input_end_date,
    )

    # 4.2  Create the Feature Store table if it does not exist.
    fe.create_table(
        name=output_table_name,
        primary_keys=[c.strip() for c in primary_keys.split(",")] + [ts_column],
        timestamp_keys=[ts_column],
        df=features_df,  # schema is inferred
    )

    # 4.3  Write or merge features into the table.
    fe.write_table(
        name=output_table_name,
        df=features_df,
        mode="merge",
    )

print("\n✅ Pickup and drop‑off feature tables successfully generated.")

dbutils.notebook.exit(0)
