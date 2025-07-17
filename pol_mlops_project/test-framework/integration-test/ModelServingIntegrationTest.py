# Databricks notebook source

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os

from mlflow.utils.databricks_utils import dbutils

notebook_path = '/Workspace/' + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

dbutils.widgets.text(
    "model_endpoint", "pol_endpoint", label="model_endpoint"
)

model_endpoint = dbutils.widgets.get("model_endpoint")


from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import DataframeSplitInput

w = WorkspaceClient()

payload = DataframeSplitInput(
    columns=[
        "trip_distance",
        "pickup_zip",
        "dropoff_zip",
        "mean_fare_window_1h_pickup_zip",
        "count_trips_window_1h_pickup_zip",
        "count_trips_window_30m_dropoff_zip",
        "dropoff_is_weekend",
    ],
    data=[[2.5, 7002, 7002, 8.5, 1, 1, 0]],
)

# Call the endpoint and get the response
response = w.serving_endpoints.query(
    name=model_endpoint,
    dataframe_split=payload,

)

# Extract the first prediction from the list
print("Response from Serving endpoint: ", response)
# prediction = float(response.predictions[0])
# prediction
# Assert that the prediction is within the expected range
# assert 8.0 <= prediction <= 9.0, f"Value out of range: {prediction}"

# print("✅ Prediction is within the range [8, 9]")


