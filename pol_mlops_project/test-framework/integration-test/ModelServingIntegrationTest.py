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
    "model_endpoint", "", label="model_endpoint"
)

dbutils.widgets.text("payload_path", "",
                     label="payload_path")

model_endpoint = dbutils.widgets.get("model_endpoint")

from databricks.sdk import WorkspaceClient
from importlib import import_module

w = WorkspaceClient()

mod = import_module("payload." + dbutils.widgets.get("payload_path"))

# Call the endpoint and get the response
response = w.serving_endpoints.query(
    name=model_endpoint,
    dataframe_split=mod.payload,

)

# Extract the first prediction from the list
print("Response: ", response)
# prediction = float(response.predictions[0])
# prediction
# Assert that the prediction is within the expected range
# assert 8.0 <= prediction <= 9.0, f"Value out of range: {prediction}"

# print("✅ Prediction is within the range [8, 9]")


