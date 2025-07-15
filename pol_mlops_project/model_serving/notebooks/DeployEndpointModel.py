# Databricks notebook source

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os

from databricks.sdk import WorkspaceClient
from mlflow.utils.databricks_utils import dbutils

notebook_path = '/Workspace/' + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Unity Catalog registered model name to use for the trained mode.
dbutils.widgets.text(
    "model_name", "pol_dev.pol_mlops_project.pol_mlops_project-model", label="Full (Three-Level) Model Name"
)

model_name = dbutils.widgets.get("model_name")


from databricks.sdk.service.serving import EndpointCoreConfigInput
from mlflow import MlflowClient

client = MlflowClient()

fs_endpoint_name_online = "pol_endpoint"
fs_model_version_online = client.get_model_version_by_alias(name=model_name, alias="champion").version

fs_endpoint_config_dict = {
    "served_models":[
        {
            "model_name":model_name,
            "model_version":fs_model_version_online,
            "scale_to_zero_enable":True,
            "workload_size":"Small"
        }
    ]
}

fs_endpoint_config = EndpointCoreConfigInput.from_dict(fs_endpoint_config_dict)

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

try:
    w.serving_endpoints.create_and_wait(
        name=fs_endpoint_name_online,
        config=fs_endpoint_config
    )
    print(f"Creating endpoint {fs_endpoint_name_online} with models {model_name} version {fs_model_version_online}")
except Exception as e:
    if "already exists" in str(e):
        pass
    else:
        raise e
