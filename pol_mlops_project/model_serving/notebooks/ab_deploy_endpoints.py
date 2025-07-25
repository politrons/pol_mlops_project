# Databricks notebook source
# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os

from mlflow.utils.databricks_utils import dbutils

notebook_path = '/Workspace/' + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------
# Widgets

dbutils.widgets.text("model_name", "", label="Full UC Model Name (catalog.schema.registered_model)")
dbutils.widgets.text("model_endpoint", "", label="Serving Endpoint Name")
dbutils.widgets.text("alias_a", "champion", label="Alias A")
dbutils.widgets.text("alias_b", "challenger", label="Alias B")
dbutils.widgets.text("weight_a", "50", label="Traffic % for Alias A")
dbutils.widgets.text("weight_b", "50", label="Traffic % for Alias B")
dbutils.widgets.dropdown("workload_size", "Small", ["Small", "Medium", "Large"], label="Workload Size")
dbutils.widgets.dropdown("scale_to_zero", "True", ["True", "False"], label="Scale to Zero")

model_name     = dbutils.widgets.get("model_name")
model_endpoint = dbutils.widgets.get("model_endpoint")
alias_a        = dbutils.widgets.get("alias_a")
alias_b        = dbutils.widgets.get("alias_b")
weight_a       = int(dbutils.widgets.get("weight_a"))
weight_b       = int(dbutils.widgets.get("weight_b"))
workload_size  = dbutils.widgets.get("workload_size")
scale_to_zero  = dbutils.widgets.get("scale_to_zero") == "True"

assert weight_a + weight_b == 100, "weight_a + weight_b must be 100"

# COMMAND ----------
# Resolve model versions from aliases

from mlflow import MlflowClient

client = MlflowClient()

ver_a = client.get_model_version_by_alias(name=model_name, alias=alias_a).version
ver_b = client.get_model_version_by_alias(name=model_name, alias=alias_b).version

served_model_name_a = f"{alias_a.replace('-', '_')}"
served_model_name_b = f"{alias_b.replace('-', '_')}"

print(f"{alias_a=} -> version {ver_a}")
print(f"{alias_b=} -> version {ver_b}")

# COMMAND ----------
# Build endpoint config with A/B routes

fs_endpoint_config_dict = {
    "served_models": [
        {
            "name": served_model_name_a,
            "model_name": model_name,
            "model_version": ver_a,
            "workload_size": workload_size,
            "scale_to_zero_enabled": scale_to_zero,
        },
        {
            "name": served_model_name_b,
            "model_name": model_name,
            "model_version": ver_b,
            "workload_size": workload_size,
            "scale_to_zero_enabled": scale_to_zero,
        },
    ],
    "traffic_config": {
        "routes": [
            {"served_model_name": served_model_name_a, "traffic_percentage": weight_a},
            {"served_model_name": served_model_name_b, "traffic_percentage": weight_b},
        ]
    },
}

from databricks.sdk.service.serving import EndpointCoreConfigInput

fs_endpoint_config = EndpointCoreConfigInput.from_dict(fs_endpoint_config_dict)

# COMMAND ----------
# Create or update the endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput

w = WorkspaceClient()

def create_or_update_endpoint(endpoint_name: str, config: EndpointCoreConfigInput):
    try:
        w.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=config
        )
        print(f"✅ Created endpoint '{endpoint_name}'")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"ℹ️ Endpoint '{endpoint_name}' exists, updating config...")
            w.serving_endpoints.update_config_and_wait(
                name=endpoint_name,
                served_models=config.served_models,
                traffic_config=config.traffic_config
            )
            print(f"✅ Updated endpoint '{endpoint_name}'")
        else:
            raise

create_or_update_endpoint(model_endpoint, fs_endpoint_config)
print(f"Traffic split -> {served_model_name_a}: {weight_a}% | {served_model_name_b}: {weight_b}%")


# COMMAND ----------
# Helper: rebalance traffic later without touching model versions
from databricks.sdk.service.serving import EndpointCoreConfigInput

def rebalance_traffic(endpoint_name: str, a_pct: int, b_pct: int,
                      a_name: str = served_model_name_a, b_name: str = served_model_name_b):
    assert a_pct + b_pct == 100, "a_pct + b_pct must be 100"
    new_cfg = EndpointCoreConfigInput.from_dict({
        "traffic_config": {
            "routes": [
                {"served_model_name": a_name, "traffic_percentage": a_pct},
                {"served_model_name": b_name, "traffic_percentage": b_pct},
            ]
        }
    })
    w.serving_endpoints.update_config_and_wait(
        name=endpoint_name,
        traffic_config=new_cfg.traffic_config
    )
    print(f"✅ Rebalanced: {a_name}={a_pct}%, {b_name}={b_pct}%")

# Example:
# rebalance_traffic(model_endpoint, 90, 10)

# COMMAND ----------
# Helper: promote challenger -> champion (flip aliases) and roll traffic to 100/0

def promote_challenger_to_champion(model_name: str,
                                   champion_alias: str = "champion",
                                   challenger_alias: str = "challenger",
                                   endpoint_name: str = model_endpoint,
                                   pct: int = 100):
    c = MlflowClient()
    new_champion_v = c.get_model_version_by_alias(model_name, challenger_alias).version
    # point champion alias to challenger version
    c.set_registered_model_alias(model_name, champion_alias, new_champion_v)
    print(f"🔁 {challenger_alias} (v{new_champion_v}) -> {champion_alias}")
    # optional: update traffic to 100% champion
    rebalance_traffic(endpoint_name, pct, 100 - pct)

# Example:
# promote_challenger_to_champion(model_name)

