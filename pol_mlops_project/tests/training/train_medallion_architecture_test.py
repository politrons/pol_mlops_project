# Databricks notebook source
# COMMAND ----------

import os

from mlflow.utils.databricks_utils import dbutils

notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# MAGIC %pip install -r ../../requirements.txt
# MAGIC %pip install pytest

dbutils.library.restartPython()

# COMMAND ----------

# ── Load dependencies from features --------------------------------------------------------
import os

notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
print("📓 Notebook path:", notebook_path)
%cd $notebook_path

%cd ../../training/features
from importlib import import_module

mod = import_module("medallion_model_trainning")
train_lightgbm = getattr(mod, "train_lightgbm")

# tests/test_train_lightgbm.py
import pytest
import pandas as pd


@pytest.mark.parametrize(
    "params",
    [
        {"objective": "regression", "metric": "rmse", "num_leaves": 4},
        {"objective": "regression", "metric": "rmse", "num_leaves": 8},
    ],
)
def test_train_lightgbm_returns_reasonable_rmse(params):
    # tiny synthetic dataset: fare = 2 * distance + noise
    pdf = pd.DataFrame(
        {
            "fare_amount":   [4.0, 6.1, 8.2, 10.3, 12.1],
            "trip_distance": [1.0, 2.0, 3.0, 4.0, 5.0],
            "pickup_zip":    ["10001"] * 5,
            "dropoff_zip":   ["10002"] * 5,
        }
    )

    model, rmse = train_lightgbm(pdf, params, num_boost_round=20)
    # Expect near-perfect fit on such a simple linear relation
    assert rmse < 0.5, f"RMSE too high: {rmse}"
    assert model.num_trees() == 20
