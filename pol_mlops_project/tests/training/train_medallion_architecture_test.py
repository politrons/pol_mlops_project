# Databricks notebook source
# COMMAND ----------

import os

from mlflow.utils.databricks_utils import dbutils

notebook_path = '/Workspace/' + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# MAGIC %pip install -r ../../requirements.txt
# MAGIC %pip install pytest

dbutils.library.restartPython()

# ── Load dependencies from features --------------------------------------------------------
import os
from pyspark.sql import SparkSession

notebook_path = '/Workspace/' + os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
print("📓 Notebook path:", notebook_path)
%cd $notebook_path

%cd ../../training/features
from importlib import import_module

mod = import_module("medallion_model_trainning")
train_lightgbm = getattr(mod, "train_lightgbm")

# tests/test_train_lightgbm.py
import pandas as pd

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.1,
    "num_leaves": 8,
    "min_data_in_leaf": 1,
    "min_data_in_bin": 1,
    "max_depth": -1,
    "min_split_gain": 0,
}

import numpy as np
import pandas as pd

rng  = np.random.default_rng(seed=0)

trip_distance = rng.uniform(0.5, 10.0, 200)

fare_amount = 3.0 * trip_distance + 1.0 + rng.normal(0.0, 0.1, 200)

pdf = pd.DataFrame(
    {
        "fare_amount": fare_amount,
        "trip_distance": trip_distance,
    }
)


model, rmse = train_lightgbm(pdf, params, num_boost_round=20)
assert rmse < 1.5, f"RMSE too high: {rmse}"

assert model.num_trees() == 20

