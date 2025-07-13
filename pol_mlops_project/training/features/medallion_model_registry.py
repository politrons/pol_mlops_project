"""
All MLflow / Unity-Catalog interactions live here.
Makes it trivial to stub these calls in unit tests.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple

import mlflow
import mlflow.lightgbm
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import pandas as pd
import lightgbm as lgb

def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

def log_lightgbm_model(
    model: lgb.Booster,
    sample_X: pd.DataFrame,
    model_name: str,
    experiment: str,
    registry_uri: str = "databricks-uc",
) -> Tuple[str, int]:
    """
    Log + register the LightGBM model, return (model_uri, versions).
    """
    mlflow.set_registry_uri(registry_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run() as run:
        mlflow.lightgbm.autolog(log_models=False)
        signature = infer_signature(sample_X, model.predict(sample_X))

        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
        )

    versions  = get_latest_model_version(model_name)
    model_uri = f"models:/{model_name}/{versions}"
    return  model_uri, versions
