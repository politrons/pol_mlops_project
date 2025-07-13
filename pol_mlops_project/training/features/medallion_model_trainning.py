"""
Pure training utilities – NO Spark, NO dbutils, NO MLflow.
Everything here can be imported and unit-tested locally or in Databricks.
"""
from __future__ import annotations
from typing import Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def split_xy(pdf: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features from target (fare_amount)."""
    return pdf.drop("fare_amount", axis=1), pdf["fare_amount"]


def train_lightgbm(
    pdf: pd.DataFrame,
    params: Dict,
    num_boost_round: int = 50,
    seed: int = 42,
) -> Tuple[lgb.Booster, float]:
    """
    Train LightGBM and return (model, RMSE_on_test).

    Returns
    -------
    model  : trained LightGBM Booster
    rmse   : float – RMSE on held-out test set
    """
    train_df, test_df = train_test_split(pdf, random_state=seed, test_size=0.2)
    X_train, y_train = split_xy(train_df)
    X_test, y_test   = split_xy(test_df)

    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=num_boost_round,
    )

    preds = model.predict(X_test)
    rmse  = mean_squared_error(y_test, preds, squared=False)
    return model, rmse
