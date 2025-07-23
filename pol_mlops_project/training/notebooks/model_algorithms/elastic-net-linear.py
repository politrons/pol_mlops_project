# ---- Train RandomForestRegressor ----
import mlflow
from sklearn.linear_model import ElasticNet

def get_model_algorithm():
    return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=123)

def log_model(model, model_name, signature, input_example):
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=model_name,
        signature=signature,
        input_example=input_example,
    )