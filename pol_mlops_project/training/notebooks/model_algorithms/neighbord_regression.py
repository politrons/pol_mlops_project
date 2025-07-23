import mlflow
from sklearn.neighbors import KNeighborsRegressor

def get_model_algorithm():
    return KNeighborsRegressor(
        n_neighbors=10,
        weights="distance",
        metric="minkowski",
    )

def log_model(model, model_name, signature, input_example):
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=model_name,
        signature=signature,
        input_example=input_example,
    )