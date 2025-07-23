# ---- Train RandomForestRegressor ----
import mlflow
from sklearn.ensemble import RandomForestRegressor

from .model_contract import ModelContract

class RandomForest(ModelContract):
    def get_model_algorithm(self):
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=123,
            n_jobs=-1,
        )

    def log_model(self, model, model_name, signature, input_example):
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example,
        )

model_contract = RandomForest()