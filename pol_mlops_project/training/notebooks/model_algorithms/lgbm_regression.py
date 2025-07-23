import mlflow
import lightgbm as lgb

from .model_contract import ModelContract

class LGBMRegression(ModelContract):
    def get_model_algorithm(self):
        return lgb.LGBMRegressor(
            num_leaves=32,
            objective="regression",
            n_estimators=100,
        )

    def log_model(self, model, model_name, signature, input_example):
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example,
        )

model_contract = LGBMRegression()