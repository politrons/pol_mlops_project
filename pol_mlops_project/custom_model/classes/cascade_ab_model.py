import os
import pandas as pd
import mlflow

class CascadeABModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_a_uri: str, model_b_uri: str):
        self.model_a_uri = model_a_uri
        self.model_b_uri = model_b_uri
        self.model_a = None
        self.model_b = None

    def load_context(self, context):
        # Load both sub-models once when the serving container starts
        self.model_a = mlflow.pyfunc.load_model(self.model_a_uri)
        self.model_b = mlflow.pyfunc.load_model(self.model_b_uri)

    def predict(self, context, model_input):
        print("New request received")
        # Ensure DataFrame
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        # Run Model A
        pred_a = pd.Series(self.model_a.predict(model_input), index=model_input.index, name="pred_a")

        # Run Model B using A's output
        pred_b = pd.Series(self.model_b.predict(model_input), index=model_input.index, name="pred_b")

        # Return combined result
        return pd.DataFrame({"pred_a": pred_a, "pred_b": pred_b})