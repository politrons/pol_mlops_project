
from mlflow.pyfunc import PythonModel
from mlflow.models import set_model

class CascadeABModel(PythonModel):
    def load_context(self, context):
        import mlflow

        self.model_a = mlflow.pyfunc.load_model(context.artifacts["model_a"])
        self.model_b = mlflow.pyfunc.load_model(context.artifacts["model_b"])


    def predict(self, context, model_input, params=None):
        # Accept pandas DataFrame or other JSONable shapes
        import pandas as pd
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        pred_a = self.model_a.predict(model_input)
        pred_b = self.model_b.predict(model_input)

        # Return dict (JSON-friendly) or DataFrame; choose what downstream expects.
        return {"pred_a": pred_a, "pred_b": pred_b}

set_model(CascadeABModel())
