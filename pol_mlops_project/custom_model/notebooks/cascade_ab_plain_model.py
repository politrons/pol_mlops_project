
from mlflow.models import set_model
import mlflow.lightgbm

class CascadeABPlainModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # context.artifacts["model_a"] y ["model_b"] son rutas locales al directorio del modelo
        self.model_a = mlflow.lightgbm.load_model(context.artifacts["model_a"])
        self.model_b = mlflow.lightgbm.load_model(context.artifacts["model_b"])

    def predict(self, context, model_input, params=None):
        print("New request received")
        import pandas as pd
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        # LightGBM sklearn API: .predict(X)
        print("Calling model A")
        pred_a = self.model_a.predict(model_input)
        print("Calling model B")
        pred_b = self.model_b.predict(model_input)

        return {"pred_a": pred_a, "pred_b": pred_b}


set_model(CascadeABPlainModel())
