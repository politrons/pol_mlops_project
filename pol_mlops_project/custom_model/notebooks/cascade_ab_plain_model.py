from mlflow.models import set_model
import mlflow.pyfunc
import mlflow.lightgbm
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)  # escribe a stderr por defecto
log = logging.getLogger("CascadeABPlainModel")

class CascadeABPlainModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        log.info("Loading submodels...")
        self.model_a = mlflow.lightgbm.load_model(context.artifacts["model_a"])
        self.model_b = mlflow.lightgbm.load_model(context.artifacts["model_b"])
        log.info("Submodels loaded.")

    def predict(self, context, model_input, params=None):
        log.info("New request received.")
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        log.debug("Input shape=%s cols=%s", model_input.shape, list(model_input.columns))

        log.info("Calling model A.")
        pred_a = self.model_a.predict(model_input)

        log.info("Calling model B.")
        pred_b = self.model_b.predict(model_input)

        log.info("Returning predictions.")
        return {"pred_a": pred_a, "pred_b": pred_b}


set_model(CascadeABPlainModel())
