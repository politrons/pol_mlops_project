from mlflow.pyfunc import PythonModel
from mlflow.models import set_model
from databricks.sdk import WorkspaceClient

class CascadeABModel(PythonModel):
    # Runs once when the serving container starts
    def load_context(self, context):
        cfg = context.model_config           # host & token were passed in model_config
        self.ws = WorkspaceClient(
            host=cfg["host"],
            token=cfg["token"]
        )

    # Runs on every inference request
    def predict(self, context, model_input, params=None):
        # Convert the incoming pandas DataFrame to list-of-dict records
        records = model_input.to_dict(orient="records")

        # Query the first downstream endpoint
        pred_a = self.ws.serving_endpoints.query(
            name="pol_endpoint",
            dataframe_records=records         # valid payload format
        ).predictions                         # list of predictions

        # Query the second downstream endpoint (same one in this example)
        pred_b = self.ws.serving_endpoints.query(
            name="pol_endpoint",
            dataframe_records=records
        ).predictions

        # Return what the caller expects—here, a JSON-friendly dict
        return {"pred_a": pred_a, "pred_b": pred_b}

set_model(CascadeABModel())
