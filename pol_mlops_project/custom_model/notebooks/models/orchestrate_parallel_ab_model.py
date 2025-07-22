import asyncio

from mlflow.pyfunc import PythonModel
from mlflow.models import set_model
from databricks.sdk import WorkspaceClient
from typing import  Optional, Any


class CascadeABModel(PythonModel):
    # Runs once when the serving container starts
    def load_context(self, context):
        cfg = context.model_config  # host & token were passed in model_config
        self.ws = WorkspaceClient(
            host=cfg["host"],
            token=cfg["token"]
        )

    async def run_process(self, records) -> Optional[list]:
        response = self.ws.serving_endpoints.query(
            name="pol_endpoint",
            dataframe_records=records).predictions
        return response

    async def run_parallel_process(self, records) -> dict[str, Any]:
        # Convert the incoming pandas DataFrame to list-of-dict records
        coroutine_a = self.run_process(records)
        coroutine_b = self.run_process(records)
        return {"pred_a": await coroutine_a, "pred_b": await coroutine_b}

    # Runs on every inference request
    def predict(self, context, model_input, params=None):
        # Convert the incoming pandas DataFrame to list-of-dict records
        records = model_input.to_dict(orient="records")
        return asyncio.run(self.run_parallel_process(records))


set_model(CascadeABModel())
