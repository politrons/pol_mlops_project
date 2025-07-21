from mlflow.pyfunc import PythonModel
from mlflow.models import set_model
from databricks.sdk import WorkspaceClient
from multiprocessing import Process, Manager

from numpy.core.multiarray import shares_memory


class CascadeABModel(PythonModel):
    # Runs once when the serving container starts
    def load_context(self, context):
        cfg = context.model_config  # host & token were passed in model_config
        self.ws = WorkspaceClient(
            host=cfg["host"],
            token=cfg["token"]
        )

    def run_process(self, idx:int, shared, records):
        process = self.ws.serving_endpoints.query(
            name="pol_endpoint",
            dataframe_records=records).predictions
        shared[idx] = process

    # Runs on every inference request
    def predict(self, context, model_input, params=None):
        # Convert the incoming pandas DataFrame to list-of-dict records
        records = model_input.to_dict(orient="records")

        with Manager() as manager:
            shared_results = manager.dict()
            # Parallel computing with Processor
            process_a = Process(target=self.run_process(1, shared_results, records))
            process_b = Process(target=self.run_process(2, shared_results, records))

            process_a.start()
            process_b.start()

            process_a.join()
            process_b.join()

            # Return what the caller expects—here, a JSON-friendly dict
            return {"pred_a": shared_results[1], "pred_b": shared_results[2]}


set_model(CascadeABModel())
