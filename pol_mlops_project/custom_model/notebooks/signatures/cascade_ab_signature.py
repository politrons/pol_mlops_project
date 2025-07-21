from mlflow.models import infer_signature, ModelSignature
import pandas as pd

# ------------------------------------------------------------------
# Build a small pandas DataFrame that matches your serving payload
# (use realistic types; multiple rows help catch dtype issues)
# ------------------------------------------------------------------
sample_df = pd.DataFrame(
    [
        [2.5, 7002, 7002, 8.5, 1, 1, 0],
        [1.2, 10018, 10167, 6.75, 2, 2, 0],
    ],
    columns=[
        "trip_distance",
        "pickup_zip",
        "dropoff_zip",
        "mean_fare_window_1h_pickup_zip",
        "count_trips_window_1h_pickup_zip",
        "count_trips_window_30m_dropoff_zip",
        "dropoff_is_weekend",
    ],
    dtype="float64"  # cast everything float; OR specify per-column below
)

# ------------------------------------------------------------------
# Build a *dummy* output DataFrame just to define the schema.
# We don't need to run the real model for signature inference.
# ------------------------------------------------------------------
sample_out_df = pd.DataFrame({
    "pred_a": [0.0, 0.0],
    "pred_b": [0.0, 0.0],
})

# ------------------------------------------------------------------
# Log custom model using model_path
# ------------------------------------------------------------------
def get_signature() -> ModelSignature: return  infer_signature(model_input=sample_df, model_output=sample_out_df)