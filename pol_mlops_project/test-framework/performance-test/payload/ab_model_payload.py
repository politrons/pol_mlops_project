import random
from typing import Any, Dict

from .payload_contract import PayloadContract

class PlainABModel(PayloadContract):
    def get_payload(self) -> Dict[str, Any]:
        return {
            "dataframe_split": {
                "columns": [
                    "trip_distance",
                    "pickup_zip",
                    "dropoff_zip",
                    "mean_fare_window_1h_pickup_zip",
                    "count_trips_window_1h_pickup_zip",
                    "count_trips_window_30m_dropoff_zip",
                    "dropoff_is_weekend"
                ],
                "data": [
                    [round(random.uniform(1, 10), 2), 7002, 7002, 8.5, 1, 1, 0]
                ]
            }
        }

payload_contract = PlainABModel()