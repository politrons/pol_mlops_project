from databricks.sdk.service.serving import DataframeSplitInput


payload = DataframeSplitInput(
    columns=[
        "trip_distance",
        "pickup_zip",
        "dropoff_zip",
        "mean_fare_window_1h_pickup_zip",
        "count_trips_window_1h_pickup_zip",
        "count_trips_window_30m_dropoff_zip",
        "dropoff_is_weekend",
    ],
    data=[[2.5, 7002, 7002, 8.5, 1, 1, 0]],
)
