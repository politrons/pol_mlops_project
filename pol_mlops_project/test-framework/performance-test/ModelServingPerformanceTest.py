# Databricks notebook source
# ----------------------------------------------------------------------------
# Locust Ultra-Min Notebook (v6) – Minimal smoke/load from Databricks
# ----------------------------------------------------------------------------
# What this does:
#   * Install Locust
#   * Configure host, endpoint, token, payload
#   * Run in-notebook Locust using FastHttpUser
#   * Target configurable TPS via constant pacing (interval = 1 / tps_target)
#   * NO warmup, NO external files, NO range validation
#   * Success criteria: HTTP 200 AND response JSON contains at least one prediction
# ----------------------------------------------------------------------------
# Re-run the final cell with different tps_target/duration to simulate cold vs warm.
# ----------------------------------------------------------------------------

# COMMAND ----------
# MAGIC %pip install -q locust geventhttpclient
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# MUST RUN FIRST after restart: patch BEFORE anything else touches ssl/urllib3
import gevent.monkey

gevent.monkey.patch_all()

# Now it is safe to import locust & friends
from locust import FastHttpUser, task, constant_pacing
from locust.env import Environment
import gevent  # safe after patch

print("gevent patched. Locust imported cleanly.")

from typing import Any, Dict

DATABRICKS_HOST = "https://adb-3644846982999534.14.azuredatabricks.net"  # <-- change
ENDPOINT_NAME = "pol_endpoint"  # <-- change

TOKEN = dbutils.secrets.get("my-scope", "databricks-token")

# Relative path used by Locust
INVOKE_PATH = f"/serving-endpoints/{ENDPOINT_NAME}/invocations"

# Sample payload (change as needed)
PAYLOAD: Dict[str, Any] = {
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
            [2.5, 7002, 7002, 8.5, 1, 1, 0]
        ]
    }
}

print("Target:", DATABRICKS_HOST + INVOKE_PATH)


# COMMAND ----------
# MAGIC %md
# MAGIC ## Minimal prediction check
# MAGIC We only care that JSON contains at least one numeric prediction.


# COMMAND ----------

def run_locust(host: str,
               path: str,
               token: str,
               payload: Dict[str, Any],
               tps_target: float = 1.0,
               duration_s: int = 60,
               verbose: bool = False):
    """
    Run a minimal Locust load for `duration_s` seconds.
    Pacing interval = 1 / tps_target (seconds). If avg resp > interval, observed TPS will be lower.
    Returns (summary_dict, env).
    """
    if tps_target <= 0:
        raise ValueError("tps_target must be > 0")
    interval = 1.0 / tps_target

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    def _task_fn(self):
        with self.client.post(
                path,
                json=payload,
                headers=headers,
                name="invoke_model",
                catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            try:
                js = resp.json()
                print("response:", js)
            except Exception as e:
                resp.failure(f"JSON error: {e}")
                return
            resp.success()

    # Dynamic user class so we can re-run in notebook
    ModelUser = type(
        "NotebookModelUser",
        (FastHttpUser,),
        {
            "host": host,
            "wait_time": constant_pacing(interval),
            "invoke_model": task(_task_fn),
        },
    )

    env = Environment(user_classes=[ModelUser])
    env.create_local_runner()

    if verbose:
        from locust.stats import stats_printer, stats_history
        gevent.spawn(stats_printer, env.stats)
        gevent.spawn(stats_history, env.runner)

    env.runner.start(user_count=1, spawn_rate=1)
    gevent.sleep(duration_s)
    env.runner.quit()
    env.runner.greenlet.join()

    s = env.stats.total
    summary = {
        "requests": s.num_requests,
        "failures": s.num_failures,
        "avg_ms": round(s.avg_response_time, 2),
        "p50_ms": round(s.median_response_time or 0.0, 2),
        "p95_ms": round(s.get_response_time_percentile(0.95), 2),
        "p99_ms": round(s.get_response_time_percentile(0.99), 2),
        "min_ms": round(s.min_response_time or 0.0, 2),
        "max_ms": round(s.max_response_time or 0.0, 2),
        "reqs_per_s_observed": round((s.num_requests / duration_s) if duration_s else 0.0, 3),
        "tps_target": tps_target,
    }
    return summary, env


# COMMAND ----------
# MAGIC %md
# MAGIC ## Run Example (~1 TPS · 30s)
# MAGIC Change tps_target / duration_s / check_prediction and re-run.

# COMMAND ----------
summary, env = run_locust(
    host=DATABRICKS_HOST,
    path=INVOKE_PATH,
    token=TOKEN,
    payload=PAYLOAD,
    tps_target=10.0,
    duration_s=300,
    verbose=False,
)
summary

# COMMAND ----------
# MAGIC %md
# MAGIC ## Quick reruns
# MAGIC * Cold vs hot: run above cell twice.
# MAGIC * Higher rate: set tps_target=5.
# MAGIC * No JSON check: set check_prediction=False.

# COMMAND ----------
print("Ready. Re-run the previous cell with your parameters.")
