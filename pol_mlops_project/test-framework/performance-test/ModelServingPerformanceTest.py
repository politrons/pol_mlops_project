# Databricks notebook source
# ----------------------------------------------------------------------------
# Performance test Locust
# ----------------------------------------------------------------------------
# What this does:
#   * Install Locust
#   * Configure host, endpoint, token, payload
#   * Run in-notebook Locust using FastHttpUser
#   * Target configurable TPS via constant pacing (interval = 1 / tps_target)
#   * Success criteria: HTTP 200 AND response JSON contains at least one prediction

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
from mlflow.utils.databricks_utils import dbutils

# Input request
dbutils.widgets.text("DATABRICKS_HOST", "https://adb-3644846982999534.14.azuredatabricks.net",
                     label="Databricks Host")

dbutils.widgets.text("ENDPOINT_NAME", "",
                     label="Endpoint Name")

dbutils.widgets.text("tps_target", "10.0",
                     label="tps_target Name")

dbutils.widgets.text("duration_s", "300",
                     label="duration_s Name")

dbutils.widgets.text("payload_path", "ab_model_payload",
                     label="payload_path")

DATABRICKS_HOST = dbutils.widgets.get("DATABRICKS_HOST")
ENDPOINT_NAME = dbutils.widgets.get("ENDPOINT_NAME")
INVOKE_PATH = f"/serving-endpoints/{ENDPOINT_NAME}/invocations"
TOKEN = dbutils.secrets.get("my-scope", "databricks-token")
tps_target = float(dbutils.widgets.get("tps_target"))
duration_s = int(dbutils.widgets.get("duration_s"))

# Assertions
dbutils.widgets.text("avg_ms", "avg_ms", label="avg_ms")
dbutils.widgets.text("p50_ms", "p50_ms", label="p50_ms")
dbutils.widgets.text("p95_ms", "p95_ms", label="p95_ms")
dbutils.widgets.text("p99_ms", "p99_ms", label="p99_ms")

avg_ms = float(dbutils.widgets.get("avg_ms"))
p50_ms = float(dbutils.widgets.get("p50_ms"))
p95_ms = float(dbutils.widgets.get("p95_ms"))
p99_ms = float(dbutils.widgets.get("p99_ms"))

# Payload

from importlib import import_module

mod = import_module("payload." + dbutils.widgets.get("payload_path"))

# Sample payload (change as needed)


print("Target:", DATABRICKS_HOST + INVOKE_PATH)


# COMMAND ----------
# MAGIC ## Run performance test.

def run_locust(host: str,
               path: str,
               token: str,
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
        payload = mod.get_payload()
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
                resp.json()
            except Exception as e:
                print("Error:", e)
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

    stats = env.stats.total
    summary = {
        "requests": stats.num_requests,
        "failures": stats.num_failures,
        "avg_ms": round(stats.avg_response_time, 2),
        "p50_ms": round(stats.median_response_time or 0.0, 2),
        "p95_ms": round(stats.get_response_time_percentile(0.95), 2),
        "p99_ms": round(stats.get_response_time_percentile(0.99), 2),
        "min_ms": round(stats.min_response_time or 0.0, 2),
        "max_ms": round(stats.max_response_time or 0.0, 2),
        "reqs_per_s_observed": round((stats.num_requests / duration_s) if duration_s else 0.0, 3),
        "tps_target": tps_target,
    }
    return summary, env


# COMMAND ----------
# MAGIC ## Run Simulation

summary, env = run_locust(
    host=DATABRICKS_HOST,
    path=INVOKE_PATH,
    token=TOKEN,
    tps_target=tps_target,
    duration_s=duration_s,
    verbose=False,
)
print(summary)

# ------------
# Assertions
# ------------
stats = env.stats.total
assert stats.avg_response_time <= avg_ms
assert stats.median_response_time <= p50_ms
assert stats.get_response_time_percentile(0.95) <= p95_ms
assert stats.get_response_time_percentile(0.99) <= p99_ms


