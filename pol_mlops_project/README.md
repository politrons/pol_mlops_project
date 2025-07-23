# pol\_mlops\_project

This repository extends the Databricks MLOps Stacks template with a complete,
production‑grade pipeline that includes

* modular feature engineering and multiple model‑training workflows
* AutoML experiments
* medallion‑architecture data pipelines
* several model flavours (baseline, elastic‑net, random‑forest, nearest‑neighbour,
  Optuna‑optimised, etc.)
* A/B model orchestration (cascade, sequential and parallel)
* model‑serving endpoints
* unit, integration and performance tests
* optional monitoring jobs

The sample task is NYC taxi‑fare regression, but every component can be adapted
to your own use‑case.  Databricks documentation for MLOps Stacks:
[https://docs.databricks.com/dev-tools/bundles/mlops-stacks.html](https://docs.databricks.com/dev-tools/bundles/mlops-stacks.html).

---

## How to customise `databricks.yaml` before deploying the bundle

Only three edits are required, plus one shared inference cluster.

| What to change         | Location                                       | Notes                                                                                |
| ---------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------ |
| `host`                 | `targets.<env>.workspace.host`                 | Replace the example URL with the URL of **your** Databricks workspace.               |
| `catalog_name`         | `targets.<env>.variables.catalog_name`         | Use an existing Unity Catalog or create one, e.g. `my_dev`, `my_staging`, `my_prod`. |
| `inference_cluster_id` | `targets.<env>.variables.inference_cluster_id` | Create the cluster first (see below) and paste its ID here.                          |

### Creating the shared inference cluster

1. In the Databricks UI, open **Clusters → Create Cluster (Advanced)** and choose
   **Shared cluster**.

2. Name it, for example `pol-inference-shared`.

3. Runtime: `13.3 LTS (includes Apache Spark 3.4.1, Scala 2.12)` and tick
   **Machine Learning**.

4. Node type: `Standard_DS3_v2` (increase if required).

5. **Libraries → PIP**:

   ```
   typing_extensions==4.14.1
   databricks-feature-engineering==0.12.1
   ```

6. Save, wait until running, copy the **Cluster ID**.

7. Paste that ID into every `inference_cluster_id` entry in `databricks.yaml`.

After those edits, run:

```bash
databricks bundle deploy --target dev   # or staging / prod / test
```

---

## Table of contents

* [Project structure](#project-structure)
* [Sample pipeline configuration](#sample-pipeline-configuration)
* [Iterating on code](#iterating-on-code)
* [Running tests](#running-tests)

---

## Project structure

```
pol_mlops_project/                         Root (polyrepo/monorepo friendly)
│
├── databricks.yaml                        Top‑level bundle file
├── requirements.txt                       Python dependencies
│
├── feature_engineering/
│   ├── features/                          Individual feature modules
│   └── notebooks/
│       └── GenerateAndWriteFeatures.py    Driver notebook
│
├── training/
│   ├── training-model-fe-workflow/
│   ├── training-model-optune-workflow/
│   ├── training-model-medallion-workflow/
│   ├── training-plain-model-workflow/
│   ├── training-elastic-net-linear-model-workflow/
│   └── training-random-forest-model-workflow/
│
├── custom_model/
│   ├── custom-model-cascade_ab_plain/
│   ├── custom-model-orchestrate_ab/
│   └── custom-model-parallel-orchestrate_ab/
│
├── model_serving/
│   ├── deploy-model-serving-endpoint/
│   ├── deploy-ab-model-serving-endpoint/
│   ├── deploy-ab-plain-model-serving-endpoint/
│   ├── deploy-orchestrate-ab-model-serving-endpoint/
│   └── deploy-parallel-orchestrate-ab-model-serving-endpoint/
│
├── pipelines/
│   └── medallion-architecture/            Bronze → Silver → Gold
│
├── auto_ml/                               AutoML experiment resources
│
├── resources/                             Bundle resource YAMLs
│
├── monitoring/                            (Optional) drift / quality jobs
│
├── tests/
│   └── feature_engineering/               Unit tests per feature module
│
└── test-framework/
    ├── unit/                              Notebook‑level tests
    ├── integration/                       End‑to‑end smoke tests
    └── performance/                       Load / latency benchmarks
```

---

## Sample pipeline configuration

### 1. Feature engineering

* Implement or extend modules in `feature_engineering/features/`.
* Each module exposes
  `compute_features_fn(input_df, ts_col, start_end_ts) → feature_df`.
* The notebook `feature_engineering/notebooks/GenerateAndWriteFeatures.py`
  discovers, executes and writes these features to a time‑series Feature Store
  table.

### 2. Model training

* Training notebooks under `training/` build training sets with Feature Store and
  train / register MLflow models.
* Keep, replace or add notebooks for your own algorithms.
* Update the matching YAML in `resources/training/` to point at the correct path
  and parameters.

### 3. Model serving

Serving workflows under `model_serving/` deploy models to Databricks Model
Serving endpoints.  Options:

* single‑model endpoint
* A/B plain baseline
* sequential cascade
* parallel orchestration

Deploy, for example:

```bash
databricks bundle deploy --target dev --name deploy-ab-model-serving-endpoint
```

### 4. Medallion architecture pipeline

`resources/pipelines/medallion-architecture-resource.yaml` provisions jobs that
ingest raw data (Bronze), clean / standardise (Silver) and aggregate (Gold).
Edit notebook parameters and Delta paths for your datasets.

---

## Iterating on code

### On Databricks (Repos)

1. Enable Git integration, clone this repo into **Repos**.
2. Attach a cluster running ML Runtime 13.3 LTS.
3. Edit notebooks and Python files directly; run cells to test.
4. Commit and push as required.

### Locally

* Python 3.8+
* `pip install -I -r requirements.txt`
* Java 8+ (for local PySpark tests)

---

## Running tests

| Test type   | Location                      | How to run                                                               |
| ----------- | ----------------------------- | ------------------------------------------------------------------------ |
| Unit        | `tests/`                      | `pytest tests` (local) or CI workflow                                    |
| Integration | `test-framework/integration/` | Deploy bundle, then run the corresponding integration‑test job           |
| Performance | `test-framework/performance/` | As above, targeting performance‑test jobs; inspect MLflow and dashboards |

---

