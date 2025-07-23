# Model Card – LightGBM Regressor (NYC Taxi Fare)

| **Model name** | `{{ model_name }}` |
| -------------- | ------------------ |
| **Version**    | `v{{ model_version }}` |
| **Date**       | 2025‑07‑23 |
| **Owner**      | Analytics / MLOps Team |
| **Framework**  | LightGBM 3.x • MLflow 2.x |

---

## 1  Purpose  
Predict the **`fare_amount`** for a NYC yellow‑taxi ride given trip distance, pickup/drop‑off ZIP codes, and recent demand patterns.  
Typical downstream use‑cases:  
* Real‑time fare estimation in the mobile app  
* Revenue forecasting & surge‑pricing simulation  
* Ops dashboards (supply vs. demand)

## 2  Intended users  
Product analysts, data scientists, pricing‑ops engineers.

## 3  Training data  
| Aspect | Details |
| ------ | ------- |
| Source | Delta table ➜ `{{ training_data_path }}` |
| Time span | Jan 2024 – Jun 2025 |
| Rows (post‑filter) | **`{{ n_rows }}`** |
| Label | `fare_amount` *(float)* |
| Features (7) | `trip_distance`, `pickup_zip`, `dropoff_zip`, `mean_fare_window_1h_pickup_zip`, `count_trips_window_1h_pickup_zip`, `count_trips_window_30m_dropoff_zip`, `dropoff_is_weekend` |
| Pre‑processing | Timestamp rounding (15 / 30 min) + window aggregations (1 h & 30 min) |

## 4  Model architecture  
* **Algorithm**: Gradient Boosting Decision Trees (LightGBM)  
* Hyper‑parameters  

| Param | Value |
| ----- | ----- |
| `num_leaves` | 32 |
| `n_estimators` | 100 |
| `objective` | `regression` |
| `random_state` | 123 |

*(Tune with Optuna when data drift is detected.)*

## 5  Evaluation  

| Split | Metric | Score |
| ----- | ------ | ----- |
| Test (20 %) | RMSE | **`{{ rmse }}`** |
| Validation | — | — |

**Interpretation** Lower RMSE → better fare accuracy in \$. Error of ±`{{ rmse | round(2) }}` means the model’s average absolute deviation is ~±`{{ rmse | round(2) }}` USD.

## 6  Limitations & risks  
* Trained on historical NYC data only; may under‑perform on airports outside the city.  
* ZIP codes with < 50 rides/hour can yield noisy window features.  
* Does **not** model extreme traffic events (marathons, blizzards); fares could be underestimated.  

## 7  Ethical considerations  
No personally identifiable information (PII) is used.  
Potential socioeconomic bias if fare estimates influence driver allocation—monitor error by borough income level.

## 8  Maintenance plan  
* **Drift check** weekly: compare RMSE on rolling 7‑day window vs training RMSE.  
* **Re‑tune hyper‑parameters** with Optuna if RMSE ↑ > 15 %.  
* **Retrain** quarterly or upon major fare‑policy change.

## 9  References  
* LightGBM paper (Ke et al., 2017)  
* Mitchell et al., “Model Cards for Model Reporting”, 2019  
* Databricks MLflow Registry docs

---

*Generated automatically by the Databricks training template (run `{{ run_id }}`).*  
