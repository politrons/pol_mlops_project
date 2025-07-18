-- Billing  Dashboard
-- --------------------

-- Daily Cost of Model Serving Serverless
WITH d AS (
  SELECT
      DATE(usage_start_time)        AS usage_date,
      SUM(usage_quantity)           AS dbus
  FROM  system.billing.usage
  WHERE sku_name LIKE '%SERVERLESS_REAL_TIME_INFERENCE%'
    AND usage_start_time >= DATEADD(day,-30,CURRENT_DATE())
  GROUP BY DATE(usage_start_time)
)
SELECT
    usage_date,
    dbus,
    dbus * 0.22                    AS cost_usd
FROM d
ORDER BY usage_date;


-- Monthly Cost Analysis for Serverless Inference
SELECT
    DATE_TRUNC('month', usage_start_time)          AS usage_month,
    SUM(usage_quantity)                            AS dbus,
    SUM(usage_quantity) * 0.22                     AS cost_usd
FROM  system.billing.usage
WHERE sku_name LIKE '%SERVERLESS_REAL_TIME_INFERENCE%'
  AND usage_start_time >= DATEADD(month, -12, CURRENT_DATE())
GROUP BY usage_month
ORDER BY usage_month;


-- Endpoints cost
SELECT
    DATE_TRUNC('month', usage_start_time)              AS usage_month,
    usage_metadata.endpoint_name                       AS endpoint,
    SUM(usage_quantity) * 0.22                         AS cost_usd
FROM  system.billing.usage
WHERE sku_name LIKE '%SERVERLESS_REAL_TIME_INFERENCE%'
  AND usage_start_time >= DATEADD(month,-12,CURRENT_DATE())
GROUP BY usage_month, endpoint
ORDER BY usage_month, cost_usd DESC;
