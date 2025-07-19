-- Billing  Dashboard
-- --------------------

-- Daily Cost of Model Serving Serverless
-- Daily cost of Serverless Real-Time Inference for the last 30 days
WITH usage_filtered AS (           -- ① limit to the records you care about
    SELECT
        DATE(u.usage_start_time)        AS usage_date,
        u.usage_quantity,               -- DBUs in this record
        u.cloud,
        u.sku_name,
        u.usage_start_time,
        u.usage_end_time
    FROM system.billing.usage AS u
    WHERE u.sku_name      LIKE '%SERVERLESS_REAL_TIME_INFERENCE%'
      AND u.usage_unit     = 'DBU'                  -- ← only DBU lines
      AND u.usage_start_time >= DATEADD(day, -30, CURRENT_DATE())
),
price_lkp AS (                     -- bring in the list price valid for the record’s time span
    SELECT
        f.*,
        lp.pricing.default AS price_per_dbu         -- numeric list price
    FROM usage_filtered      AS f
    LEFT JOIN system.billing.list_prices AS lp      -- Databricks-recommended join
           ON  f.cloud            = lp.cloud
          AND f.sku_name          = lp.sku_name
          AND f.usage_start_time >= lp.price_start_time
          AND (f.usage_end_time  <= lp.price_end_time OR lp.price_end_time IS NULL)
)
SELECT                              -- ③ aggregate to daily totals
    usage_date,
    SUM(usage_quantity)                   AS dbus,
    SUM(usage_quantity * price_per_dbu)   AS cost_usd
FROM price_lkp
GROUP BY usage_date
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


-- Total cost for all products
-- ---------------------------

-- Flatten the list_prices JSON once
WITH prices_flat AS (
  SELECT
      account_id,
      sku_name,
      TO_TIMESTAMP(price_start_time)                                  AS price_start_ts,
      COALESCE(TO_TIMESTAMP(price_end_time),
               TO_TIMESTAMP('9999-12-31 23:59:59'))                   AS price_end_ts,
      pricing.effective_list.`default`                                AS unit_price
  FROM system.billing.list_prices
),

-- Optionally keep the most-recent definition for each SKU+account
latest_prices AS (
  SELECT *
  FROM (
      SELECT *,
             ROW_NUMBER() OVER (PARTITION BY account_id, sku_name
                                ORDER BY price_start_ts DESC) AS rn
      FROM prices_flat
  )
  WHERE rn = 1
),

-- Join usage to prices, then aggregate
usage_costs AS (
  SELECT
      u.sku_name,
      SUM(u.usage_quantity * p.unit_price)             AS total_cost
  FROM system.billing.usage            u
  JOIN latest_prices    p
    ON u.sku_name          = p.sku_name
    AND u.usage_start_time >= p.price_start_ts
    AND u.usage_start_time <  p.price_end_ts
  GROUP BY  u.sku_name
)

-- Final select (helpful if you want ordering or further filtering)
SELECT *
FROM usage_costs
ORDER BY  sku_name;


-- Endpoint cost
-- --------------

-- Monthly cost of Serverless Real-Time Inference by endpoint (last 12 months)
WITH usage_filtered AS (                     -- ① limit to the usage rows we need
    SELECT
        DATE_TRUNC('month', u.usage_start_time) AS usage_month,
        u.cloud,
        u.sku_name,
        CAST(u.usage_metadata.endpoint_name AS STRING) AS endpoint,   -- endpoint name
        u.usage_quantity,                       -- DBUs in this record
        u.usage_start_time,
        u.usage_end_time
    FROM system.billing.usage AS u
    WHERE u.sku_name      LIKE '%SERVERLESS_REAL_TIME_INFERENCE%'
      AND u.usage_unit     = 'DBU'                              -- only DBU lines
      AND u.usage_start_time >= DATEADD(month, -12, CURRENT_DATE())
),
price_lkp AS (                             -- pull in the price valid at that time
    SELECT
        uf.*,
        lp.pricing.default AS price_per_dbu      -- numeric list price (USD)
    FROM usage_filtered            AS uf
    LEFT JOIN system.billing.list_prices AS lp   -- Databricks-style join
           ON  uf.cloud            = lp.cloud
          AND uf.sku_name          = lp.sku_name
          AND uf.usage_start_time >= lp.price_start_time
          AND (uf.usage_end_time  <= lp.price_end_time OR lp.price_end_time IS NULL)
)
SELECT                                      -- aggregate to month × endpoint
    usage_month,
    endpoint,
    SUM(usage_quantity * price_per_dbu)       AS cost_usd
FROM price_lkp
GROUP BY usage_month, endpoint
ORDER BY usage_month, cost_usd DESC;

