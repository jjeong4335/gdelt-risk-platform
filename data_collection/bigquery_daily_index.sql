-- Daily Geo-Tension inputs for 2016 → today, computed INSIDE BigQuery.
-- Only ~3,700 rows (one per day) come back, instead of downloading ~2 TB of
-- raw GDELT files and filtering them on a laptop.
--
-- Same rules as the original pipeline:
--   data_collection/gdelt_download.sh  : keyword filter + category, applied to
--                                        GKG column 7 (V2Counts), which the
--                                        original awk script used as "themes"
--   pyspark_pipeline/geo_tension_index.py :
--     * international only  = at least one non-US 2-letter country code (#RS#)
--     * negative tone       = 3rd value of V2Tone, kept if 0..50
--     * per day: article count, average negative tone, most common category
-- The 0-10 scaling, yearly thresholds and spike detection are done afterwards
-- in Python (data_collection/build_history.py), exactly as the Spark job did.
--
-- HOW TO RUN (no billing needed):
--   1. console.cloud.google.com/bigquery, in a project WITHOUT billing
--      (BigQuery sandbox: 1 TB of query processing free per month).
--   2. Paste this whole file into the editor. Before running, the top-right
--      corner shows "This query will process N GB when run". That is the cost
--      check: stay well under 1,000 GB (1 TB).
--   3. Run, then Save results -> CSV (local file). Save it as
--      ~/dashboard_data_rebuilt/gdelt_daily.csv

WITH rows AS (
  SELECT
    PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS day,
    SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(2)] AS FLOAT64) AS neg,
    CASE
      WHEN REGEXP_CONTAINS(V2Counts, r'MILITARY|WAR|WEAPON|MISSILE|INVASION|AIRSTRIKE') THEN 'CONFLICT'
      WHEN REGEXP_CONTAINS(V2Counts, r'SANCTION|EMBARGO|TARIFF|EXPORT_CONTROL') THEN 'ECONOMIC_PRESSURE'
      WHEN REGEXP_CONTAINS(V2Counts, r'COUP|PROTEST|RIOT') THEN 'POLITICAL_INSTABILITY'
      WHEN REGEXP_CONTAINS(V2Counts, r'DIPLOMATIC|EXPULSION') THEN 'DIPLOMATIC_TENSION'
      WHEN REGEXP_CONTAINS(V2Counts, r'CEASEFIRE') THEN 'DEESCALATION'
      WHEN REGEXP_CONTAINS(V2Counts, r'ESPIONAGE|CYBERATTACK') THEN 'INTELLIGENCE_CYBER'
      WHEN REGEXP_CONTAINS(V2Counts, r'NUCLEAR|BLOCKADE|ANNEXATION') THEN 'SPECIAL'
    END AS category,
    -- BigQuery regex has no look-ahead, so "#(?!US)[A-Z]{2}#" becomes:
    -- extract every 2-letter code and check that one of them isn't US.
    EXISTS(
      SELECT 1 FROM UNNEST(REGEXP_EXTRACT_ALL(V2Counts, r'#([A-Z]{2})#')) AS code
      WHERE code != 'US'
    ) AS international
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  -- Partition filter: BigQuery only reads these days (this is what keeps the cost down).
  WHERE _PARTITIONTIME >= TIMESTAMP('2016-01-01')
),
kept AS (
  SELECT day, neg, category
  FROM rows
  WHERE category IS NOT NULL AND international AND neg BETWEEN 0 AND 50
),
daily AS (
  SELECT day, COUNT(*) AS total_events, AVG(neg) AS avg_negative_tone
  FROM kept
  GROUP BY day
),
dominant AS (
  SELECT day, ARRAY_AGG(category ORDER BY n DESC, category LIMIT 1)[OFFSET(0)] AS dominant_category
  FROM (SELECT day, category, COUNT(*) AS n FROM kept GROUP BY day, category)
  GROUP BY day
)
SELECT d.day AS date, d.total_events, d.avg_negative_tone, m.dominant_category
FROM daily AS d
JOIN dominant AS m USING (day)
WHERE d.day >= DATE '2016-01-01'
ORDER BY date
