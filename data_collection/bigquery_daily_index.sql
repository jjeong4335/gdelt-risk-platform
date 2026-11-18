-- STEP 1 of the data-driven rebuild: daily Geo-Tension inputs, 2016 -> today,
-- computed inside BigQuery so only ~3,900 rows (one per day) come back.
--
-- Fixes vs the original course pipeline (data_collection/gdelt_download.sh):
--   * Keywords are matched against the article's THEMES (GKG "Themes" column).
--     The original awk script matched column 7, which is V2Counts (event
--     counts such as PROTEST or KILL), so military and sanction keywords
--     almost never matched and every day's category came out the same.
--   * Keywords are matched as whole theme tokens (MILITARY, TAX_WEAPONS_...),
--     so WAR no longer matches AWARD, SOFTWARE or WARNING.
--   * "International" is decided from the article's LOCATIONS (at least one
--     non-US country code), which is where GDELT stores country codes.
--   * V1 Themes/Locations are used instead of the V2 versions: same values
--     without character offsets, so the query reads far fewer bytes.
-- Unchanged: negative tone = 3rd value of V2Tone, kept if 0..50.
--
-- HOW TO RUN: paste into the BigQuery console. Before clicking Run, read
-- "This query will process N GB" in the top-right corner (free up to 1 TB a
-- month). Save the result as CSV -> ~/dashboard_data_rebuilt/gdelt_daily.csv

WITH raw_rows AS (
  SELECT
    PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS day,
    SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(2)] AS FLOAT64) AS neg,
    CASE
      WHEN REGEXP_CONTAINS(Themes, r'(^|[;_])(MILITARY|ARMEDCONFLICT|WAR|WEAPONS?|MISSILES?|INVASION|AIRSTRIKES?)([;_,]|$)') THEN 'CONFLICT'
      WHEN REGEXP_CONTAINS(Themes, r'(^|[;_])(SANCTIONS?|EMBARGO|TARIFFS?|EXPORT_CONTROLS?)([;_,]|$)') THEN 'ECONOMIC_PRESSURE'
      WHEN REGEXP_CONTAINS(Themes, r'(^|[;_])(COUP|PROTESTS?|RIOTS?)([;_,]|$)') THEN 'POLITICAL_INSTABILITY'
      WHEN REGEXP_CONTAINS(Themes, r'(^|[;_])(DIPLOMATIC|DIPLOMACY|EXPULSION)([;_,]|$)') THEN 'DIPLOMATIC_TENSION'
      WHEN REGEXP_CONTAINS(Themes, r'(^|[;_])(CEASEFIRE)([;_,]|$)') THEN 'DEESCALATION'
      WHEN REGEXP_CONTAINS(Themes, r'(^|[;_])(ESPIONAGE|CYBERATTACK|CYBER_ATTACK)([;_,]|$)') THEN 'INTELLIGENCE_CYBER'
      WHEN REGEXP_CONTAINS(Themes, r'(^|[;_])(NUCLEAR|BLOCKADE|ANNEXATION)([;_,]|$)') THEN 'SPECIAL'
    END AS category,
    -- Locations entries look like "1#Russia#RS#RS#55.75#37.61#-1;" (field 3 = country code).
    EXISTS(
      SELECT 1 FROM UNNEST(SPLIT(Locations, ';')) AS loc
      WHERE SPLIT(loc, '#')[SAFE_OFFSET(2)] NOT IN ('', 'US')
    ) AS international
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE _PARTITIONTIME >= TIMESTAMP('2016-01-01')
),
kept AS (
  SELECT day, neg, category
  FROM raw_rows
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
