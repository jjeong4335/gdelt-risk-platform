-- STEP 3 of the data-driven rebuild: news for each automatically detected event.
-- This is a TEMPLATE: data_collection/build_history.py fills in the event
-- windows and writes the ready-to-run query to ~/dashboard_data_rebuilt/event_news.sql.
--
-- For every event (peak day +- 3 days) it returns, computed in BigQuery:
--   * how many articles touch each category (an article can have several)
--   * up to {{ARTICLES_PER_EVENT}} matching articles (closest to the peak first) with domain and URL
--   * per-category, per-country and per-theme article counts on the core days
--     (peak -1..+1); build_events.py compares them across events to name each one
-- Same article filter as bigquery_daily_index.sql (theme keywords, international, tone 0..50).
-- Only the listed days' partitions are read, so this costs much less than step 1.
-- Save the result as CSV -> ~/dashboard_data_rebuilt/event_news.csv

WITH events AS (
  SELECT * FROM UNNEST([
{{EVENTS}}
  ])
),
arts AS (
  SELECT
    PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS article_date,
    SourceCommonName AS domain,
    DocumentIdentifier AS url,
    Themes AS themes,
    Locations AS locations,
    SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(2)] AS FLOAT64) AS neg,
    -- An article can belong to several categories (multi-label).
    ARRAY(SELECT c FROM UNNEST([
      IF(REGEXP_CONTAINS(Themes, r'(^|[;_])(MILITARY|ARMEDCONFLICT|WAR|WEAPONS?|MISSILES?|INVASION|AIRSTRIKES?)([;_,]|$)'), 'CONFLICT', NULL),
      IF(REGEXP_CONTAINS(Themes, r'(^|[;_])(SANCTIONS?|EMBARGO|TARIFFS?|EXPORT_CONTROLS?)([;_,]|$)'), 'ECONOMIC_PRESSURE', NULL),
      IF(REGEXP_CONTAINS(Themes, r'(^|[;_])(COUP|PROTESTS?|RIOTS?)([;_,]|$)'), 'POLITICAL_INSTABILITY', NULL),
      IF(REGEXP_CONTAINS(Themes, r'(^|[;_])(DIPLOMATIC|DIPLOMACY|EXPULSION)([;_,]|$)'), 'DIPLOMATIC_TENSION', NULL),
      IF(REGEXP_CONTAINS(Themes, r'(^|[;_])(CEASEFIRE)([;_,]|$)'), 'DEESCALATION', NULL),
      IF(REGEXP_CONTAINS(Themes, r'(^|[;_])(ESPIONAGE|CYBERATTACK|CYBER_ATTACK)([;_,]|$)'), 'INTELLIGENCE_CYBER', NULL),
      IF(REGEXP_CONTAINS(Themes, r'(^|[;_])(NUCLEAR|BLOCKADE|ANNEXATION)([;_,]|$)'), 'SPECIAL', NULL),
      -- naming-only tag: not used to decide whether an article is kept (same filter as step 1)
      IF(REGEXP_CONTAINS(Themes, r'(^|[;_])(TERROR|TERRORISM|SUICIDE_ATTACK)([;_,]|$)'), 'TERRORISM', NULL)
    ]) AS c WHERE c IS NOT NULL) AS cats
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE _PARTITIONTIME IN (
    {{PARTITIONS}}
  )
),
matched AS (
  SELECT e.event_date, a.*, DATE_DIFF(a.article_date, e.event_date, DAY) AS days_from_spike
  FROM arts AS a
  JOIN events AS e ON a.article_date BETWEEN e.start_day AND e.end_day
  WHERE EXISTS(SELECT 1 FROM UNNEST(a.cats) AS c WHERE c != 'TERRORISM')
    AND a.neg BETWEEN 0 AND 50
    AND EXISTS(SELECT 1 FROM UNNEST(SPLIT(a.locations, ';')) AS loc
               WHERE SPLIT(loc, '#')[SAFE_OFFSET(2)] NOT IN ('', 'US'))
),
window_size AS (
  SELECT event_date, COUNT(*) AS articles_in_window FROM matched GROUP BY event_date
),
core_size AS (
  SELECT event_date, COUNT(*) AS articles_core FROM matched WHERE ABS(days_from_spike) <= 1 GROUP BY event_date
),
cat_counts AS (
  -- how many of the event's core-day articles (peak -1..+1) touch each category
  SELECT event_date, STRING_AGG(CONCAT(c, ':', CAST(n AS STRING)), '|' ORDER BY n DESC) AS category_counts
  FROM (SELECT event_date, c, COUNT(*) AS n FROM matched, UNNEST(cats) AS c
        WHERE ABS(days_from_spike) <= 1 GROUP BY event_date, c)
  GROUP BY event_date
),
countries AS (
  SELECT event_date,
         STRING_AGG(CONCAT(code, ':', CAST(n AS STRING)), '|' ORDER BY n DESC LIMIT 40) AS top_countries
  FROM (
    SELECT event_date, code, COUNT(*) AS n
    FROM (
      SELECT m.event_date, code
      FROM matched AS m,
           UNNEST(ARRAY(SELECT DISTINCT SPLIT(loc, '#')[SAFE_OFFSET(2)]
                        FROM UNNEST(SPLIT(m.locations, ';')) AS loc)) AS code
      WHERE code IS NOT NULL AND code NOT IN ('', 'US') AND ABS(m.days_from_spike) <= 1
    )
    GROUP BY event_date, code
  )
  GROUP BY event_date
),
themes AS (
  SELECT event_date,
         STRING_AGG(CONCAT(theme, ':', CAST(n AS STRING)), '|' ORDER BY n DESC LIMIT 80) AS top_themes
  FROM (
    SELECT event_date, theme, COUNT(*) AS n
    FROM (
      SELECT m.event_date, theme
      FROM matched AS m,
           UNNEST(ARRAY(SELECT DISTINCT t FROM UNNEST(SPLIT(m.themes, ';')) AS t WHERE t != '')) AS theme
      WHERE ABS(m.days_from_spike) <= 1
    )
    GROUP BY event_date, theme
  )
  GROUP BY event_date
),
sample AS (
  SELECT * EXCEPT (dup, rk)
  FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY event_date ORDER BY ABS(days_from_spike), neg DESC, url) AS rk
    FROM (
      SELECT event_date, article_date, days_from_spike, domain, url, ARRAY_TO_STRING(cats, ',') AS category, neg,
             ROW_NUMBER() OVER (PARTITION BY event_date, url ORDER BY ABS(days_from_spike)) AS dup
      FROM matched
    )
    WHERE dup = 1
  )
  WHERE rk <= {{ARTICLES_PER_EVENT}}
)
SELECT s.event_date, s.article_date, s.days_from_spike, s.domain, s.url, s.category, s.neg,
       w.articles_in_window, z.articles_core, k.category_counts, c.top_countries, t.top_themes
FROM sample AS s
LEFT JOIN window_size AS w USING (event_date)
LEFT JOIN core_size AS z USING (event_date)
LEFT JOIN cat_counts AS k USING (event_date)
LEFT JOIN countries AS c USING (event_date)
LEFT JOIN themes AS t USING (event_date)
ORDER BY s.event_date, ABS(s.days_from_spike), s.neg DESC
