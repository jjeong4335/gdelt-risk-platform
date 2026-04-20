"""
Extract news URLs and themes for each spike event (±3 days) from GDELT TSV.
Saves as parquet for dashboard use.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import re

spark = SparkSession.builder \
    .appName("SpikeNewsExtract") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# ── 1. Load spike events ──────────────────────────────────────────
spike_events = spark.read.parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/spike_events/"
).select("date", "dominant_category") \
 .withColumnRenamed("date", "spike_date")

print(f"Spike events: {spike_events.count()}")

# ── 2. Load GDELT TSV ─────────────────────────────────────────────
gdelt = spark.read.csv(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/gdelt/gdelt_filtered_full.tsv",
    sep="\t", header=False
).select(
    F.col("_c1").alias("datetime"),
    F.col("_c2").alias("domain"),
    F.col("_c3").alias("url"),
    F.col("_c4").alias("themes"),
    F.col("_c6").alias("category")
).filter(
    F.col("url").isNotNull() & F.col("datetime").isNotNull()
).withColumn(
    "date",
    F.to_date(F.col("datetime").substr(1, 8), "yyyyMMdd")
).filter(F.col("date").isNotNull())

print(f"GDELT rows: {gdelt.count()}")

# ── 3. Join: spike ±3 days ────────────────────────────────────────
joined = gdelt.alias("g").join(
    spike_events.alias("s"),
    F.datediff(F.col("g.date"), F.col("s.spike_date")).between(-3, 3)
).select(
    F.col("s.spike_date"),
    F.col("s.dominant_category"),
    F.col("g.date").alias("article_date"),
    F.col("g.domain"),
    F.col("g.url"),
    F.col("g.themes"),
    F.datediff(F.col("g.date"), F.col("s.spike_date")).alias("days_from_spike")
)

# ── 4. Extract slug title from URL ────────────────────────────────
@F.udf(StringType())
def url_to_title(url):
    """Extract readable title from URL slug."""
    try:
        parts = str(url).rstrip("/").split("/")
        slug = parts[-1].split(".")[0].split("?")[0]
        # Remove numbers and clean up
        title = re.sub(r'[-_]', ' ', slug)
        title = re.sub(r'\d+', '', title).strip()
        title = ' '.join(w.capitalize() for w in title.split() if len(w) > 2)
        return title if len(title) > 15 else None
    except:
        return None

joined = joined.withColumn("title", url_to_title(F.col("url"))) \
               .filter(F.col("title").isNotNull())

# ── 5. Sample top URLs per spike date ─────────────────────────────
from pyspark.sql.window import Window
window = Window.partitionBy("spike_date").orderBy(F.col("days_from_spike").asc())
result = joined.withColumn("rank", F.row_number().over(window)) \
               .filter(F.col("rank") <= 50) \
               .drop("rank")

print("Sample:")
result.show(5, truncate=80)

# ── 6. Save ───────────────────────────────────────────────────────
result.write.mode("overwrite").parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/spike_news/"
)

print("Done!")
spark.stop()
