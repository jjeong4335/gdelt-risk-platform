from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("GeoTensionIndex") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED") \
    .getOrCreate()

# ── 1. Load GDELT filtered data ──────────────────────────────────
schema = StructType([
    StructField("record_id", StringType()),
    StructField("date",      StringType()),
    StructField("source",    StringType()),
    StructField("url",       StringType()),
    StructField("themes",    StringType()),
    StructField("tone",      StringType()),
    StructField("category",  StringType()),
])

df = spark.read.csv(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/gdelt/gdelt_filtered_full.tsv",
    sep="\t", schema=schema
)

# ── 2. Parse date ─────────────────────────────────────────────────
df = df.withColumn(
    "date",
    F.to_date(
        F.when(F.length(F.col("date")) >= 8, F.col("date").substr(1, 8)),
        "yyyyMMdd"
    )
).filter(F.col("date").isNotNull())

# ── 3. Filter international articles only ─────────────────────────
# Keep articles mentioning at least one non-US country code
# e.g. #RS# (Russia), #UP# (Ukraine), #IR# (Iran)
df = df.filter(F.col("themes").rlike("#(?!US)[A-Z]{2}#"))

# ── 4. Parse tone ─────────────────────────────────────────────────
df = df.withColumn(
    "neg_score",
    F.split(F.col("tone"), ",").getItem(2).cast(DoubleType())
).filter(
    F.col("neg_score").isNotNull() &
    (F.col("neg_score") >= 0) &
    (F.col("neg_score") <= 50)
)

print(f"Total valid rows: {df.count()}")

# ── 5. Daily Geo-Tension Index ────────────────────────────────────
# Formula: avg(negative_tone) x log(article_count + 1)
daily = df.groupBy("date").agg(
    F.count("*").alias("total_events"),
    F.avg("neg_score").alias("avg_negative_tone"),
).withColumn(
    "geo_tension_raw",
    F.col("avg_negative_tone") * F.log(F.col("total_events") + 1)
)

# ── 6. Normalize to 0-10 scale (p1 - p99) ────────────────────────
p1, p99 = daily.approxQuantile("geo_tension_raw", [0.01, 0.99], 0.01)
print(f"p01={p1:.4f}, p99={p99:.4f}")

daily = daily.withColumn(
    "geo_tension_index",
    F.greatest(
        F.lit(0.0),
        F.least(
            F.lit(10.0),
            (F.col("geo_tension_raw") - p1) / (p99 - p1) * 10
        )
    )
)

# ── 7. Dominant category per day ─────────────────────────────────
cat_daily = df.groupBy("date", "category").agg(
    F.count("*").alias("cat_count")
)
window_cat = Window.partitionBy("date").orderBy(F.col("cat_count").desc())
dominant_category = cat_daily \
    .withColumn("rank", F.rank().over(window_cat)) \
    .filter(F.col("rank") == 1) \
    .select("date", "category") \
    .withColumnRenamed("category", "dominant_category")

daily = daily.join(dominant_category, on="date", how="left")

# ── 8. Detect spike events (mean + 3*std per year) ───────────────
year_stats = daily.withColumn("year", F.year("date")) \
    .groupBy("year").agg(
        F.mean("geo_tension_index").alias("year_mean"),
        F.stddev("geo_tension_index").alias("year_std")
    )

daily_with_year = daily \
    .withColumn("year", F.year("date")) \
    .join(year_stats, on="year", how="left") \
    .withColumn("threshold", F.col("year_mean") + 3 * F.col("year_std"))

spike_events = daily_with_year.filter(
    F.col("geo_tension_index") > F.col("threshold")
).withColumn("is_spike", F.lit(True)) \
 .drop("year", "year_mean", "year_std")

print(f"Total spike events: {spike_events.count()}")
spike_events.orderBy(F.col("geo_tension_index").desc()).show(10)

# ── 9. Save to HDFS ──────────────────────────────────────────────
daily.select("date", "geo_tension_index", "total_events", "dominant_category") \
     .write.mode("overwrite").parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/geo_tension_index/"
)

spike_events.select("date", "geo_tension_index", "total_events", "threshold", "is_spike", "dominant_category") \
            .write.mode("overwrite").parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/spike_events/"
)

print("Done!")
spark.stop()
