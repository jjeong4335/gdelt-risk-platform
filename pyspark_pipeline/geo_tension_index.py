from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("GeoTensionIndex") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
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
    sep="\t",
    schema=schema
)

# ── 2. Parse date safely ──────────────────────────────────────────
# Some rows have corrupted date strings (e.g. '2016031#') due to gzip errors
df = df.withColumn(
    "date",
    F.to_date(
        F.when(
            F.length(F.col("date")) >= 8,
            F.col("date").substr(1, 8)
        ),
        "yyyyMMdd"
    )
).filter(F.col("date").isNotNull())

# ── 3. Parse tone score ───────────────────────────────────────────
# GDELT tone field: avg_tone, pos_score, neg_score, polarity, ...
df = df.withColumn(
    "avg_tone",
    F.split(F.col("tone"), ",").getItem(0).cast(DoubleType())
).withColumn(
    "neg_score",
    F.split(F.col("tone"), ",").getItem(2).cast(DoubleType())
).filter(
    F.col("avg_tone").isNotNull() & 
    F.col("neg_score").isNotNull() &
    (F.col("neg_score") >= 0) &
    (F.col("neg_score") <= 100)  # filter out corrupted rows with out-of-range values
)

print(f"Total valid rows: {df.count()}")

# ── 4. Daily Geo-Tension Index per category ───────────────────────
# tension_score = (avg_negativity / 100) * log(event_count + 1)
# neg_score is in percentage units, divide by 100 to normalize
geo_tension_by_category = df.groupBy("date", "category").agg(
    F.count("*").alias("event_count"),
    F.avg("neg_score").alias("avg_negativity"),
    F.avg("avg_tone").alias("avg_tone")
).withColumn(
    "tension_score",
    (F.col("avg_negativity") / 100) * F.log(F.col("event_count") + 1)
)

# ── 5. Overall daily tension index ───────────────────────────────
daily_tension = geo_tension_by_category.groupBy("date").agg(
    F.sum("tension_score").alias("geo_tension_index"),
    F.sum("event_count").alias("total_events")
).orderBy("date")

# ── 6. Detect spike events (threshold = mean + 3*std) ────────────
stats = daily_tension.agg(
    F.mean("geo_tension_index").alias("mean"),
    F.stddev("geo_tension_index").alias("std")
).collect()[0]

threshold = stats["mean"] + 3 * stats["std"]
print(f"Spike threshold: {threshold:.4f}")

spike_events = daily_tension.filter(
    F.col("geo_tension_index") > threshold
).withColumn("is_spike", F.lit(True))

print(f"Total spike events: {spike_events.count()}")
spike_events.show(20)

# ── 7. Save results to HDFS ───────────────────────────────────────
daily_tension.write.mode("overwrite").parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/geo_tension_index/"
)

geo_tension_by_category.write.mode("overwrite").parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/geo_tension_by_category/"
)

spike_events.write.mode("overwrite").parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/spike_events/"
)

print("Done!")
spark.stop()