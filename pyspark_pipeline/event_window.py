from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("EventWindowAggregation") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# ── 1. Load spike events ──────────────────────────────────────────
spike_events = spark.read.parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/spike_events/"
).select("date", "geo_tension_index", "dominant_category") \
 .withColumnRenamed("date", "spike_date")

print(f"Total spike events: {spike_events.count()}")

# ── 2. Load S&P 500 price data ────────────────────────────────────
sp500 = spark.read \
    .option("mergeSchema", "true") \
    .option("datetimeRebaseMode", "CORRECTED") \
    .option("int96RebaseMode", "CORRECTED") \
    .parquet(
        "hdfs:///user/jj4335_nyu_edu/gdelt_project/sp500_converted/sp500_2016.parquet",
        "hdfs:///user/jj4335_nyu_edu/gdelt_project/sp500_converted/sp500_2017.parquet",
        "hdfs:///user/jj4335_nyu_edu/gdelt_project/sp500_converted/sp500_2018.parquet",
        "hdfs:///user/jj4335_nyu_edu/gdelt_project/sp500_converted/sp500_2019.parquet",
        "hdfs:///user/jj4335_nyu_edu/gdelt_project/sp500_converted/sp500_2020.parquet",
        "hdfs:///user/jj4335_nyu_edu/gdelt_project/sp500_converted/sp500_2021.parquet",
        "hdfs:///user/jj4335_nyu_edu/gdelt_project/sp500_converted/sp500_2022.parquet",
        "hdfs:///user/jj4335_nyu_edu/gdelt_project/sp500_converted/sp500_2023.parquet",
        "hdfs:///user/jj4335_nyu_edu/gdelt_project/sp500_converted/sp500_2024.parquet",
        "hdfs:///user/jj4335_nyu_edu/gdelt_project/sp500_converted/sp500_2025.parquet",
        "hdfs:///user/jj4335_nyu_edu/gdelt_project/sp500_converted/sp500_2026.parquet"
    )

ticker_cols = [c for c in sp500.columns if c != "Date"]

sp500_long = sp500.select(
    F.to_date(
        (F.col("Date") / 1e9).cast("timestamp")
    ).alias("date"),
    F.explode(F.create_map(*[
        x for ticker in ticker_cols
        for x in (F.lit(ticker), F.col(f"`{ticker}`"))
    ])).alias("ticker", "close")
).filter(F.col("close").isNotNull() & F.col("date").isNotNull())

print(f"Total S&P 500 rows: {sp500_long.count()}")

# ── 3. Join spike events with price data (±30 day window) ─────────
joined = sp500_long.alias("prices").join(
    spike_events.alias("spikes"),
    F.datediff(
        F.col("prices.date"),
        F.col("spikes.spike_date")
    ).between(-30, 30)
).select(
    F.col("spikes.spike_date"),
    F.col("spikes.geo_tension_index"),
    F.col("spikes.dominant_category"),
    F.col("prices.date").alias("price_date"),
    F.col("prices.ticker"),
    F.col("prices.close"),
    F.datediff(
        F.col("prices.date"),
        F.col("spikes.spike_date")
    ).alias("days_from_spike")
)

# ── 4. Compute return relative to spike day (day 0) ──────────────
price_day0 = joined.filter(F.col("days_from_spike") == 0) \
    .select("spike_date", "ticker", F.col("close").alias("close_day0"))

joined_with_base = joined.join(
    price_day0,
    on=["spike_date", "ticker"],
    how="inner"
).withColumn(
    "return_pct",
    ((F.col("close") - F.col("close_day0")) / F.col("close_day0")) * 100
)

# ── 5. Aggregate per spike_date + ticker + days_from_spike ────────
# Keep spike_date so dashboard can filter by event
ticker_reaction_by_spike = joined_with_base.groupBy(
    "spike_date", "dominant_category", "ticker", "days_from_spike"
).agg(
    F.avg("return_pct").alias("avg_return_pct"),
    F.count("*").alias("num_obs")
).orderBy("spike_date", "ticker", "days_from_spike")

# ── 6. Overall summary per ticker (across all spikes) ────────────
ticker_summary = joined_with_base.groupBy("ticker").agg(
    F.avg(
        F.when(F.col("days_from_spike") == 5, F.col("return_pct"))
    ).alias("avg_return_5d"),
    F.avg(
        F.when(F.col("days_from_spike") == 30, F.col("return_pct"))
    ).alias("avg_return_30d"),
    F.min("return_pct").alias("worst_drawdown"),
    F.count("spike_date").alias("num_events")
)

# ── 7. Overall reaction curve (across all spikes) ─────────────────
ticker_reaction = joined_with_base.groupBy(
    "ticker", "days_from_spike"
).agg(
    F.avg("return_pct").alias("avg_return_pct"),
    F.count("spike_date").alias("num_events")
).orderBy("ticker", "days_from_spike")

print("Sample spike-level reaction:")
ticker_reaction_by_spike.show(10)

# ── 8. Save results ───────────────────────────────────────────────
ticker_reaction_by_spike.write.mode("overwrite").parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/ticker_reaction_by_spike/"
)

ticker_reaction.write.mode("overwrite").parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/ticker_reaction/"
)

ticker_summary.write.mode("overwrite").parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/ticker_summary/"
)

print("Done!")
spark.stop()
