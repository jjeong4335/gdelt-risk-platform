from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("EventWindowAggregation") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# ── 1. Load spike events ──────────────────────────────────────────
spike_events = spark.read.parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/spike_events/"
).select("date", "geo_tension_index") \
 .withColumnRenamed("date", "spike_date")

print(f"Total spike events: {spike_events.count()}")

# ── 2. Load S&P 500 price data ────────────────────────────────────
# Read all yearly parquet files
sp500 = spark.read.parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/sp500/"
)

# Melt wide format (date x tickers) to long format (date, ticker, close)
# sp500 columns: date index + ticker columns
ticker_cols = [c for c in sp500.columns if c != "Date" and c != "__index_level_0__"]

# Stack all ticker columns into long format
sp500_long = sp500.select(
    F.col("Date").alias("date"),
    F.explode(F.create_map(*[
        x for ticker in ticker_cols
        for x in (F.lit(ticker), F.col(f"`{ticker}`"))
    ])).alias("ticker", "close")
).filter(F.col("close").isNotNull())

sp500_long = sp500_long.withColumn("date", F.to_date(F.col("date")))

print(f"Total S&P 500 rows: {sp500_long.count()}")
sp500_long.show(5)

# ── 3. For each spike event, compute ±30-day returns ─────────────
# Cross join spike events with price data
# Then filter for ±30 day window

joined = sp500_long.alias("prices").join(
    spike_events.alias("spikes"),
    F.datediff(
        F.col("prices.date"),
        F.col("spikes.spike_date")
    ).between(-30, 30)
).select(
    F.col("spikes.spike_date"),
    F.col("spikes.geo_tension_index"),
    F.col("prices.date").alias("price_date"),
    F.col("prices.ticker"),
    F.col("prices.close"),
    F.datediff(
        F.col("prices.date"),
        F.col("spikes.spike_date")
    ).alias("days_from_spike")
)

# ── 4. Compute return relative to spike day (day 0) ──────────────
# Get price on spike day (day 0)
price_day0 = joined.filter(F.col("days_from_spike") == 0) \
    .select("spike_date", "ticker", F.col("close").alias("close_day0"))

# Join back to get relative return
joined_with_base = joined.join(
    price_day0,
    on=["spike_date", "ticker"],
    how="inner"
).withColumn(
    "return_pct",
    ((F.col("close") - F.col("close_day0")) / F.col("close_day0")) * 100
)

# ── 5. Aggregate: avg return per ticker per days_from_spike ──────
ticker_reaction = joined_with_base.groupBy(
    "ticker", "days_from_spike"
).agg(
    F.avg("return_pct").alias("avg_return_pct"),
    F.count("spike_date").alias("num_events")
).orderBy("ticker", "days_from_spike")

# ── 6. Key metrics per ticker ─────────────────────────────────────
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

print("Top 10 tickers by 5-day return:")
ticker_summary.orderBy(F.col("avg_return_5d").desc()).show(10)

print("Top 10 tickers by worst drawdown:")
ticker_summary.orderBy(F.col("worst_drawdown").asc()).show(10)

# ── 7. Save results ───────────────────────────────────────────────
ticker_reaction.write.mode("overwrite").parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/ticker_reaction/"
)

ticker_summary.write.mode("overwrite").parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/ticker_summary/"
)

print("Done!")
spark.stop()
