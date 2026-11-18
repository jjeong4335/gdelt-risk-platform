from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("PortfolioRiskEngine") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# ── 1. Load ticker summary (precomputed reaction patterns) ────────
ticker_summary = spark.read.parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/ticker_summary/"
)

# ── 2. Define sample portfolio ────────────────────────────────────
# Format: (ticker, weight_pct)
# Total weights must sum to 100
portfolio = [
    ("AAPL", 30.0),
    ("XOM",  50.0),
    ("LMT",  20.0),
]

portfolio_df = spark.createDataFrame(portfolio, ["ticker", "weight"])

# ── 3. Join portfolio with ticker reaction patterns ───────────────
portfolio_with_stats = portfolio_df.join(
    ticker_summary,
    on="ticker",
    how="left"
)

portfolio_with_stats.show()

# ── 4. Compute weighted risk summary ─────────────────────────────
# weighted_return = weight * avg_return / 100
risk_summary = portfolio_with_stats.agg(
    F.sum(
        F.col("weight") * F.col("avg_return_5d") / 100
    ).alias("weighted_avg_return_5d"),
    F.sum(
        F.col("weight") * F.col("avg_return_30d") / 100
    ).alias("weighted_avg_return_30d"),
    F.sum(
        F.col("weight") * F.col("worst_drawdown") / 100
    ).alias("weighted_worst_drawdown"),
)

print("\n=== Portfolio Risk Summary ===")
risk_summary.show()

# ── 5. Most exposed holding ───────────────────────────────────────
most_exposed = portfolio_with_stats \
    .withColumn("exposure_score", F.abs(F.col("worst_drawdown")) * F.col("weight") / 100) \
    .orderBy(F.col("exposure_score").desc()) \
    .select("ticker", "weight", "avg_return_5d", "worst_drawdown", "exposure_score")

print("=== Holdings by Exposure ===")
most_exposed.show()

# ── 6. Save risk summary ──────────────────────────────────────────
risk_summary.write.mode("overwrite").parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/portfolio_risk_summary/"
)

most_exposed.write.mode("overwrite").parquet(
    "hdfs:///user/jj4335_nyu_edu/gdelt_project/portfolio_exposure/"
)

print("Done!")
spark.stop()
